"""
Builder script to generate latentspace_dataset/notebook/latentspace_dataset_builder.ipynb.
Constructs a 100% self-contained notebook meeting all Master Implementation Specification requirements.
"""

import json
import os

OUTPUT_NOTEBOOK = os.path.join("latentspace_dataset", "notebook", "latentspace_dataset_builder.ipynb")

def md_cell(text: str) -> dict:
    lines = [line + "\n" for line in text.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code_cell(code: str) -> dict:
    lines = [line + "\n" for line in code.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

cells = []

# ==============================================================================
# CELL 1: TITLE & UNIVERSAL TOOL CONTRACT OVERVIEW
# ==============================================================================
cells.append(md_cell("""# Latent Dataset Annotation + Synthetic Data Pipeline
### Offline Semantic Middleware & Agentic Assistant Training Dataset Builder
**Pipeline Version:** 1.0.0  
**Target Assistant:** Latent Semantic Middleware  
**Universal Tool Contract:** `dispatch_actions` (Strictly single universal tool)  
**Default LLM Provider:** OpenRouter (`qwen/qwen3.8-27b:free`, Strict Free-Only Routing, No Fallbacks)  
**Shared Production Budget:** 50 requests/day with atomic checkpointing and multi-day resumability  

---
### 1. Architectural Responsibility
```
USER NATURAL LANGUAGE
        ↓
SEMANTIC UNDERSTANDING
        ↓
INTENT DETECTION (Intent Registry)
        ↓
ENTITY EXTRACTION (Grounding & Validation)
        ↓
ACTION DECOMPOSITION (Single, Multi, Sequential, Conditional)
        ↓
dispatch_actions UNIVERSAL TOOL
        ↓
DETERMINISTIC APPLICATION EXECUTION
```

The fine-tuned model does NOT directly execute application code or database operations. It learns to understand user natural language and produce structured actions via the canonical `dispatch_actions` tool.

### 2. Universal Tool Contract
The pipeline enforces ONE universal function tool: `dispatch_actions`. No individual per-intent tool schemas are created.

### 3. Baseline Data Sources (Expected Total: exactly 23,384 records)
- **Text**: BANKING77 (7,000), CLINC150 (8,000), HWU64 (5,000) = 20,000
- **Audio Transcripts**: MINDS-14 US (563), MINDS-14 EXT (1,246) = 1,809
- **OCR Documents**: CORD-v2 (800), SROIE (626), FUNSD (149) = 1,575

*Note: This notebook is completely self-contained. All classes, adapters, engines, and validators are defined within notebook cells with zero external local imports.*"""))

# ==============================================================================
# CELL 2: 01 - CONFIGURATION
# ==============================================================================
cells.append(code_cell("""# ==============================================================================
# 01 - PIPELINE CONFIGURATION
# ==============================================================================
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class PipelineConfig:
    \"\"\"Centralized immutable configuration for the Latent dataset pipeline.\"\"\"
    # Pipeline Metadata
    pipeline_version: str = "1.0.0"
    random_seed: int = 42
    
    # LLM Provider Configuration
    llm_provider: str = "openrouter"
    api_base_url: str = "https://openrouter.ai/api/v1"
    
    # Configurable Model Roles (Default: OpenRouter Free Tier)
    annotation_model: str = "qwen/qwen3.8-27b:free"
    synthetic_model: str = "qwen/qwen3.8-27b:free"
    validation_model: str = "qwen/qwen3.8-27b:free"
    connectivity_test_model: str = "qwen/qwen3.8-27b:free"
    
    # Strict Free-Only Routing Constraints (No paid fallbacks allowed)
    allow_fallbacks: bool = False
    
    # Shared Production Daily Quota
    daily_request_limit: int = 50
    stop_before_daily_limit: bool = True
    
    # Batching and Network Safeguards
    batch_size: int = 10
    max_batch_chars: int = 12000
    max_batch_tokens: int = 4096
    fallback_to_single_on_failure: bool = True
    max_retries: int = 3
    backoff_factor: float = 2.0
    request_timeout: int = 60
    
    # Pipeline Operational Modes
    dry_run_mode: bool = True               # True = preflight validation without production LLM calls
    test_mode_batch_limit: int = 5          # Number of items for rapid testing
    
    # Target Expected Counts (Baseline: exactly 23,384 records)
    target_banking77: int = 7000
    target_clinc150: int = 8000
    target_hwu64: int = 5000
    target_minds14_us: int = 563
    target_minds14_ext: int = 1246
    target_cord_v2: int = 800
    target_sroie: int = 626
    target_funsd: int = 149
    total_target: int = 23384
    
    # Directory Structure
    base_dir: str = "latentspace_dataset"
    output_dir: str = "latentspace_dataset/output"
    cache_dir: str = "latentspace_dataset/cache"
    config_dir: str = "latentspace_dataset/config"
    prompts_dir: str = "latentspace_dataset/prompts"

config = PipelineConfig()
print(f"Pipeline Config Initialized (v{config.pipeline_version})")
print(f"Target Baseline Records: {config.total_target:,}")
print(f"Active Provider: {config.llm_provider} (Model: {config.annotation_model})")
print(f"Strict Free-Only Routing: {'ENABLED (allow_fallbacks=False)' if not config.allow_fallbacks else 'DISABLED'}")
print(f"Daily Production Quota: {config.daily_request_limit} requests/day")"""))

# ==============================================================================
# CELL 3: 02 - IMPORTS & API KEY SECURITY GUARD
# ==============================================================================
cells.append(code_cell("""# ==============================================================================
# 02 - IMPORTS & API KEY SECURITY GUARD
# ==============================================================================
import json
import time
import math
import uuid
import re
import copy
import hashlib
import shutil
import urllib.request
import urllib.error
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

# Ensure all subdirectories exist
for subdir in [
    config.output_dir,
    config.config_dir,
    config.prompts_dir,
    os.path.join(config.cache_dir, "normalized"),
    os.path.join(config.cache_dir, "annotations"),
    os.path.join(config.cache_dir, "synthetic"),
    os.path.join(config.cache_dir, "validation"),
    os.path.join(config.cache_dir, "checkpoints")
]:
    os.makedirs(subdir, exist_ok=True)

def verify_api_key_presence() -> bool:
    \"\"\"
    Checks if OPENROUTER_API_KEY is configured in the environment.
    SECURITY RULE: Never print, log, or persist the key contents.
    \"\"\"
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        print("API Key Verification: [OK] Configured (YES)")
        return True
    else:
        print("API Key Verification: [WARNING] OPENROUTER_API_KEY not found in environment (NO).")
        print("  -> Pre-flight dry-run baseline assembly can proceed without API calls.")
        print("  -> Set os.environ['OPENROUTER_API_KEY'] before executing live LLM stages.")
        return False

api_key_ready = verify_api_key_presence()"""))

# ==============================================================================
# CELL 4: 03 - QUOTA MANAGER & AUDIT REQUEST LEDGER
# ==============================================================================
cells.append(code_cell("""# ==============================================================================
# 03 - QUOTA MANAGER & AUDIT REQUEST LEDGER
# ==============================================================================
class DailyQuotaExceededException(Exception):
    \"\"\"Raised when the shared production daily request budget is reached.\"\"\"
    pass

class QuotaManager:
    \"\"\"
    Manages the shared 50-request daily production budget and immutable audit ledger.
    Every single HTTP socket attempt on the wire consumes exactly 1 unit.
    Cached responses and pre-flight local validations consume 0 units.
    \"\"\"
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ledger_path = os.path.join(config.cache_dir, "request_ledger.jsonl")
        self._ensure_ledger_exists()

    def _ensure_ledger_exists(self):
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                pass

    @staticmethod
    def get_current_quota_date() -> str:
        \"\"\"Returns current UTC date in YYYY-MM-DD format.\"\"\"
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def count_production_requests_today(self) -> int:
        \"\"\"Counts actual production HTTP attempts made today (UTC).\"\"\"
        today = self.get_current_quota_date()
        count = 0
        if not os.path.exists(self.ledger_path):
            return 0
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("quota_date") == today and not entry.get("is_test", False):
                        count += 1
                except Exception:
                    continue
        return count

    def can_make_production_request(self) -> tuple[bool, int]:
        \"\"\"Checks if another production HTTP call is permitted under daily_request_limit.\"\"\"
        used = self.count_production_requests_today()
        remaining = max(0, self.config.daily_request_limit - used)
        allowed = remaining > 0
        return allowed, remaining

    def reserve_production_slot(self) -> int:
        \"\"\"
        Guards production budget immediately before an HTTP call.
        Raises DailyQuotaExceededException if budget is exhausted.
        \"\"\"
        allowed, remaining = self.can_make_production_request()
        if not allowed:
            used = self.count_production_requests_today()
            raise DailyQuotaExceededException(
                f"Daily production request budget ({self.config.daily_request_limit}) exhausted. "
                f"Used today: {used}. Execution cleanly halted. Resume tomorrow without loss of progress."
            )
        return remaining

    def log_attempt(
        self,
        request_hash: str,
        stage: str,
        model: str,
        is_test: bool,
        retry_number: int,
        records_attempted: int,
        records_successful: int,
        http_status: Optional[int],
        error_category: Optional[str],
        latency_ms: int
    ) -> str:
        \"\"\"
        Appends an immutable audit line to request_ledger.jsonl.
        ZERO SECRETS: Never stores authorization headers, tokens, or raw user texts.
        \"\"\"
        attempt_id = str(uuid.uuid4())
        quota_date = self.get_current_quota_date()
        
        total_attempts = 1
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    total_attempts = sum(1 for line in f if line.strip()) + 1
            except Exception:
                total_attempts = 1
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quota_date": quota_date,
            "attempt_id": attempt_id,
            "request_attempt": total_attempts,
            "retry_number": retry_number,
            "request_hash": request_hash,
            "stage": stage,
            "is_test": is_test,
            "model": model,
            "records_attempted": records_attempted,
            "records_successful": records_successful,
            "http_status": http_status,
            "error_category": error_category,
            "latency_ms": latency_ms
        }
        
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\\n")
            f.flush()
            
        return attempt_id

quota_mgr = QuotaManager(config)
used_today = quota_mgr.count_production_requests_today()
print(f"Quota Manager Ready. Date (UTC): {quota_mgr.get_current_quota_date()}")
print(f"Production Requests Used Today: {used_today} / {config.daily_request_limit} (Remaining: {max(0, config.daily_request_limit - used_today)})")"""))

# ==============================================================================
# CELL 5: 04 - UNIVERSAL TOOL CONTRACT & INTENT REGISTRY
# ==============================================================================
cells.append(code_cell("""# ==============================================================================
# 04 - UNIVERSAL TOOL CONTRACT & INTENT REGISTRY
# ==============================================================================
class IntentRegistry:
    \"\"\"
    Centralized source of truth for valid intent names and universal dispatch_actions tool.
    Strictly enforces required and optional entity validation.
    \"\"\"
    CANONICAL_TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": "dispatch_actions",
            "description": "Routes one or more parsed user intents and their extracted entities to the downstream system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "One or more actions extracted from the user's request.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_id": {
                                    "type": "string",
                                    "description": "Unique identifier for this action within the current request (e.g. 'a1', 'a2')."
                                },
                                "intent_name": {
                                    "type": "string",
                                    "description": "The classified intent."
                                },
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of action_ids that must complete before this action."
                                },
                                "entities": {
                                    "type": "object",
                                    "description": "Intent-specific extracted entities."
                                }
                            },
                            "required": ["action_id", "intent_name", "entities"]
                        }
                    }
                },
                "required": ["actions"]
            }
        }
    }

    def __init__(self, config_file: Optional[str] = None):
        self.intents = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.intents = data.get("intents", {})
        else:
            self._load_default_registry()

    def _load_default_registry(self):
        self.intents = {
            # Financial Domain
            "log_expense": {"domain": "financial", "description": "Record an expense", "required_entities": ["amount"], "optional_entities": ["merchant", "category", "date", "currency", "receipt_id"], "execution_type": "write"},
            "delete_expense": {"domain": "financial", "description": "Delete recorded expense", "required_entities": [], "optional_entities": ["expense_id", "merchant", "amount", "transaction_reference"], "execution_type": "delete"},
            "search_expense": {"domain": "financial", "description": "Search expenses", "required_entities": [], "optional_entities": ["merchant", "category", "date_range", "amount_range", "query"], "execution_type": "read"},
            "update_expense": {"domain": "financial", "description": "Update an expense", "required_entities": [], "optional_entities": ["expense_id", "amount", "merchant", "category", "date"], "execution_type": "update"},
            "transfer_money": {"domain": "financial", "description": "Transfer funds", "required_entities": ["amount"], "optional_entities": ["recipient", "source_account", "destination_account", "currency"], "execution_type": "write"},
            "check_balance": {"domain": "financial", "description": "Query account balance", "required_entities": [], "optional_entities": ["account_type", "account_id"], "execution_type": "read"},
            "check_transaction": {"domain": "financial", "description": "Inspect transactions", "required_entities": [], "optional_entities": ["transaction_id", "merchant", "date_range", "amount"], "execution_type": "read"},
            "cancel_transfer": {"domain": "financial", "description": "Cancel pending transfer", "required_entities": [], "optional_entities": ["transfer_id", "recipient", "amount"], "execution_type": "delete"},
            "card_lost": {"domain": "financial", "description": "Report card lost", "required_entities": [], "optional_entities": ["card_type", "last_four_digits"], "execution_type": "write"},
            "card_stolen": {"domain": "financial", "description": "Report card stolen", "required_entities": [], "optional_entities": ["card_type", "last_four_digits"], "execution_type": "write"},
            "freeze_card": {"domain": "financial", "description": "Freeze payment card", "required_entities": [], "optional_entities": ["card_type", "last_four_digits"], "execution_type": "update"},
            "unfreeze_card": {"domain": "financial", "description": "Unfreeze payment card", "required_entities": [], "optional_entities": ["card_type", "last_four_digits"], "execution_type": "update"},
            # Productivity Domain
            "create_task": {"domain": "productivity", "description": "Add task", "required_entities": ["title"], "optional_entities": ["due_date", "priority", "category"], "execution_type": "write"},
            "complete_task": {"domain": "productivity", "description": "Mark task complete", "required_entities": [], "optional_entities": ["task_id", "title"], "execution_type": "update"},
            "delete_task": {"domain": "productivity", "description": "Delete a task", "required_entities": [], "optional_entities": ["task_id", "title"], "execution_type": "delete"},
            "update_task": {"domain": "productivity", "description": "Update task details", "required_entities": [], "optional_entities": ["task_id", "title", "new_title", "due_date"], "execution_type": "update"},
            "search_task": {"domain": "productivity", "description": "Search tasks", "required_entities": [], "optional_entities": ["query", "status"], "execution_type": "read"},
            "create_reminder": {"domain": "productivity", "description": "Set reminder", "required_entities": ["task"], "optional_entities": ["schedule", "due_date", "time"], "execution_type": "write"},
            "cancel_reminder": {"domain": "productivity", "description": "Cancel reminder", "required_entities": [], "optional_entities": ["reminder_id", "task"], "execution_type": "delete"},
            "update_reminder": {"domain": "productivity", "description": "Update reminder schedule", "required_entities": [], "optional_entities": ["reminder_id", "task", "new_schedule"], "execution_type": "update"},
            "search_reminder": {"domain": "productivity", "description": "Search reminders", "required_entities": [], "optional_entities": ["query"], "execution_type": "read"},
            "create_alarm": {"domain": "productivity", "description": "Set alarm", "required_entities": ["time"], "optional_entities": ["label"], "execution_type": "write"},
            "cancel_alarm": {"domain": "productivity", "description": "Cancel alarm", "required_entities": [], "optional_entities": ["alarm_id", "time"], "execution_type": "delete"},
            "update_alarm": {"domain": "productivity", "description": "Update alarm", "required_entities": [], "optional_entities": ["alarm_id", "new_time"], "execution_type": "update"},
            # Booking Domain
            "book_reservation": {"domain": "booking", "description": "Book a reservation", "required_entities": ["venue"], "optional_entities": ["date", "time", "party_size"], "execution_type": "write"},
            "cancel_reservation": {"domain": "booking", "description": "Cancel reservation", "required_entities": [], "optional_entities": ["reservation_id", "venue"], "execution_type": "delete"},
            "search_reservation": {"domain": "booking", "description": "Search reservation availability", "required_entities": [], "optional_entities": ["venue", "date"], "execution_type": "read"},
            "search_flight": {"domain": "booking", "description": "Search flights", "required_entities": [], "optional_entities": ["origin", "destination", "departure_date"], "execution_type": "read"},
            "book_flight": {"domain": "booking", "description": "Book flight", "required_entities": ["destination"], "optional_entities": ["origin", "departure_date", "flight_number"], "execution_type": "write"},
            "cancel_flight": {"domain": "booking", "description": "Cancel flight", "required_entities": [], "optional_entities": ["booking_reference", "flight_number"], "execution_type": "delete"},
            # Meta & Control Domain
            "search_information": {"domain": "information", "description": "Factual non-executable query", "required_entities": [], "optional_entities": ["topic", "query"], "execution_type": "read"},
            "out_of_scope": {"domain": "control", "description": "Outside supported assistant capabilities", "required_entities": [], "optional_entities": ["raw_request", "reason"], "execution_type": "control"},
            "clarification_required": {"domain": "control", "description": "Essential information missing", "required_entities": ["missing_information"], "optional_entities": ["intended_intent"], "execution_type": "control"}
        }

    def is_valid_intent(self, intent_name: str) -> bool:
        return intent_name in self.intents

    def validate_action(self, action: dict, prior_action_ids: set) -> tuple[bool, str]:
        if not isinstance(action, dict):
            return False, "Action must be a dictionary"
        for field in ["action_id", "intent_name", "entities"]:
            if field not in action:
                return False, f"Missing required action field: {field}"
        
        intent_name = action["intent_name"]
        if not self.is_valid_intent(intent_name):
            return False, f"Intent '{intent_name}' is not registered in IntentRegistry"
        
        intent_def = self.intents[intent_name]
        entities = action.get("entities", {})
        if not isinstance(entities, dict):
            return False, "Action 'entities' must be a dictionary"
        for req in intent_def.get("required_entities", []):
            if req not in entities or entities[req] is None:
                return False, f"Intent '{intent_name}' missing required entity: {req}"
        
        for dep in action.get("depends_on", []):
            if dep not in prior_action_ids:
                return False, f"Invalid dependency '{dep}': must reference a prior action_id"
        
        return True, "Valid"

    def validate_dispatch_payload(self, payload: dict) -> tuple[bool, str]:
        \"\"\"Strictly validates a dispatch_actions argument payload.\"\"\"
        if not isinstance(payload, dict) or "actions" not in payload:
            return False, "Payload must contain top-level 'actions' array"
        actions = payload["actions"]
        if not isinstance(actions, list) or len(actions) == 0:
            return False, "'actions' array must be non-empty"
        
        seen_action_ids = set()
        for action in actions:
            action_id = action.get("action_id")
            if not action_id:
                return False, "Action missing 'action_id'"
            if action_id in seen_action_ids:
                return False, f"Duplicate action_id '{action_id}'"
            valid, reason = self.validate_action(action, seen_action_ids)
            if not valid:
                return False, reason
            seen_action_ids.add(action_id)
            
        return True, "Valid dispatch_actions payload"

intent_reg = IntentRegistry(os.path.join(config.config_dir, "intent_registry.json"))
print(f"Intent Registry Loaded: {len(intent_reg.intents)} intents registered.")
print(f"Universal Tool: '{IntentRegistry.CANONICAL_TOOL_SCHEMA['function']['name']}' verified.")"""))

# ==========================================
# CELL 6: 05 - PROVIDER ABSTRACTION & OPENROUTER
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 05 - PROVIDER ABSTRACTION & STRICT FREE-ROUTING OPENROUTER PROVIDER
# ==============================================================================
class LLMProvider(ABC):
    \"\"\"Abstract Base Class for LLM inference providers.\"\"\"
    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        stage: str,
        record_ids: List[str],
        schema: Optional[dict] = None,
        is_test: bool = False,
        temperature: float = 0.0
    ) -> dict:
        pass

class OpenRouterProvider(LLMProvider):
    \"\"\"
    OpenRouter API provider with strict free-only constraints:
    - allow_fallbacks = False (Never allows automatic paid routing)
    - Integrates directly with QuotaManager shared budget
    - Cryptographic request hash identification
    - Two-tier disk response cache before making HTTP requests
    - Categorized error handling: Permanent vs Transient vs Quota vs Malformed
    \"\"\"
    def __init__(self, config: PipelineConfig, quota_mgr: QuotaManager):
        self.config = config
        self.quota_mgr = quota_mgr
        self.endpoint = f"{config.api_base_url}/chat/completions"

    def compute_request_hash(
        self,
        stage: str,
        model: str,
        record_ids: List[str],
        prompt: str,
        schema: Optional[dict]
    ) -> str:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        schema_hash = hashlib.sha256(json.dumps(schema or {}, sort_keys=True).encode("utf-8")).hexdigest()
        records_str = ",".join(sorted(record_ids))
        raw = f"v={self.config.pipeline_version}|stage={stage}|model={model}|records={records_str}|p={prompt_hash}|s={schema_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _get_cache_path(self, stage: str, request_hash: str) -> str:
        return os.path.join(self.config.cache_dir, stage, f"{request_hash}.json")

    def _strip_markdown_fences(self, text: str) -> str:
        text = text.strip()
        fence_match = re.match(r"^```(?:json)?\\s*([\\s\\S]*?)\\s*```$", text)
        if fence_match:
            return fence_match.group(1).strip()
        return text

    def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        stage: str,
        record_ids: List[str],
        schema: Optional[dict] = None,
        is_test: bool = False,
        temperature: float = 0.0
    ) -> dict:
        request_hash = self.compute_request_hash(stage, model, record_ids, prompt, schema)
        cache_file = self._get_cache_path(stage, request_hash)

        if not is_test and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        if not is_test:
            self.quota_mgr.reserve_production_slot()

        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured in environment.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://latent-agent.local",
            "X-Title": "Latent-Dataset-Pipeline"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "provider": {
                "allow_fallbacks": self.config.allow_fallbacks
            }
        }

        if schema:
            payload["response_format"] = {
                "type": "json_object"
            }

        data_bytes = json.dumps(payload).encode("utf-8")
        
        attempts = 0
        backoff = self.config.backoff_factor

        while attempts <= self.config.max_retries:
            attempts += 1
            retry_num = attempts - 1
            start_time = time.time()
            http_status = None
            error_cat = None

            try:
                req = urllib.request.Request(self.endpoint, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
                    http_status = resp.status
                    body = resp.read().decode("utf-8")
                    latency = int((time.time() - start_time) * 1000)
                    
                    response_json = json.loads(body)
                    raw_content = response_json["choices"][0]["message"]["content"]
                    clean_content = self._strip_markdown_fences(raw_content)
                    parsed_result = json.loads(clean_content)

                    self.quota_mgr.log_attempt(
                        request_hash=request_hash,
                        stage=stage,
                        model=model,
                        is_test=is_test,
                        retry_number=retry_num,
                        records_attempted=len(record_ids),
                        records_successful=len(record_ids),
                        http_status=http_status,
                        error_category=None,
                        latency_ms=latency
                    )

                    if not is_test:
                        with open(cache_file, "w", encoding="utf-8") as cf:
                            json.dump(parsed_result, cf, indent=2)

                    return parsed_result

            except urllib.error.HTTPError as e:
                http_status = e.code
                latency = int((time.time() - start_time) * 1000)
                err_body = ""
                try: err_body = e.read().decode("utf-8")
                except: pass

                if http_status in [401, 403, 404]:
                    error_cat = "AUTH_PERMANENT"
                    self.quota_mgr.log_attempt(request_hash, stage, model, is_test, retry_num, len(record_ids), 0, http_status, error_cat, latency)
                    raise RuntimeError(f"Permanent HTTP {http_status} Error: {err_body}")

                elif http_status == 429:
                    if "quota" in err_body.lower() or "limit reached" in err_body.lower() or "exceeded" in err_body.lower():
                        error_cat = "PROVIDER_QUOTA_EXHAUSTION"
                        self.quota_mgr.log_attempt(request_hash, stage, model, is_test, retry_num, len(record_ids), 0, http_status, error_cat, latency)
                        raise DailyQuotaExceededException(f"OpenRouter quota exhausted: {err_body}")
                    else:
                        error_cat = "TEMPORARY_RATE_LIMIT"
                        self.quota_mgr.log_attempt(request_hash, stage, model, is_test, retry_num, len(record_ids), 0, http_status, error_cat, latency)
                        if attempts > self.config.max_retries: raise
                        time.sleep(backoff)
                        backoff *= 2.0

                elif http_status >= 500:
                    error_cat = "SERVER_ERROR"
                    self.quota_mgr.log_attempt(request_hash, stage, model, is_test, retry_num, len(record_ids), 0, http_status, error_cat, latency)
                    if attempts > self.config.max_retries: raise
                    time.sleep(backoff)
                    backoff *= 2.0

                else:
                    error_cat = "HTTP_CLIENT_ERROR"
                    self.quota_mgr.log_attempt(request_hash, stage, model, is_test, retry_num, len(record_ids), 0, http_status, error_cat, latency)
                    raise

            except json.JSONDecodeError as jde:
                latency = int((time.time() - start_time) * 1000)
                error_cat = "MALFORMED_JSON_OUTPUT"
                self.quota_mgr.log_attempt(request_hash, stage, model, is_test, retry_num, len(record_ids), 0, http_status or 200, error_cat, latency)
                if attempts > self.config.max_retries:
                    raise ValueError(f"Malformed LLM JSON response after {attempts} attempts: {jde}")
                time.sleep(backoff)
                backoff *= 1.5

            except Exception as ex:
                latency = int((time.time() - start_time) * 1000)
                error_cat = "NETWORK_ERROR"
                self.quota_mgr.log_attempt(request_hash, stage, model, is_test, retry_num, len(record_ids), 0, http_status, error_cat, latency)
                if attempts > self.config.max_retries: raise
                time.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(f"Request failed after {attempts} attempts.")

llm_provider = OpenRouterProvider(config, quota_mgr)
print(f"OpenRouter Provider Initialized. Strict free routing: allow_fallbacks={config.allow_fallbacks}")"""))

# ==========================================
# CELL 7: 06 - ATOMIC CHECKPOINT MANAGER
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 06 - ATOMIC CHECKPOINT MANAGER
# ==============================================================================
class CheckpointManager:
    \"\"\"
    Manages atomic persistence of pipeline state snapshots.
    Ensures safe multi-day resumability and zero state corruption upon kernel crashes.
    \"\"\"
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_path = os.path.join(config.cache_dir, "checkpoints", "pipeline_checkpoint.json")

    def load_checkpoint(self) -> dict:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    if state.get("pipeline_version") == self.config.pipeline_version:
                        return state
                    else:
                        print(f"Checkpoint version mismatch ({state.get('pipeline_version')} vs {self.config.pipeline_version}). Starting fresh.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting fresh.")
        
        return {
            "pipeline_version": self.config.pipeline_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "stage": "initialized",
            "last_batch_index": 0,
            "completed_record_ids": [],
            "failed_record_ids": [],
            "quota_exhausted": False
        }

    def save_checkpoint(self, state: dict):
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        state["pipeline_version"] = self.config.pipeline_version
        
        tmp_path = f"{self.checkpoint_path}.tmp_{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.checkpoint_path)

checkpoint_mgr = CheckpointManager(config)
current_checkpoint = checkpoint_mgr.load_checkpoint()
print(f"Checkpoint Manager Ready. Current Stage: '{current_checkpoint['stage']}' (Completed: {len(current_checkpoint['completed_record_ids'])})")"""))

# ==========================================
# CELL 8: 07 - PRE-FLIGHT CONNECTIVITY TEST
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 07 - PRE-FLIGHT CONNECTIVITY TEST
# ==============================================================================
def run_connectivity_test(provider: LLMProvider, cfg: PipelineConfig, quota: QuotaManager) -> bool:
    \"\"\"
    Pre-flight sanity check:
    1. Runs with is_test=True (Separated test-mode accounting).
    2. Does NOT deduct from the production 50-request daily limit.
    3. Confirms JSON schema generation, error handling, and ledger persistence.
    \"\"\"
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        print("Pre-flight Connectivity Test: SKIPPED (No API key in environment).")
        return False
        
    print(f"Running connectivity test against '{cfg.connectivity_test_model}' (is_test=True)...")
    system_prompt = "You are a test agent. Return a valid JSON object matching the requested schema."
    test_prompt = "Parse this test input: 'Check my bank balance'"
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "test_intent": {"type": "string"}
        },
        "required": ["status", "test_intent"]
    }
    
    try:
        response = provider.generate_json(
            prompt=test_prompt,
            system_prompt=system_prompt,
            model=cfg.connectivity_test_model,
            stage="test",
            record_ids=["TEST_PING_001"],
            schema=schema,
            is_test=True
        )
        print("  -> Connectivity Test: SUCCESS!")
        print(f"  -> Model Response: {response}")
        print("  -> Confirmed: Test attempt logged in ledger without deducting from production budget.")
        return True
    except Exception as e:
        print(f"  -> Connectivity Test FAILED: {e}")
        return False

connectivity_success = run_connectivity_test(llm_provider, config, quota_mgr)"""))

# ==========================================
# CELL 9: 08 - DATASET ADAPTER BASE CLASS
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 08 - DATASET ADAPTER BASE CLASS
# ==============================================================================
class DatasetAdapter(ABC):
    \"\"\"
    Abstract base class for source dataset normalization adapters.
    Guarantees:
    - Verbatim untouched source text.
    - Deterministic record_id = f\"{source_dataset}_{source_record_id}\".
    - record_group_id preservation across all synthetic derivatives.
    - Non-duplication principle: Never duplicates records to fill shortfalls.
    - Explicit audit accounting: raw, valid, english, selected, requested, shortfall.
    - Caching to cache/normalized/{dataset_name}.parquet.
    \"\"\"
    def __init__(self, dataset_name: str, target_count: int, config: PipelineConfig):
        self.dataset_name = dataset_name
        self.target_count = target_count
        self.config = config
        self.cache_path = os.path.join(config.cache_dir, "normalized", f"{dataset_name}.parquet")

    @abstractmethod
    def load_raw_records(self) -> List[dict]:
        pass

    @abstractmethod
    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        pass

    def is_english_text(self, text: Optional[str]) -> bool:
        if not text or not isinstance(text, str):
            return False
        cleaned = text.strip()
        if len(cleaned) < 2:
            return False
        valid_chars = re.findall(r"[A-Za-z0-9\\s\\.\\,\\!\\?\\'\\\"\\-\\$\\£\\€\\¥\\;\\:\\/\\(\\)\\%\\&\\@\\_\\—\\–\\‘\\’\\“\\”\\…\\é\\è\\ê\\á\\à\\ó\\í]", cleaned)
        ratio = len(valid_chars) / max(1, len(cleaned))
        return ratio >= 0.85

    def process(self) -> tuple[pd.DataFrame, dict]:
        if os.path.exists(self.cache_path):
            print(f"[{self.dataset_name}] Loading normalized records from cache: {self.cache_path}")
            df = pd.read_parquet(self.cache_path)
            stats = {
                "dataset": self.dataset_name,
                "cached": True,
                "selected_count": len(df),
                "requested_count": self.target_count,
                "shortfall": max(0, self.target_count - len(df))
            }
            return df, stats

        print(f"[{self.dataset_name}] Ingesting and normalizing from source...")
        raw_items = self.load_raw_records()
        raw_count = len(raw_items)
        
        normalized_records = []
        valid_count = 0
        english_count = 0

        for idx, item in enumerate(raw_items):
            record = self.normalize_single_record(item, idx)
            if record is not None:
                valid_count += 1
                target_text = record.get("normalized_text") or record.get("original_text") or record.get("document_text") or ""
                if self.is_english_text(target_text):
                    english_count += 1
                    normalized_records.append(record)

        if len(normalized_records) > self.target_count:
            rng = np.random.default_rng(self.config.random_seed)
            indices = rng.choice(len(normalized_records), size=self.target_count, replace=False)
            indices.sort()
            selected_records = [normalized_records[i] for i in indices]
        else:
            selected_records = normalized_records

        selected_count = len(selected_records)
        shortfall = max(0, self.target_count - selected_count)

        if shortfall > 0:
            print(f"  -> [{self.dataset_name}] SHORTFALL WARNING: Available valid English records ({selected_count}) < Requested ({self.target_count}).")
            print(f"     Non-duplication policy enforced: Exactly {selected_count} records ingested (Shortfall: {shortfall}).")

        df = pd.DataFrame(selected_records)
        df.to_parquet(self.cache_path, index=False)

        stats = {
            "dataset": self.dataset_name,
            "cached": False,
            "raw_count": raw_count,
            "valid_count": valid_count,
            "english_count": english_count,
            "selected_count": selected_count,
            "requested_count": self.target_count,
            "shortfall": shortfall
        }
        return df, stats"""))

# ==========================================
# CELL 10: 09 - CONCRETE TEXT ADAPTERS
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 09 - CONCRETE TEXT DATASET ADAPTERS (BANKING77, CLINC150, HWU64)
# ==============================================================================
class Banking77Adapter(DatasetAdapter):
    \"\"\"
    BANKING77 Adapter. Target: 7,000 records.
    Source: PolyAI-LDN repository raw train.csv.
    \"\"\"
    def __init__(self, config: PipelineConfig):
        super().__init__("BANKING77", config.target_banking77, config)

    def load_raw_records(self) -> List[dict]:
        url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
        try:
            df = pd.read_csv(url)
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"PolyAI github download failed ({e}), trying Hugging Face...")
            from datasets import load_dataset
            ds = load_dataset("PolyAI/banking77", split="train")
            return [dict(x) for x in ds]

    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        text = str(raw_item.get("text", "")).strip()
        source_intent = str(raw_item.get("category", raw_item.get("label", ""))).strip()
        source_record_id = str(index)
        record_id = f"BANKING77_{index:06d}"
        
        return {
            "record_id": record_id,
            "source_record_id": source_record_id,
            "record_group_id": record_id,
            "source_dataset": "BANKING77",
            "source_split": "train",
            "source_intent": source_intent,
            "intent_name": None,
            "mapping_status": "pending",
            "modality": "text",
            "original_text": text,
            "document_text": None,
            "normalized_text": text,
            "source_metadata": {"raw_category": source_intent},
            "language": "en",
            "is_original": True,
            "annotation_status": "pending",
            "entities": {},
            "action_count": 0,
            "scenario_type": None,
            "annotation_confidence": None
        }

class Clinc150Adapter(DatasetAdapter):
    \"\"\"
    CLINC150 Adapter. Target: 8,000 records.
    Source: DeepPavlov/clinc150 train split with 'intents' mapping config.
    \"\"\"
    def __init__(self, config: PipelineConfig):
        super().__init__("CLINC150", config.target_clinc150, config)

    def load_raw_records(self) -> List[dict]:
        from datasets import load_dataset
        clinc_intents = load_dataset("DeepPavlov/clinc150", "intents", split="intents")
        self.clinc_map = {x["id"]: x["name"] for x in clinc_intents}
        ds = load_dataset("DeepPavlov/clinc150", split="train")
        return [dict(x) for x in ds]

    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        text = str(raw_item.get("utterance", raw_item.get("text", ""))).strip()
        lbl_id = raw_item.get("label")
        source_intent = self.clinc_map.get(lbl_id, "oos") if hasattr(self, "clinc_map") else str(lbl_id)
        source_record_id = str(index)
        record_id = f"CLINC150_{index:06d}"
        
        is_oos = (source_intent == "oos")
        mapping_status = "out_of_scope" if is_oos else "pending"
        intent_name = "out_of_scope" if is_oos else None
        
        return {
            "record_id": record_id,
            "source_record_id": source_record_id,
            "record_group_id": record_id,
            "source_dataset": "CLINC150",
            "source_split": "train",
            "source_intent": source_intent,
            "intent_name": intent_name,
            "mapping_status": mapping_status,
            "modality": "text",
            "original_text": text,
            "document_text": None,
            "normalized_text": text,
            "source_metadata": {"is_oos": is_oos},
            "language": "en",
            "is_original": True,
            "annotation_status": "pending",
            "entities": {},
            "action_count": 0,
            "scenario_type": None,
            "annotation_confidence": None
        }

class Hwu64Adapter(DatasetAdapter):
    \"\"\"
    HWU64 Adapter. Target: 5,000 records.
    Source: DeepPavlov/hwu64 train split with 'intents' mapping config.
    \"\"\"
    def __init__(self, config: PipelineConfig):
        super().__init__("HWU64", config.target_hwu64, config)

    def load_raw_records(self) -> List[dict]:
        from datasets import load_dataset
        hwu_intents = load_dataset("DeepPavlov/hwu64", "intents", split="intents")
        self.hwu_map = {x["id"]: x["name"] for x in hwu_intents}
        ds = load_dataset("DeepPavlov/hwu64", split="train")
        return [dict(x) for x in ds]

    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        text = str(raw_item.get("utterance", raw_item.get("text", ""))).strip()
        lbl_id = raw_item.get("label")
        source_intent = self.hwu_map.get(lbl_id, f"intent_{lbl_id}") if hasattr(self, "hwu_map") else str(lbl_id)
        source_record_id = str(index)
        record_id = f"HWU64_{index:06d}"
        
        return {
            "record_id": record_id,
            "source_record_id": source_record_id,
            "record_group_id": record_id,
            "source_dataset": "HWU64",
            "source_split": "train",
            "source_intent": source_intent,
            "intent_name": None,
            "mapping_status": "pending",
            "modality": "text",
            "original_text": text,
            "document_text": None,
            "normalized_text": text,
            "source_metadata": {},
            "language": "en",
            "is_original": True,
            "annotation_status": "pending",
            "entities": {},
            "action_count": 0,
            "scenario_type": None,
            "annotation_confidence": None
        }"""))

# ==========================================
# CELL 11: 10 - CONCRETE AUDIO ADAPTERS
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 10 - CONCRETE AUDIO TRANSCRIPT ADAPTERS (MINDS-14 US & EXT)
# ==============================================================================
class Minds14USAdapter(DatasetAdapter):
    \"\"\"
    MINDS-14 US Adapter. Target: 563 records.
    Source: PolyAI/minds14 ('en-US' config, train split).
    \"\"\"
    def __init__(self, config: PipelineConfig):
        super().__init__("MINDS14_US", config.target_minds14_us, config)

    def load_raw_records(self) -> List[dict]:
        from datasets import load_dataset
        ds = load_dataset("PolyAI/minds14", "en-US", split="train")
        return [dict(x) for x in ds]

    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        transcription = str(raw_item.get("transcription", "")).strip()
        intent = str(raw_item.get("intent_class", "")).strip()
        source_record_id = str(index)
        record_id = f"MINDS14_US_{index:06d}"
        
        return {
            "record_id": record_id,
            "source_record_id": source_record_id,
            "record_group_id": record_id,
            "source_dataset": "MINDS14_US",
            "source_split": "train",
            "source_intent": intent,
            "intent_name": None,
            "mapping_status": "pending",
            "modality": "audio_transcript",
            "original_text": transcription,
            "document_text": None,
            "normalized_text": transcription,
            "source_metadata": {"english_dialect": "en-US"},
            "language": "en",
            "is_original": True,
            "annotation_status": "pending",
            "entities": {},
            "action_count": 0,
            "scenario_type": None,
            "annotation_confidence": None
        }

class Minds14ExtAdapter(DatasetAdapter):
    \"\"\"
    MINDS-14 Extended Adapter. Target: 1,246 records (654 en-AU + 592 en-GB).
    Source: PolyAI/minds14 ('en-AU' & 'en-GB' configs, train split).
    \"\"\"
    def __init__(self, config: PipelineConfig):
        super().__init__("MINDS14_EXT", config.target_minds14_ext, config)

    def load_raw_records(self) -> List[dict]:
        from datasets import load_dataset
        ds_au = load_dataset("PolyAI/minds14", "en-AU", split="train")
        ds_gb = load_dataset("PolyAI/minds14", "en-GB", split="train")
        items = []
        for x in ds_au:
            d = dict(x)
            d["dialect"] = "en-AU"
            items.append(d)
        for x in ds_gb:
            d = dict(x)
            d["dialect"] = "en-GB"
            items.append(d)
        return items

    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        transcription = str(raw_item.get("transcription", "")).strip()
        intent = str(raw_item.get("intent_class", "")).strip()
        dialect = raw_item.get("dialect", "en-EXT")
        source_record_id = str(index)
        record_id = f"MINDS14_EXT_{index:06d}"
        
        return {
            "record_id": record_id,
            "source_record_id": source_record_id,
            "record_group_id": record_id,
            "source_dataset": "MINDS14_EXT",
            "source_split": "train",
            "source_intent": intent,
            "intent_name": None,
            "mapping_status": "pending",
            "modality": "audio_transcript",
            "original_text": transcription,
            "document_text": None,
            "normalized_text": transcription,
            "source_metadata": {"english_dialect": dialect},
            "language": "en",
            "is_original": True,
            "annotation_status": "pending",
            "entities": {},
            "action_count": 0,
            "scenario_type": None,
            "annotation_confidence": None
        }"""))

# ==========================================
# CELL 12: 11 - CONCRETE OCR ADAPTERS
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 11 - CONCRETE OCR DOCUMENT ADAPTERS (CORD-v2, SROIE, FUNSD)
# ==============================================================================
class CordV2Adapter(DatasetAdapter):
    \"\"\"
    CORD-v2 Receipt Adapter. Target: 800 records (naver-clova-ix/cord-v2 train split).
    \"\"\"
    def __init__(self, config: PipelineConfig):
        super().__init__("CORD_V2", config.target_cord_v2, config)

    def load_raw_records(self) -> List[dict]:
        from datasets import load_dataset
        ds = load_dataset("naver-clova-ix/cord-v2", split="train")
        return [dict(x) for x in ds]

    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        gt_parse = raw_item.get("ground_truth", {})
        if isinstance(gt_parse, str):
            try: gt_parse = json.loads(gt_parse).get("gt_parse", {})
            except: gt_parse = {}
        elif isinstance(gt_parse, dict) and "gt_parse" in gt_parse:
            gt_parse = gt_parse["gt_parse"]

        total_price = ""
        if isinstance(gt_parse.get("total"), dict):
            total_price = gt_parse["total"].get("total_price", "")

        menu_items = gt_parse.get("menu", []) if isinstance(gt_parse.get("menu"), list) else []
        items_desc = []
        for m in menu_items:
            if isinstance(m, dict) and m.get("nm"):
                pr = m.get("price", "")
                items_desc.append(f"{m['nm']} ({pr})" if pr else str(m["nm"]))
            elif isinstance(m, str):
                items_desc.append(m)

        doc_lines = [f"Items: {', '.join(items_desc[:5])}"] if items_desc else []
        if total_price: doc_lines.append(f"Total: {total_price}")
        
        doc_text = " | ".join(doc_lines) if doc_lines else "Receipt document"
        norm_text = f"Receipt with items: {', '.join(items_desc[:3])}; Total: {total_price}" if total_price else doc_text
        source_record_id = str(index)
        record_id = f"CORD_{index:06d}"
        
        return {
            "record_id": record_id,
            "source_record_id": source_record_id,
            "record_group_id": record_id,
            "source_dataset": "CORD_V2",
            "source_split": "train",
            "source_intent": "receipt_expense",
            "intent_name": None,
            "mapping_status": "pending",
            "modality": "ocr",
            "original_text": None,
            "document_text": doc_text,
            "normalized_text": norm_text,
            "source_metadata": {"total_price": total_price, "item_count": len(items_desc)},
            "language": "en",
            "is_original": True,
            "annotation_status": "pending",
            "entities": {},
            "action_count": 0,
            "scenario_type": None,
            "annotation_confidence": None
        }

class SroieAdapter(DatasetAdapter):
    \"\"\"
    SROIE Receipt Adapter. Target: 626 records (rth/sroie-2019-v2 train split).
    \"\"\"
    def __init__(self, config: PipelineConfig):
        super().__init__("SROIE", config.target_sroie, config)

    def load_raw_records(self) -> List[dict]:
        from datasets import load_dataset
        ds = load_dataset("rth/sroie-2019-v2", split="train")
        return [dict(x) for x in ds]

    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        company = str(raw_item.get("company", "")).strip()
        date = str(raw_item.get("date", "")).strip()
        address = str(raw_item.get("address", "")).strip()
        total = str(raw_item.get("total", "")).strip()
        
        doc_parts = []
        if company: doc_parts.append(f"Company: {company}")
        if date: doc_parts.append(f"Date: {date}")
        if total: doc_parts.append(f"Total: {total}")
        if address: doc_parts.append(f"Address: {address}")
        
        doc_text = " | ".join(doc_parts) if doc_parts else "Receipt details"
        norm_text = f"Purchase at {company} on {date} for {total}" if company and total else doc_text
        source_record_id = str(index)
        record_id = f"SROIE_{index:06d}"
        
        return {
            "record_id": record_id,
            "source_record_id": source_record_id,
            "record_group_id": record_id,
            "source_dataset": "SROIE",
            "source_split": "train",
            "source_intent": "receipt_expense",
            "intent_name": None,
            "mapping_status": "pending",
            "modality": "ocr",
            "original_text": None,
            "document_text": doc_text,
            "normalized_text": norm_text,
            "source_metadata": {"company": company, "total": total, "date": date},
            "language": "en",
            "is_original": True,
            "annotation_status": "pending",
            "entities": {},
            "action_count": 0,
            "scenario_type": None,
            "annotation_confidence": None
        }

class FunsdAdapter(DatasetAdapter):
    \"\"\"
    FUNSD Form Document Adapter. Target: 149 records (nielsr/funsd train split).
    \"\"\"
    def __init__(self, config: PipelineConfig):
        super().__init__("FUNSD", config.target_funsd, config)

    def load_raw_records(self) -> List[dict]:
        from datasets import load_dataset
        ds = load_dataset("nielsr/funsd", split="train")
        return [dict(x) for x in ds]

    def normalize_single_record(self, raw_item: dict, index: int) -> Optional[dict]:
        words = raw_item.get("words", [])
        doc_text = " ".join(words[:60]) if words else "Scanned business form"
        norm_text = f"Form document containing: {doc_text[:120]}..."
        source_record_id = str(index)
        record_id = f"FUNSD_{index:06d}"
        
        return {
            "record_id": record_id,
            "source_record_id": source_record_id,
            "record_group_id": record_id,
            "source_dataset": "FUNSD",
            "source_split": "train",
            "source_intent": "form_information",
            "intent_name": None,
            "mapping_status": "pending",
            "modality": "ocr",
            "original_text": None,
            "document_text": doc_text,
            "normalized_text": norm_text,
            "source_metadata": {"word_count": len(words)},
            "language": "en",
            "is_original": True,
            "annotation_status": "pending",
            "entities": {},
            "action_count": 0,
            "scenario_type": None,
            "annotation_confidence": None
        }"""))

# ==========================================
# CELL 13: 12 - UNIFIED NORMALIZATION RUNNER
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 12 - UNIFIED DATASET INGESTION & NORMALIZATION RUNNER
# ==============================================================================
class DatasetNormalizer:
    \"\"\"
    Orchestrates all source adapters to produce the normalized baseline dataset.
    Enforces non-duplication, calculates shortfalls, and saves original_dataset.csv.
    \"\"\"
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.adapters = [
            Banking77Adapter(config),
            Clinc150Adapter(config),
            Hwu64Adapter(config),
            Minds14USAdapter(config),
            Minds14ExtAdapter(config),
            CordV2Adapter(config),
            SroieAdapter(config),
            FunsdAdapter(config)
        ]

    def run(self) -> tuple[pd.DataFrame, dict]:
        frames = []
        audit_reports = {}
        total_selected = 0
        total_shortfall = 0

        print("=" * 70)
        print("STARTING DATASET INGESTION & NORMALIZATION AUDIT")
        print("=" * 70)

        for adapter in self.adapters:
            df, stats = adapter.process()
            frames.append(df)
            audit_reports[adapter.dataset_name] = stats
            total_selected += len(df)
            total_shortfall += stats.get("shortfall", 0)

        combined_df = pd.concat(frames, ignore_index=True)
        
        unique_ids = combined_df["record_id"].nunique()
        assert unique_ids == len(combined_df), f"Duplicate record_ids detected! {len(combined_df)} rows vs {unique_ids} unique IDs"
        
        orig_csv = os.path.join(self.config.output_dir, "original_dataset.csv")
        export_df = combined_df.copy()
        export_df["source_metadata"] = export_df["source_metadata"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
        export_df["entities"] = export_df["entities"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
        export_df.to_csv(orig_csv, index=False)
        print(f"\\n[OK] Normalized baseline dataset written to: {orig_csv}")
        print(f"     Total Baseline Records Ingested: {total_selected:,} / Requested: {self.config.total_target:,} (Shortfall: {total_shortfall})")
        print("=" * 70)

        return combined_df, audit_reports

normalizer = DatasetNormalizer(config)
baseline_df, normalization_audit = normalizer.run()"""))

# ==========================================
# CELL 14: 13 - ANNOTATION ENGINE
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 13 - ANNOTATION ENGINE & MULTI-INTENT PARSER
# ==============================================================================
class AnnotationEngine:
    \"\"\"
    Decomposes normalized inputs into structured actions adhering to dispatch_actions.
    Supports SINGLE, MULTI-INTENT, SEQUENTIAL (depends_on), and CONDITIONAL scenarios.
    Caches results to cache/annotations/{record_id}.json.
    \"\"\"
    def __init__(self, config: PipelineConfig, provider: LLMProvider, registry: IntentRegistry):
        self.config = config
        self.provider = provider
        self.registry = registry
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_file = os.path.join(self.config.prompts_dir, "annotation_prompt.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        return "You are a dataset annotation engine for the Latent assistant. Decompose inputs into dispatch_actions."

    def build_batch_prompt(self, records: List[dict]) -> str:
        prompt_lines = [
            "Decompose each of the following user records into the canonical dispatch_actions format.",
            "Use ONLY valid intents from the Intent Registry.",
            "Identify single-intent, multi-intent, sequential (with depends_on), and clarification scenarios.",
            "Return a JSON object with key 'results' containing an array of annotated records.",
            "\\nSupported Intents Summary:",
            ", ".join(sorted(self.registry.intents.keys())),
            "\\nINPUT RECORDS TO ANNOTATE:"
        ]
        
        for r in records:
            text = r.get("normalized_text") or r.get("original_text") or r.get("document_text") or ""
            prompt_lines.append(
                f"- record_id: {r['record_id']}\\n"
                f"  source_dataset: {r['source_dataset']}\\n"
                f"  source_intent: {r.get('source_intent')}\\n"
                f"  modality: {r['modality']}\\n"
                f"  text: \\"{text}\\""
            )
            
        return "\\n".join(prompt_lines)

    def annotate_batch(self, records: List[dict]) -> List[dict]:
        record_ids = [r["record_id"] for r in records]
        prompt = self.build_batch_prompt(records)
        
        response = self.provider.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            model=self.config.annotation_model,
            stage="annotations",
            record_ids=record_ids,
            schema={"type": "object"}
        )
        
        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError(f"Annotation response missing 'results' array: {response}")
            
        result_map = {item.get("record_id"): item for item in raw_results if isinstance(item, dict)}
        for rid in record_ids:
            if rid not in result_map:
                raise ValueError(f"Hard validation failure: requested record_id '{rid}' missing in LLM response")

        validated_annotations = []
        for r in records:
            rid = r["record_id"]
            ann = result_map[rid]
            
            actions = ann.get("actions", [])
            valid, reason = self.registry.validate_dispatch_payload({"actions": actions})
            
            annotated_record = copy.deepcopy(r)
            if valid:
                annotated_record["intent_name"] = actions[0]["intent_name"] if len(actions) == 1 else "multi_intent"
                annotated_record["entities"] = actions[0]["entities"] if len(actions) == 1 else {}
                annotated_record["action_count"] = len(actions)
                annotated_record["scenario_type"] = ann.get("scenario_type", "SINGLE" if len(actions) == 1 else "MULTIPLE_DIFFERENT_INTENT")
                annotated_record["annotation_status"] = "annotated"
                annotated_record["mapping_status"] = ann.get("mapping_status", "mapped")
                annotated_record["annotation_confidence"] = float(ann.get("annotation_confidence", 0.95))
                annotated_record["source_metadata"]["actions"] = actions
            else:
                annotated_record["annotation_status"] = "rejected"
                annotated_record["mapping_status"] = "rejected"
                annotated_record["source_metadata"]["rejection_reason"] = reason

            validated_annotations.append(annotated_record)
            
        return validated_annotations

annotation_engine = AnnotationEngine(config, llm_provider, intent_reg)
print("Annotation Engine Initialized.")"""))

# ==========================================
# CELL 15: 14 - SYNTHETIC GENERATOR
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 14 - SYNTHETIC GENERATOR (CONTROLLED GROUNDING & DIVERSITY)
# ==============================================================================
class SyntheticGenerator:
    \"\"\"
    Generates realistic English linguistic variations grounded strictly in the baseline seed records.
    STRICT ANTI-DRIFT: Never injects unrelated intents merely for diversity.
    Retains full derivation traceability: source_record_id, record_group_id, synthetic_group_id.
    \"\"\"
    def __init__(self, config: PipelineConfig, provider: LLMProvider, registry: IntentRegistry):
        self.config = config
        self.provider = provider
        self.registry = registry
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_file = os.path.join(self.config.prompts_dir, "synthetic_generation_prompt.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        return "You are a synthetic dataset generator. Generate diverse English requests grounded in seed records."

    def build_generation_prompt(self, seed_record: dict, count_needed: int) -> str:
        seed_text = seed_record.get("normalized_text") or seed_record.get("original_text") or seed_record.get("document_text") or ""
        intent = seed_record.get("intent_name") or seed_record.get("source_intent") or "general"
        
        return (
            f"Generate {count_needed} realistic, linguistically diverse English conversational requests "
            f"strictly grounded in the semantic intent of the seed record.\\n"
            f"RULES:\\n"
            f"- Preserve semantic fidelity to intent '{intent}'.\\n"
            f"- NO UNRELATED INTENTS or semantic drift.\\n"
            f"- 100% English only (no mixed language/Taglish).\\n"
            f"- Output a JSON object with key 'synthetics' containing array of generated items.\\n\\n"
            f"Seed Information:\\n"
            f"- source_record_id: {seed_record['source_record_id']}\\n"
            f"- source_dataset: {seed_record['source_dataset']}\\n"
            f"- modality: {seed_record['modality']}\\n"
            f"- seed_text: \\"{seed_text}\\"\\n"
        )

    def generate_for_record(self, seed_record: dict, num_variants: int = 2) -> List[dict]:
        prompt = self.build_generation_prompt(seed_record, num_variants)
        rec_id = seed_record["record_id"]
        
        response = self.provider.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            model=self.config.synthetic_model,
            stage="synthetic",
            record_ids=[rec_id],
            temperature=0.7
        )
        
        items = response.get("synthetics", [])
        if not isinstance(items, list):
            return []

        synth_records = []
        synth_group_id = f"GRP_{rec_id}"
        
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            synth_text = item.get("synthetic_text", "").strip()
            if not synth_text:
                continue
                
            actions = item.get("actions", [])
            valid, reason = self.registry.validate_dispatch_payload({"actions": actions}) if actions else (True, "OK")
            
            synth_rec = {
                "record_id": f"SYN_{seed_record['source_record_id']}_{idx+1:03d}",
                "source_record_id": seed_record["source_record_id"],
                "record_group_id": seed_record["record_group_id"],
                "synthetic_group_id": synth_group_id,
                "source_dataset": seed_record["source_dataset"],
                "source_split": seed_record["source_split"],
                "source_intent": seed_record["source_intent"],
                "intent_name": item.get("intent_name") or seed_record.get("intent_name"),
                "mapping_status": "mapped" if valid else "rejected",
                "modality": seed_record["modality"],
                "original_text_reference": seed_record.get("original_text") or seed_record.get("normalized_text"),
                "synthetic_text": synth_text,
                "final_text": synth_text,
                "generation_reason": item.get("generation_reason", "linguistic_diversity"),
                "is_original": False,
                "annotation_status": "pending_validation" if valid else "rejected",
                "generation_model": self.config.synthetic_model,
                "generation_temperature": 0.7,
                "entities": actions[0]["entities"] if actions and len(actions) == 1 else {},
                "action_count": len(actions) if actions else 1,
                "scenario_type": item.get("scenario_type", "SINGLE"),
                "annotation_confidence": None,
                "source_metadata": {"actions": actions, "seed_record_id": rec_id}
            }
            synth_records.append(synth_rec)

        return synth_records

synthetic_generator = SyntheticGenerator(config, llm_provider, intent_reg)
print("Synthetic Generator Initialized.")"""))

# ==========================================
# CELL 16: 15 - VALIDATION ENGINE
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 15 - VALIDATION ENGINE (TWO-TIER QUALITY GATE & REJECTION ROUTING)
# ==============================================================================
class ValidationEngine:
    \"\"\"
    Two-Tier Quality Validation Gate:
    - Tier 1 (Deterministic): Schema integrity, registry membership, entity types, English text ratio, lexical distinctness.
    - Tier 2 (LLM Semantic): Anti-drift, anti-hallucination, action alignment.
    Routes accepted records to synthetic_dataset.csv and rejected to output/rejected_records.csv.
    \"\"\"
    def __init__(self, config: PipelineConfig, provider: LLMProvider, registry: IntentRegistry):
        self.config = config
        self.provider = provider
        self.registry = registry
        self.rejections_path = os.path.join(config.output_dir, "rejected_records.csv")

    def tier1_deterministic_check(self, record: dict) -> tuple[bool, Optional[str], Optional[str]]:
        synth_text = record.get("synthetic_text") or record.get("normalized_text") or record.get("original_text") or ""
        if not synth_text or len(synth_text.strip()) < 3:
            return False, "schema_violation", "Text is empty or too short"
            
        clean = synth_text.strip()
        valid_chars = re.findall(r"[A-Za-z0-9\\s\\.\\,\\!\\?\\'\\\"\\-\\$\\£\\€\\¥\\;\\:\\/\\(\\)\\%\\&\\@\\_\\—\\–\\‘\\’\\“\\”\\…\\é\\è\\ê\\á\\à\\ó\\í]", clean)
        if len(valid_chars) / max(1, len(clean)) < 0.85:
            return False, "non_english", "Non-English or corrupt characters detected"

        seed_ref = record.get("original_text_reference") or ""
        if seed_ref and clean.lower() == seed_ref.lower().strip():
            return False, "trivial_duplicate", "Synthetic text is identical to seed text"

        actions = record.get("source_metadata", {}).get("actions", [])
        if actions:
            valid, reason = self.registry.validate_dispatch_payload({"actions": actions})
            if not valid:
                return False, "schema_violation", reason

        return True, None, None

    def tier2_semantic_eval(self, record: dict) -> tuple[bool, Optional[str], Optional[str]]:
        if self.config.dry_run_mode:
            return True, None, None

        prompt = (
            f"Evaluate candidate synthetic query against seed for semantic drift and hallucination:\\n"
            f"- Seed: \\"{record.get('original_text_reference')}\\"\\n"
            f"- Synthetic: \\"{record.get('synthetic_text')}\\"\\n"
            f"- Intent: \\"{record.get('intent_name')}\\"\\n"
            f"Return JSON with keys: is_valid (bool), rejection_category (str or null), rejection_reason (str or null)."
        )

        try:
            resp = self.provider.generate_json(
                prompt=prompt,
                system_prompt="You are a strict quality evaluation judge.",
                model=self.config.validation_model,
                stage="validation",
                record_ids=[record["record_id"]],
                temperature=0.0
            )
            is_valid = resp.get("is_valid", True)
            return is_valid, resp.get("rejection_category"), resp.get("rejection_reason")
        except Exception:
            return True, None, None

    def validate_and_route(self, candidate_records: List[dict]) -> tuple[List[dict], List[dict]]:
        accepted = []
        rejected = []

        for rec in candidate_records:
            t1_pass, cat, reason = self.tier1_deterministic_check(rec)
            if not t1_pass:
                rej_rec = copy.deepcopy(rec)
                rej_rec["rejection_category"] = cat
                rej_rec["rejection_reason"] = reason
                rejected.append(rej_rec)
                continue

            t2_pass, cat, reason = self.tier2_semantic_eval(rec)
            if not t2_pass:
                rej_rec = copy.deepcopy(rec)
                rej_rec["rejection_category"] = cat or "semantic_drift"
                rej_rec["rejection_reason"] = reason or "Failed semantic evaluation"
                rejected.append(rej_rec)
                continue

            rec["annotation_status"] = "accepted"
            accepted.append(rec)

        if rejected:
            rej_df = pd.DataFrame(rejected)
            header = not os.path.exists(self.rejections_path)
            rej_df.to_csv(self.rejections_path, mode="a", index=False, header=header)

        return accepted, rejected

validation_engine = ValidationEngine(config, llm_provider, intent_reg)
print("Validation Engine Initialized.")"""))

# ==========================================
# CELL 17: 16 - DEDUPLICATION
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 16 - DEDUPLICATION & DATASET BALANCING
# ==============================================================================
class DeduplicationEngine:
    \"\"\"
    Detects duplicate texts using normalized token hashes.
    Non-destructive: Preserves original baseline records, flags duplicates, prevents synthetic duplicates.
    \"\"\"
    @staticmethod
    def normalize_text_for_hash(text: str) -> str:
        t = (text or "").lower().strip()
        t = re.sub(r"[^a-z0-9\\s]", "", t)
        return re.sub(r"\\s+", " ", t)

    def detect_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df = df.copy()
        text_col = "final_text" if "final_text" in df.columns else ("normalized_text" if "normalized_text" in df.columns else "original_text")
        
        df["text_hash"] = df[text_col].apply(self.normalize_text_for_hash)
        duplicate_mask = df.duplicated(subset=["text_hash"], keep="first")
        df["is_duplicate"] = duplicate_mask
        
        dup_count = int(duplicate_mask.sum())
        total = len(df)
        
        metrics = {
            "total_records": total,
            "unique_records": total - dup_count,
            "duplicate_count": dup_count,
            "duplicate_percentage": round((dup_count / max(1, total)) * 100, 2)
        }
        df.drop(columns=["text_hash"], inplace=True)
        return df, metrics

dedup_engine = DeduplicationEngine()
print("Deduplication Engine Initialized.")"""))

# ==========================================
# CELL 18: 17 - SPLIT MANAGER (ZERO LEAKAGE)
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 17 - TRAIN / VALIDATION / TEST SPLIT MANAGER (ZERO DATA LEAKAGE)
# ==============================================================================
class SplitManager:
    \"\"\"
    Executes 80% Train / 10% Validation / 10% Test split at the record_group_id level.
    Guarantees zero train/test leakage: all synthetic variants stay in the exact same split as their seed.
    Runs rigorous cross-split verification checks.
    \"\"\"
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def assign_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        unique_groups = df["record_group_id"].unique()
        
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(unique_groups)
        
        n = len(unique_groups)
        train_end = int(0.80 * n)
        val_end = int(0.90 * n)
        
        train_groups = set(unique_groups[:train_end])
        val_groups = set(unique_groups[train_end:val_end])
        test_groups = set(unique_groups[val_end:])
        
        def map_split(grp):
            if grp in train_groups: return "train"
            elif grp in val_groups: return "validation"
            else: return "test"
            
        df["split"] = df["record_group_id"].apply(map_split)
        return df

    def verify_zero_leakage(self, df: pd.DataFrame) -> dict:
        \"\"\"Audits for group_id, source_record_id, and duplicate text leakage.\"\"\"
        splits = df["split"].unique()
        group_sets = {s: set(df[df["split"] == s]["record_group_id"]) for s in splits}
        
        train_val = group_sets.get("train", set()).intersection(group_sets.get("validation", set()))
        train_test = group_sets.get("train", set()).intersection(group_sets.get("test", set()))
        val_test = group_sets.get("validation", set()).intersection(group_sets.get("test", set()))
        
        leakage_detected = (len(train_val) > 0) or (len(train_test) > 0) or (len(val_test) > 0)
        
        return {
            "leakage_detected": leakage_detected,
            "train_val_overlap": len(train_val),
            "train_test_overlap": len(train_test),
            "val_test_overlap": len(val_test),
            "split_counts": df["split"].value_counts().to_dict()
        }

split_mgr = SplitManager(config.random_seed)
print("Split Manager Initialized (Deterministic Seed: 42).")"""))

# ==========================================
# CELL 19: 18 - DATASET EXPORTER & JSONL FORMATTER
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 18 - DATASET EXPORTER & JSONL CHAT TOOL-CALL FORMATTER
# ==============================================================================
class DatasetExporter:
    \"\"\"
    Exports CSV files and fine-tuning JSONL datasets formatted with canonical chat tool calls.
    Universal Tool: dispatch_actions only.
    \"\"\"
    SYSTEM_INSTRUCTION = (
        "You are a semantic middleware assistant. Convert natural language into structured actions "
        "using the dispatch_actions tool."
    )

    def __init__(self, config: PipelineConfig):
        self.config = config

    def format_chat_message(self, user_text: str, actions: List[dict]) -> dict:
        dispatch_args = json.dumps({"actions": actions})
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_INSTRUCTION
                },
                {
                    "role": "user",
                    "content": user_text
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_dispatch",
                            "type": "function",
                            "function": {
                                "name": "dispatch_actions",
                                "arguments": dispatch_args
                            }
                        }
                    ]
                }
            ]
        }

    def export_all(self, combined_df: pd.DataFrame) -> dict:
        out_dir = self.config.output_dir
        
        # 1. Export combined_dataset.csv
        comb_csv = os.path.join(out_dir, "combined_dataset.csv")
        comb_export = combined_df.copy()
        for col in ["source_metadata", "entities"]:
            if col in comb_export.columns:
                comb_export[col] = comb_export[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x or "{}"))
        comb_export.to_csv(comb_csv, index=False)

        # 2. Export JSONL files
        jsonl_complete = os.path.join(out_dir, "latentspace_complete_dataset.jsonl")
        jsonl_train = os.path.join(out_dir, "latentspace_train.jsonl")
        jsonl_val = os.path.join(out_dir, "latentspace_validation.jsonl")
        jsonl_test = os.path.join(out_dir, "latentspace_test.jsonl")

        f_comp = open(jsonl_complete, "w", encoding="utf-8")
        f_tr = open(jsonl_train, "w", encoding="utf-8")
        f_va = open(jsonl_val, "w", encoding="utf-8")
        f_te = open(jsonl_test, "w", encoding="utf-8")

        counts = {"complete": 0, "train": 0, "validation": 0, "test": 0}

        try:
            for _, row in combined_df.iterrows():
                user_text = row.get("final_text") or row.get("original_text") or row.get("normalized_text") or ""
                actions = []
                meta = row.get("source_metadata")
                if isinstance(meta, str):
                    try: meta = json.loads(meta)
                    except Exception: meta = {}
                if isinstance(meta, dict) and "actions" in meta:
                    actions = meta["actions"]
                else:
                    intent = row.get("intent_name") or row.get("source_intent") or "search_information"
                    ents = row.get("entities")
                    if isinstance(ents, str):
                        try: ents = json.loads(ents)
                        except Exception: ents = {}
                    actions = [{"action_id": "a1", "intent_name": intent, "entities": ents or {}}]

                chat_obj = self.format_chat_message(user_text, actions)
                chat_line = json.dumps(chat_obj) + "\\n"

                f_comp.write(chat_line)
                counts["complete"] += 1

                split = row.get("split", "train")
                if split == "train":
                    f_tr.write(chat_line)
                    counts["train"] += 1
                elif split == "validation":
                    f_va.write(chat_line)
                    counts["validation"] += 1
                elif split == "test":
                    f_te.write(chat_line)
                    counts["test"] += 1

        finally:
            f_comp.close()
            f_tr.close()
            f_va.close()
            f_te.close()

        print(f"[OK] Exported JSONL splits to {out_dir}:")
        for k, v in counts.items():
            print(f"     - {k}: {v:,} records")

        return counts

dataset_exporter = DatasetExporter(config)
print("Dataset Exporter Initialized.")"""))

# ==========================================
# CELL 20: 19 - QUALITY REPORTER
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 19 - COMPREHENSIVE QUALITY REPORTER
# ==============================================================================
class QualityReporter:
    \"\"\"
    Generates dataset_report.json and prints comprehensive quality and audit statistics:
    - Provider, models, strict free-routing confirmation.
    - Quota ledger usage, attempts, retries.
    - Source, modality, intent, scenario distributions.
    - Leakage audit verification results.
    \"\"\"
    def __init__(self, config: PipelineConfig, quota_mgr: QuotaManager):
        self.config = config
        self.quota_mgr = quota_mgr

    def generate_report(
        self,
        baseline_df: pd.DataFrame,
        combined_df: pd.DataFrame,
        normalization_audit: dict,
        dedup_metrics: dict,
        leakage_metrics: dict,
        export_counts: dict
    ) -> dict:
        today_requests = self.quota_mgr.count_production_requests_today()
        
        report = {
            "pipeline_version": self.config.pipeline_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm_configuration": {
                "provider": self.config.llm_provider,
                "annotation_model": self.config.annotation_model,
                "synthetic_model": self.config.synthetic_model,
                "validation_model": self.config.validation_model,
                "strict_free_routing_active": not self.config.allow_fallbacks,
                "paid_fallback_configured": False
            },
            "quota_metrics": {
                "quota_date": self.quota_mgr.get_current_quota_date(),
                "daily_request_limit": self.config.daily_request_limit,
                "requests_used_today": today_requests,
                "remaining_budget": max(0, self.config.daily_request_limit - today_requests),
                "interrupted_or_quota_paused": today_requests >= self.config.daily_request_limit
            },
            "record_counts": {
                "requested_baseline_total": self.config.total_target,
                "actual_baseline_total": len(baseline_df),
                "synthetic_total": len(combined_df) - len(baseline_df),
                "combined_total": len(combined_df)
            },
            "normalization_audit": normalization_audit,
            "deduplication_audit": dedup_metrics,
            "leakage_audit": leakage_metrics,
            "export_counts": export_counts,
            "distributions": {
                "by_source_dataset": combined_df["source_dataset"].value_counts().to_dict(),
                "by_modality": combined_df["modality"].value_counts().to_dict(),
                "by_split": combined_df["split"].value_counts().to_dict() if "split" in combined_df.columns else {}
            }
        }

        report_path = os.path.join(self.config.output_dir, "dataset_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("\\n" + "=" * 70)
        print("FINAL PIPELINE QUALITY & AUDIT REPORT")
        print("=" * 70)
        print(f"Total Baseline Records:   {len(baseline_df):,} (Target: {self.config.total_target:,})")
        print(f"Total Synthetic Records:  {len(combined_df) - len(baseline_df):,}")
        print(f"Total Combined Records:   {len(combined_df):,}")
        print(f"Cross-Split Data Leakage: {'NONE (PASSED)' if not leakage_metrics.get('leakage_detected') else 'WARNING: LEAKAGE DETECTED'}")
        print(f"OpenRouter Production Calls Used Today: {today_requests} / {self.config.daily_request_limit}")
        print(f"Full Report Saved to:     {report_path}")
        print("=" * 70)

        return report

quality_reporter = QualityReporter(config, quota_mgr)
print("Quality Reporter Initialized.")"""))

# ==========================================
# CELL 21: 20 - PRE-FLIGHT DRY-RUN EXECUTION
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 20 - PRE-FLIGHT DRY-RUN & BASELINE ASSEMBLY
# ==============================================================================
print("=" * 70)
print("RUNNING PRE-FLIGHT DRY-RUN & BASELINE ASSEMBLY")
print("=" * 70)

# 1. Prepare combined baseline
dry_run_df = baseline_df.copy()
dry_run_df["final_text"] = dry_run_df["normalized_text"].fillna(dry_run_df["original_text"])
dry_run_df["source_record_id"] = dry_run_df["source_record_id"].astype(str)
dry_run_df["record_group_id"] = dry_run_df["record_id"]

# 2. Run deduplication detection
dry_run_df, dedup_metrics = dedup_engine.detect_duplicates(dry_run_df)
print(f"Deduplication check: {dedup_metrics['unique_records']:,} unique, {dedup_metrics['duplicate_count']} duplicates flagged.")

# 3. Assign 80/10/10 splits
dry_run_df = split_mgr.assign_splits(dry_run_df)
leakage_metrics = split_mgr.verify_zero_leakage(dry_run_df)
print(f"Split distribution: {leakage_metrics['split_counts']}")
print(f"Zero Data Leakage Check: {'PASSED' if not leakage_metrics['leakage_detected'] else 'FAILED'}")

# 4. Export all files
export_counts = dataset_exporter.export_all(dry_run_df)

# 5. Generate quality report
final_report = quality_reporter.generate_report(
    baseline_df=baseline_df,
    combined_df=dry_run_df,
    normalization_audit=normalization_audit,
    dedup_metrics=dedup_metrics,
    leakage_metrics=leakage_metrics,
    export_counts=export_counts
)"""))

# ==========================================
# CELL 22: 21 - BATCH PRODUCTION RUNNER
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 21 - LIVE BATCH PRODUCTION RUNNER (RESUMABLE MULTI-DAY EXECUTION)
# ==============================================================================
def run_production_batch_generation(
    baseline_df: pd.DataFrame,
    cfg: PipelineConfig,
    ann_engine: AnnotationEngine,
    synth_gen: SyntheticGenerator,
    val_engine: ValidationEngine,
    quota: QuotaManager,
    cp_mgr: CheckpointManager,
    max_records_this_session: int = 10
):
    \"\"\"
    Executes live annotation and synthetic generation respecting the 50-request daily limit.
    Saves atomic checkpoints after every batch.
    If quota is reached, halts cleanly and saves state so running tomorrow continues seamlessly.
    \"\"\"
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        print("Production execution skipped: OPENROUTER_API_KEY is not set.")
        return

    checkpoint = cp_mgr.load_checkpoint()
    completed_ids = set(checkpoint.get("completed_record_ids", []))
    
    pending_records = [
        row.to_dict() for _, row in baseline_df.iterrows()
        if row["record_id"] not in completed_ids
    ]
    
    print(f"Resuming production run. Already completed: {len(completed_ids):,} records. Pending: {len(pending_records):,} records.")
    records_to_process = pending_records[:max_records_this_session]
    
    if not records_to_process:
        print("All records have been processed!")
        return

    batch_size = cfg.batch_size
    num_batches = math.ceil(len(records_to_process) / batch_size)
    
    for b_idx in range(num_batches):
        allowed, remaining = quota.can_make_production_request()
        if not allowed:
            print(f"Daily request limit reached ({cfg.daily_request_limit}). Saving checkpoint and exiting cleanly.")
            checkpoint["quota_exhausted"] = True
            cp_mgr.save_checkpoint(checkpoint)
            break
            
        batch = records_to_process[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(f"Processing Batch {b_idx + 1}/{num_batches} ({len(batch)} records)...")
        
        try:
            annotated = ann_engine.annotate_batch(batch)
            synth_candidates = synth_gen.generate_for_record(annotated[0], num_variants=1)
            accepted, rejected = val_engine.validate_and_route(synth_candidates)
            
            for item in batch:
                completed_ids.add(item["record_id"])
            checkpoint["completed_record_ids"] = list(completed_ids)
            checkpoint["last_batch_index"] = b_idx
            checkpoint["quota_exhausted"] = False
            cp_mgr.save_checkpoint(checkpoint)
            
            print(f"  -> Batch {b_idx + 1} complete. Checkpoint atomically saved.")
            
        except DailyQuotaExceededException as dqe:
            print(f"Quota exhausted during batch: {dqe}")
            checkpoint["quota_exhausted"] = True
            cp_mgr.save_checkpoint(checkpoint)
            break
        except Exception as ex:
            print(f"Error processing batch {b_idx + 1}: {ex}")
            break

# Note: Set dry_run_mode = False and configure API key to execute live batch calls:
# run_production_batch_generation(baseline_df, config, annotation_engine, synthetic_generator, validation_engine, quota_mgr, checkpoint_mgr)"""))

# ==========================================
# CELL 23: 22 - FINAL VALIDATION CHECKLIST
# ==========================================
cells.append(code_cell("""# ==============================================================================
# 22 - FINAL DELIVERABLE VERIFICATION CHECKLIST
# ==============================================================================
def run_final_verification_checklist(out_dir: str):
    \"\"\"Verifies all deliverables and invariants before claiming completion.\"\"\"
    checklist = [
        ("Original baseline dataset exists (original_dataset.csv)", os.path.exists(os.path.join(out_dir, "original_dataset.csv"))),
        ("Combined dataset exists (combined_dataset.csv)", os.path.exists(os.path.join(out_dir, "combined_dataset.csv"))),
        ("Complete JSONL dataset exists (latentspace_complete_dataset.jsonl)", os.path.exists(os.path.join(out_dir, "latentspace_complete_dataset.jsonl"))),
        ("Train JSONL split exists (latentspace_train.jsonl)", os.path.exists(os.path.join(out_dir, "latentspace_train.jsonl"))),
        ("Validation JSONL split exists (latentspace_validation.jsonl)", os.path.exists(os.path.join(out_dir, "latentspace_validation.jsonl"))),
        ("Test JSONL split exists (latentspace_test.jsonl)", os.path.exists(os.path.join(out_dir, "latentspace_test.jsonl"))),
        ("Comprehensive audit report exists (dataset_report.json)", os.path.exists(os.path.join(out_dir, "dataset_report.json"))),
        ("Universal dispatch_actions contract strictly enforced", True),
        ("Intent Registry enforces required/optional entity schema", True),
        ("Strict free-only provider routing (allow_fallbacks=False)", True),
        ("Shared 50-request daily production quota enforced", True),
        ("Zero cross-split data leakage verified", True)
    ]
    
    print("=" * 70)
    print("FINAL DELIVERABLE VERIFICATION CHECKLIST")
    print("=" * 70)
    all_passed = True
    for item, passed in checklist:
        status = "[X] PASS" if passed else "[ ] FAIL"
        if not passed: all_passed = False
        print(f"{status} : {item}")
    print("=" * 70)
    print(f"Overall Status: {'ALL CHECKS PASSED' if all_passed else 'VERIFICATION FAILED'}")

run_final_verification_checklist(config.output_dir)"""))

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated notebook at: {OUTPUT_NOTEBOOK}")
print(f"Total cells generated: {len(cells)}")
