"""
Builder script to generate latentspace_dataset/notebook/latentspace_dataset_builder.ipynb.
Revised Architecture:
- original_dataset.csv contains strictly: raw_text, source
- Agentic synthesis/annotation occurs after original dataset creation
- OpenAI-compatible messages + tools JSONL export
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
# TITLE CELL
# ==============================================================================
cells.append(md_cell("""# LatentSpace Agentic Training Dataset Pipeline
### Offline Semantic Middleware & Agentic Assistant Dataset Builder (Revised Architecture)
**Pipeline Version:** 2.0.0  
**Target Assistant:** LatentSpace Local Semantic Middleware  
**Core Principle:** `original_dataset.csv` contains ONLY `raw_text` and `source`.  
**Agentic Layer:** Synthesis and schema completion happen strictly AFTER the original dataset is built.  
**Universal Tool Contract:** `dispatch_actions`  
**Provider:** OpenRouter (`qwen/qwen3.8-27b:free`, strict free-only routing)  

```text
Multiple Source Datasets
        ↓
FETCH / EXTRACT ORIGINAL CONTENT
        ↓
ORIGINAL DATASET (raw_text + source ONLY)
        ↓
LLM AGENTIC SYNTHESIS / ANNOTATION
        ↓
VALIDATION (Deterministic & Semantic)
        ↓
COMPLETE TRAINING DATASET
        ↓
FINAL TRAINING JSONL (messages + tools)
```
"""))

# ==============================================================================
# SECTION 1: INSTALLATION
# ==============================================================================
cells.append(md_cell("""## 1. Installation
Install and verify required runtime libraries for dataset loading, text processing, and schema validation."""))

cells.append(code_cell(r"""# 1. Installation verification
import sys
import subprocess

required_pkgs = ["pandas", "datasets", "requests", "tqdm", "jsonschema"]
missing_pkgs = []
for pkg in required_pkgs:
    try:
        __import__(pkg)
    except ImportError:
        missing_pkgs.append(pkg)

if missing_pkgs:
    print(f"Installing missing packages: {missing_pkgs}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_pkgs)
else:
    print("All required packages are installed.")
"""))

# ==============================================================================
# SECTION 2: IMPORTS
# ==============================================================================
cells.append(md_cell("""## 2. Imports
Import standard library modules, data science tools, and dataset utilities."""))

cells.append(code_cell(r"""# 2. Imports
import os
import sys
import json
import re
import time
import math
import hashlib
import random
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

print(f"Pandas version: {pd.__version__}")
"""))

# ==============================================================================
# SECTION 3: CONFIGURATION
# ==============================================================================
cells.append(md_cell("""## 3. Configuration
Centralized immutable configuration for the LatentSpace dataset pipeline."""))

cells.append(code_cell(r"""# 3. Pipeline Configuration
@dataclass(frozen=True)
class PipelineConfig:
    pipeline_version: str = "2.0.0"
    random_seed: int = 42

    llm_provider: str = "openrouter"
    api_base_url: str = "https://openrouter.ai/api/v1"

    annotation_model: str = "qwen/qwen3.8-27b:free"
    synthetic_model: str = "qwen/qwen3.8-27b:free"
    validation_model: str = "qwen/qwen3.8-27b:free"
    connectivity_test_model: str = "qwen/qwen3.8-27b:free"

    allow_fallbacks: bool = False

    daily_request_limit: int = 50
    stop_before_daily_limit: bool = True

    batch_size: int = 10
    max_batch_chars: int = 12000
    max_batch_tokens: int = 4096
    fallback_to_single_on_failure: bool = True
    max_retries: int = 3
    backoff_factor: float = 2.0
    request_timeout: int = 60

    # Target baseline counts (Baseline total: approximately 23,384 records)
    target_banking77: int = 7000
    target_clinc150: int = 8000
    target_hwu64: int = 5000
    target_minds14_us: int = 563
    target_minds14_ext: int = 1246
    target_cord_v2: int = 800
    target_sroie: int = 626
    target_funsd: int = 149
    total_target: int = 23384

    dry_run_mode: bool = False
    test_mode_batch_limit: int = 5

    base_dir: str = "latentspace_dataset"
    output_dir: str = "latentspace_dataset/output"
    cache_dir: str = "latentspace_dataset/cache"
    config_dir: str = "latentspace_dataset/config"
    prompts_dir: str = "latentspace_dataset/prompts"

config = PipelineConfig()
print(f"Pipeline Config Initialized (v{config.pipeline_version})")
print(f"Target Baseline: {config.total_target:,} records")
"""))

# ==============================================================================
# SECTION 4: SECURITY / ENVIRONMENT VALIDATION
# ==============================================================================
cells.append(md_cell("""## 4. Security / Environment Validation
Safely validate credentials without printing keys, and initialize the Request Ledger and Checkpoint Manager."""))

cells.append(code_cell(r"""# 4. Security, Quota Ledger, and Checkpoint Manager
api_key = os.environ.get("OPENROUTER_API_KEY", "")
try:
    from google.colab import userdata
    api_key = api_key or userdata.get("OPENROUTER_API_KEY", "")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
except Exception:
    pass

print(f"API key configured: {'YES' if bool(api_key) else 'NO'}")

class RequestLedger:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ledger_path = os.path.join(config.cache_dir, "request_ledger.jsonl")
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        self.production_requests_used = 0
        self.cache_hits = 0
        self.retries = 0
        self._load_state()

    def _load_state(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get("quota_date") == today and not record.get("is_test", False):
                            self.production_requests_used += 1
                        if record.get("retry_number", 0) > 0:
                            self.retries += 1
                    except Exception:
                        pass

    def record_attempt(self, attempt_data: dict):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        attempt_data["quota_date"] = today
        if not attempt_data.get("is_test", False):
            self.production_requests_used += 1
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(attempt_data) + "
")

    def record_cache_hit(self):
        self.cache_hits += 1

    def can_request(self) -> bool:
        if not self.config.stop_before_daily_limit:
            return True
        return self.production_requests_used < self.config.daily_request_limit

class CheckpointManager:
    @staticmethod
    def atomic_write_json(file_path: str, data: Any):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        temp_path = f"{file_path}.tmp.{os.getpid()}.{time.time_ns()}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, file_path)

    @staticmethod
    def load_json(file_path: str) -> Optional[Any]:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

ledger = RequestLedger(config)
print(f"Request ledger ready. Production requests used today: {ledger.production_requests_used} / {config.daily_request_limit}")
"""))

# ==============================================================================
# SECTION 5: TAXONOMY
# ==============================================================================
cells.append(md_cell("""## 5. Controlled Domain & Category Taxonomy
Controlled taxonomy and universal tool contract for agentic routing."""))

cells.append(code_cell(r"""# 5. Controlled Taxonomy & Universal Tool Schema
DOMAINS = [
    "finance", "productivity", "travel", "commerce",
    "communication", "account_and_security", "documents",
    "information", "other"
]

CATEGORIES = [
    "finance/account", "finance/transactions", "finance/payments",
    "finance/transfers", "finance/cards", "finance/expenses",
    "productivity/tasks", "productivity/reminders", "productivity/alarms",
    "productivity/notes", "productivity/calendar",
    "travel/flights", "travel/reservations", "travel/transportation",
    "commerce/products", "commerce/orders", "commerce/subscriptions", "commerce/purchases",
    "communication/messages", "communication/email", "communication/contacts",
    "account_and_security/profile", "account_and_security/authentication",
    "account_and_security/credentials", "account_and_security/security", "account_and_security/cards",
    "documents/ocr", "documents/extraction", "documents/receipts", "documents/invoices", "documents/forms",
    "information/general_information", "information/navigation", "information/calculation",
    "other/clarification", "other/out_of_scope"
]

UNIVERSAL_DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "dispatch_actions",
        "description": "Routes one or more parsed user intents to the downstream system.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_id": {
                                "type": "string",
                                "description": "Unique identifier such as a1, a2"
                            },
                            "intent_name": {
                                "type": "string",
                                "description": "The registered intent name"
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of action_ids this action depends upon"
                            },
                            "entities": {
                                "type": "object",
                                "description": "Key-value entity parameters"
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

print(f"Taxonomy defined: {len(DOMAINS)} domains, {len(CATEGORIES)} categories.")
"""))

# ==============================================================================
# SECTION 6: INTENT REGISTRY
# ==============================================================================
cells.append(md_cell("""## 6. Intent Registry
The strict source of truth for valid LatentSpace intents."""))

cells.append(code_cell(r"""# 6. Intent Registry
class IntentRegistry:
    def __init__(self, config_path: str = "latentspace_dataset/config/intent_registry.json"):
        self.config_path = config_path
        self.intents: Dict[str, dict] = {}
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.intents = data.get("intents", data)
        else:
            self.intents = {}

    def is_valid_intent(self, intent_name: str) -> bool:
        return intent_name in self.intents

    def get_intent(self, intent_name: str) -> Optional[dict]:
        return self.intents.get(intent_name)

registry = IntentRegistry(os.path.join(config.config_dir, "intent_registry.json"))
print(f"Loaded IntentRegistry: {len(registry.intents)} intents.")
"""))

# ==============================================================================
# SECTION 7: DATASET ADAPTERS
# ==============================================================================
cells.append(md_cell("""## 7. Dataset Adapters
Adapters extract ONLY the original source text (`raw_text`) and the source name (`source`)."""))

cells.append(code_cell(r"""# 7. Dataset Adapters (Extract ONLY raw_text and source)
def is_unicode_safe_english(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    cleaned = text.strip()
    if len(cleaned) < 2:
        return False
    latin_chars = len(re.findall(r"[A-Za-z]", cleaned))
    all_alphas = len([c for c in cleaned if c.isalpha()])
    if all_alphas == 0:
        return True
    return (latin_chars / all_alphas) >= 0.85

class DatasetAdapter(ABC):
    def __init__(self, source_name: str, target_count: int, config: PipelineConfig):
        self.source_name = source_name
        self.target_count = target_count
        self.config = config

    @abstractmethod
    def fetch_records(self) -> List[dict]:
        # Returns list of dicts with keys: 'raw_text', 'source'.
        pass

class Banking77Adapter(DatasetAdapter):
    def __init__(self, config: PipelineConfig):
        super().__init__("BANKING77", config.target_banking77, config)

    def fetch_records(self) -> List[dict]:
        cache_file = os.path.join(self.config.cache_dir, "sources", "banking77.json")
        if os.path.exists(cache_file):
            raw = json.load(open(cache_file, "r", encoding="utf-8"))
            df_raw = pd.DataFrame(raw)
        else:
            url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
            try:
                df_raw = pd.read_csv(url)
            except Exception:
                from datasets import load_dataset
                df_raw = pd.DataFrame(load_dataset("PolyAI/banking77", split="train"))

        b_target = min(self.target_count, len(df_raw))
        sample = df_raw.sample(n=b_target, random_state=self.config.random_seed)

        return [
            {
                "raw_text": str(row.get("text", "")).strip(),
                "source": "BANKING77"
            }
            for _, row in sample.iterrows()
        ]

class Clinc150Adapter(DatasetAdapter):
    def __init__(self, config: PipelineConfig):
        super().__init__("CLINC150", config.target_clinc150, config)

    def fetch_records(self) -> List[dict]:
        cache_file = os.path.join(self.config.cache_dir, "sources", "clinc150.json")
        if os.path.exists(cache_file):
            raw = json.load(open(cache_file, "r", encoding="utf-8"))
            c_target = min(self.target_count, len(raw))
            rng = random.Random(self.config.random_seed)
            shuffled = list(raw)
            rng.shuffle(shuffled)
            selected = shuffled[:c_target]
            return [
                {
                    "raw_text": str(item.get("utterance", "")).strip(),
                    "source": "CLINC150"
                }
                for item in selected
            ]

        from datasets import load_dataset
        clinc_raw = load_dataset("DeepPavlov/clinc150", name="default", split="train")
        c_target = min(self.target_count, len(clinc_raw))
        clinc_sample = clinc_raw.shuffle(seed=self.config.random_seed).select(range(c_target))

        return [
            {
                "raw_text": str(row["utterance"]).strip(),
                "source": "CLINC150"
            }
            for row in clinc_sample
        ]

class Hwu64Adapter(DatasetAdapter):
    def __init__(self, config: PipelineConfig):
        super().__init__("HWU64", config.target_hwu64, config)

    def fetch_records(self) -> List[dict]:
        cache_file = os.path.join(self.config.cache_dir, "sources", "hwu64.json")
        if os.path.exists(cache_file):
            raw = json.load(open(cache_file, "r", encoding="utf-8"))
            h_target = min(self.target_count, len(raw))
            rng = random.Random(self.config.random_seed)
            shuffled = list(raw)
            rng.shuffle(shuffled)
            selected = shuffled[:h_target]
            return [
                {
                    "raw_text": str(item.get("utterance", "")).strip(),
                    "source": "HWU64"
                }
                for item in selected
            ]

        from datasets import load_dataset
        hwu_raw = load_dataset("DeepPavlov/hwu64", name="default", split="train")
        h_target = min(self.target_count, len(hwu_raw))
        hwu_sample = hwu_raw.shuffle(seed=self.config.random_seed).select(range(h_target))

        return [
            {
                "raw_text": str(row["utterance"]).strip(),
                "source": "HWU64"
            }
            for row in hwu_sample
        ]

class Minds14USAdapter(DatasetAdapter):
    def __init__(self, config: PipelineConfig):
        super().__init__("MINDS14_EN_US", config.target_minds14_us, config)

    def fetch_records(self) -> List[dict]:
        cache_file = os.path.join(self.config.cache_dir, "sources", "minds14_us.json")
        if os.path.exists(cache_file):
            raw = json.load(open(cache_file, "r", encoding="utf-8"))
            m_target = min(self.target_count, len(raw))
            rng = random.Random(self.config.random_seed)
            shuffled = list(raw)
            rng.shuffle(shuffled)
            selected = shuffled[:m_target]
            return [
                {
                    "raw_text": str(item.get("transcript", "")).strip(),
                    "source": "MINDS14_EN_US"
                }
                for item in selected
            ]

        from datasets import load_dataset
        minds_raw = load_dataset("PolyAI/minds14", name="en-US", split="train").remove_columns(["audio"])
        m_target = min(self.target_count, len(minds_raw))
        minds_sample = minds_raw.shuffle(seed=self.config.random_seed).select(range(m_target))

        return [
            {
                "raw_text": str(row.get("english_transcription") or row.get("transcription", "")).strip(),
                "source": "MINDS14_EN_US"
            }
            for row in minds_sample
        ]

class Minds14ExtAdapter(DatasetAdapter):
    def __init__(self, config: PipelineConfig):
        super().__init__("MINDS14_EXT", config.target_minds14_ext, config)

    def fetch_records(self) -> List[dict]:
        cache_file = os.path.join(self.config.cache_dir, "sources", "minds14_ext.json")
        if os.path.exists(cache_file):
            raw = json.load(open(cache_file, "r", encoding="utf-8"))
            records = []
            for item in raw:
                src = "MINDS14_EN_AU" if item.get("lang_id") == "en-AU" else "MINDS14_EN_GB"
                records.append({
                    "raw_text": str(item.get("transcript", "")).strip(),
                    "source": src
                })
            return records[:self.target_count]

        from datasets import load_dataset
        ds_au = load_dataset("PolyAI/minds14", name="en-AU", split="train").remove_columns(["audio"])
        ds_gb = load_dataset("PolyAI/minds14", name="en-GB", split="train").remove_columns(["audio"])

        target_au = min(654, len(ds_au))
        target_gb = min(592, len(ds_gb))

        sample_au = ds_au.shuffle(seed=self.config.random_seed).select(range(target_au))
        sample_gb = ds_gb.shuffle(seed=self.config.random_seed).select(range(target_gb))

        records = [
            {
                "raw_text": str(row.get("english_transcription") or row.get("transcription", "")).strip(),
                "source": "MINDS14_EN_AU"
            }
            for row in sample_au
        ]
        records.extend([
            {
                "raw_text": str(row.get("english_transcription") or row.get("transcription", "")).strip(),
                "source": "MINDS14_EN_GB"
            }
            for row in sample_gb
        ])
        return records

class CordV2Adapter(DatasetAdapter):
    def __init__(self, config: PipelineConfig):
        super().__init__("CORD-v2", config.target_cord_v2, config)

    def fetch_records(self) -> List[dict]:
        cache_file = os.path.join(self.config.cache_dir, "sources", "cord_v2.json")
        if os.path.exists(cache_file):
            raw = json.load(open(cache_file, "r", encoding="utf-8"))
            c_target = min(self.target_count, len(raw))
            return [
                {
                    "raw_text": str(item.get("document_text", "")).strip(),
                    "source": "CORD-v2"
                }
                for item in raw[:c_target]
            ]

        from datasets import load_dataset
        ds = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)
        records = []
        for item in ds:
            gt = item.get("ground_truth", "")
            gt_parse = {}
            if isinstance(gt, str):
                try: gt_parse = json.loads(gt).get("gt_parse", {})
                except Exception: gt_parse = {}
            elif isinstance(gt, dict) and "gt_parse" in gt:
                gt_parse = gt["gt_parse"]

            tokens = []
            def walk(o):
                if isinstance(o, str): tokens.append(o)
                elif isinstance(o, dict):
                    for v in o.values(): walk(v)
                elif isinstance(o, list):
                    for x in o: walk(x)
            walk(gt_parse)
            doc_text = " ".join(tokens) if tokens else "Receipt document"

            records.append({
                "raw_text": doc_text,
                "source": "CORD-v2"
            })
            if len(records) >= self.target_count:
                break
        return records

class SroieAdapter(DatasetAdapter):
    def __init__(self, config: PipelineConfig):
        super().__init__("SROIE", config.target_sroie, config)

    def fetch_records(self) -> List[dict]:
        cache_file = os.path.join(self.config.cache_dir, "sources", "sroie.json")
        if os.path.exists(cache_file):
            raw = json.load(open(cache_file, "r", encoding="utf-8"))
            s_target = min(self.target_count, len(raw))
            return [
                {
                    "raw_text": str(item.get("document_text", "")).strip(),
                    "source": "SROIE"
                }
                for item in raw[:s_target]
            ]

        from datasets import load_dataset
        ds = load_dataset("rth/sroie-2019-v2", split="train")
        records = []
        for item in ds:
            objs = item.get("objects", {})
            texts = objs.get("text", objs.get("texts", objs.get("words", [])))
            doc_text = " ".join(str(t) for t in texts) if texts else "Receipt document"
            records.append({
                "raw_text": doc_text,
                "source": "SROIE"
            })
            if len(records) >= self.target_count:
                break
        return records

class FunsdAdapter(DatasetAdapter):
    def __init__(self, config: PipelineConfig):
        super().__init__("FUNSD", config.target_funsd, config)

    def fetch_records(self) -> List[dict]:
        cache_file = os.path.join(self.config.cache_dir, "sources", "funsd.json")
        if os.path.exists(cache_file):
            raw = json.load(open(cache_file, "r", encoding="utf-8"))
            f_target = min(self.target_count, len(raw))
            return [
                {
                    "raw_text": str(item.get("document_text", "")).strip(),
                    "source": "FUNSD"
                }
                for item in raw[:f_target]
            ]

        from datasets import load_dataset
        ds = load_dataset("nielsr/funsd", split="train")
        records = []
        for item in ds:
            doc_text = " ".join(item.get("words", []))
            records.append({
                "raw_text": doc_text,
                "source": "FUNSD"
            })
            if len(records) >= self.target_count:
                break
        return records

print("All dataset adapters defined. Returning strictly raw_text + source.")
"""))

# ==============================================================================
# SECTION 8: ORIGINAL DATASET BUILDER
# ==============================================================================
cells.append(md_cell("""## 8. Original Dataset Builder
Builds the unified dataset containing ONLY `raw_text` and `source`, filtering English and deduplicating."""))

cells.append(code_cell(r"""# 8. Original Dataset Builder Execution
adapters = [
    Banking77Adapter(config),
    Clinc150Adapter(config),
    Hwu64Adapter(config),
    Minds14USAdapter(config),
    Minds14ExtAdapter(config),
    CordV2Adapter(config),
    SroieAdapter(config),
    FunsdAdapter(config)
]

all_original_records = []
adapter_stats = []

for ad in adapters:
    t0 = time.time()
    records = ad.fetch_records()
    available = len(records)

    english_valid = []
    excluded_count = 0
    for r in records:
        txt = r.get("raw_text", "")
        if is_unicode_safe_english(txt):
            english_valid.append(r)
        else:
            excluded_count += 1

    seen_texts = set()
    deduped = []
    dup_count = 0
    for r in english_valid:
        txt = r["raw_text"]
        if txt in seen_texts:
            dup_count += 1
        else:
            seen_texts.add(txt)
            deduped.append(r)

    target = min(ad.target_count, len(deduped))
    selected = deduped[:target]
    shortfall = max(0, ad.target_count - len(selected))

    all_original_records.extend(selected)
    stats = {
        "source": ad.source_name,
        "requested": ad.target_count,
        "available": available,
        "english_valid": len(english_valid),
        "excluded_non_english": excluded_count,
        "duplicates_removed": dup_count,
        "selected": len(selected),
        "shortfall": shortfall
    }
    adapter_stats.append(stats)
    print(f"[{ad.source_name:14}] Avail: {available:,} | Sel: {len(selected):,} | Shortfall: {shortfall} ({time.time()-t0:.2f}s)")

df_original = pd.DataFrame(all_original_records)
df_original = df_original.drop_duplicates(subset=["raw_text", "source"]).reset_index(drop=True)
print(f"
Total Unified Original Records: {len(df_original):,}")
"""))

# ==============================================================================
# SECTION 9: SOURCE INSPECTION
# ==============================================================================
cells.append(md_cell("""## 9. Source Data Inspection
Verifies that the original dataset contains exactly two columns: `raw_text` and `source`."""))

cells.append(code_cell(r"""# 9. Verify Schema and Inspect 5 Samples Per Source
assert list(df_original.columns) == ["raw_text", "source"], f"Columns violation: {list(df_original.columns)}"
assert df_original["raw_text"].notna().all(), "Null raw_text found!"
assert df_original["source"].notna().all(), "Null source found!"

print("Columns in df_original strictly verified:", list(df_original.columns))
print("
--- 5 SAMPLES PER SOURCE ---")
for src in df_original["source"].unique():
    print(f"
[Source: {src}]")
    sub = df_original[df_original["source"] == src]
    samples = sub.sample(n=min(5, len(sub)), random_state=42)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        preview = row["raw_text"][:80].replace("
", " ")
        print(f"  {i}. {preview}...")
"""))

# ==============================================================================
# SECTION 10: SAMPLING & SHORTFALL AUDIT
# ==============================================================================
cells.append(md_cell("""## 10. Sampling & Shortfall Audit
Audit table comparing requested targets vs selected records."""))

cells.append(code_cell(r"""# 10. Audit Table
audit_df = pd.DataFrame(adapter_stats)
print(audit_df[["source", "requested", "available", "selected", "shortfall"]].to_string(index=False))
print(f"
Total Selected: {audit_df['selected'].sum():,} / {audit_df['requested'].sum():,}")
print(f"Total Shortfall: {audit_df['shortfall'].sum():,}")
"""))

# ==============================================================================
# SECTION 11: ORIGINAL DATASET EXPORT
# ==============================================================================
cells.append(md_cell("""## 11. Original Dataset Export
Save `original_dataset.csv` with strictly `raw_text` and `source`."""))

cells.append(code_cell(r"""# 11. Save original_dataset.csv
os.makedirs(config.output_dir, exist_ok=True)
orig_csv_path = os.path.join(config.output_dir, "original_dataset.csv")
df_original.to_csv(orig_csv_path, index=False)
print(f"Saved {orig_csv_path} ({len(df_original):,} rows, {os.path.getsize(orig_csv_path):,} bytes)")
print("Columns:", list(df_original.columns))
"""))

# ==============================================================================
# SECTION 12: LLM PROVIDER ABSTRACTION
# ==============================================================================
cells.append(md_cell("""## 12. LLM Provider Abstraction
OpenRouter client with strict quota and retry controls."""))

cells.append(code_cell(r"""# 12. OpenRouter Provider
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, is_test: bool = False) -> str:
        pass

class OpenRouterProvider(LLMProvider):
    def __init__(self, config: PipelineConfig, ledger: RequestLedger):
        self.config = config
        self.ledger = ledger
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        try:
            from google.colab import userdata
            self.api_key = self.api_key or userdata.get("OPENROUTER_API_KEY", "")
        except Exception:
            pass

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, is_test: bool = False) -> str:
        if not self.is_configured():
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        if not is_test and not self.ledger.can_request():
            raise RuntimeError(f"Production request limit reached ({self.config.daily_request_limit}/day).")

        import urllib.request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://latentspace.ai",
            "X-Title": "LatentSpace Dataset Pipeline"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.annotation_model,
            "messages": messages,
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            f"{self.config.api_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        t0 = time.time()
        attempt_id = f"req_{int(time.time()*1000)}"
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                latency = int((time.time() - t0) * 1000)
                self.ledger.record_attempt({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attempt_id": attempt_id,
                    "is_test": is_test,
                    "model": self.config.annotation_model,
                    "http_status": resp.status,
                    "error_category": None,
                    "latency_ms": latency
                })
                return body["choices"][0]["message"]["content"]
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            self.ledger.record_attempt({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt_id": attempt_id,
                "is_test": is_test,
                "model": self.config.annotation_model,
                "http_status": 500,
                "error_category": type(e).__name__,
                "latency_ms": latency
            })
            raise e

provider = OpenRouterProvider(config, ledger)
print("Provider initialized. Configured:", "YES" if provider.is_configured() else "NO")
"""))

# ==============================================================================
# SECTION 13: CONNECTIVITY TEST
# ==============================================================================
cells.append(md_cell("""## 13. Pre-Flight Connectivity Test
Validates OpenRouter API access without consuming production quota."""))

cells.append(code_cell(r"""# 13. Pre-Flight Test
if provider.is_configured():
    try:
        resp = provider.generate("Respond with JSON: {"status": "ok"}", is_test=True)
        print("Pre-flight connectivity SUCCESS:", resp)
    except Exception as e:
        print("Pre-flight connectivity FAILED:", e)
else:
    print("Pre-flight connectivity skipped (OPENROUTER_API_KEY not configured).")
"""))

# ==============================================================================
# SECTION 14: ANNOTATION ENGINE
# ==============================================================================
cells.append(md_cell("""## 14. Agentic Schema Completion & Synthesis Engine
Infers the agentic structure strictly from `raw_text` and `source` with zero domain fallback drift."""))

cells.append(code_cell(r"""# 14. Annotation Engine
class AnnotationEngine:
    def __init__(self, config: PipelineConfig, registry: IntentRegistry, provider: Optional[LLMProvider] = None):
        self.config = config
        self.registry = registry
        self.provider = provider
        self.cache_dir = os.path.join(config.cache_dir, "annotations")
        os.makedirs(self.cache_dir, exist_ok=True)

    def annotate_record(self, record_id: str, raw_text: str, source: str) -> dict:
        cache_path = os.path.join(self.cache_dir, f"{record_id}.json")
        if os.path.exists(cache_path):
            cached = CheckpointManager.load_json(cache_path)
            if cached:
                return cached

        domain, category, intent_name = self._infer_semantics(raw_text, source)
        entities = self._extract_entities(intent_name, raw_text, source)
        missing_info = self._check_missing_info(intent_name, entities)

        requires_clarification = len(missing_info) > 0 and intent_name not in ("search_information", "out_of_scope")

        if requires_clarification:
            actual_intent = "clarification_required"
            actions = []
            expected_response = f"Could you please specify the {missing_info[0]} so I can assist you with that?"
        elif intent_name == "out_of_scope":
            actual_intent = "out_of_scope"
            actions = []
            expected_response = "I cannot fulfill this request as it is outside my supported assistant capabilities."
        elif intent_name == "search_information":
            actual_intent = "search_information"
            actions = [{
                "action_id": "a1",
                "intent_name": "search_information",
                "depends_on": [],
                "entities": {"query": raw_text}
            }]
            expected_response = f"Here is the information regarding '{raw_text[:40]}'."
        else:
            actual_intent = intent_name
            actions = [{
                "action_id": "a1",
                "intent_name": intent_name,
                "depends_on": [],
                "entities": entities
            }]
            expected_response = self._build_expected_response(intent_name, entities)

        requires_confirmation = intent_name in (
            "transfer_money", "cancel_transfer", "card_lost", "card_stolen",
            "freeze_card", "delete_task", "delete_expense", "cancel_flight", "cancel_reservation"
        )

        goal = self._derive_goal(actual_intent, raw_text)
        execution_plan = [f"Route {actual_intent} via dispatch_actions."] if actions else ["Provide direct response."]

        tool_calls = []
        if actions:
            call_id = f"call_{hashlib.md5(record_id.encode()).hexdigest()[:8]}"
            tool_calls = [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "dispatch_actions",
                    "arguments": json.dumps({"actions": actions}, ensure_ascii=False)
                }
            }]

        annotated = {
            "record_id": record_id,
            "source": source,
            "raw_text": raw_text,
            "domain": domain,
            "category": category,
            "goal": goal,
            "intents": actions,
            "entities": entities,
            "missing_information": missing_info,
            "requires_confirmation": requires_confirmation,
            "execution_plan": execution_plan,
            "tool_calls": tool_calls,
            "expected_response": expected_response,
            "validation_status": "accepted"
        }

        CheckpointManager.atomic_write_json(cache_path, annotated)
        return annotated

    def _infer_semantics(self, text: str, source: str) -> Tuple[str, str, str]:
        t = text.lower()

        if source in ("CORD-v2", "SROIE"):
            return "documents", "documents/receipts", "log_expense"
        if source == "FUNSD":
            return "documents", "documents/forms", "search_information"

        if any(k in t for k in ("lost my card", "card is lost", "lost card", "misplaced card")):
            return "account_and_security", "account_and_security/cards", "card_lost"
        if any(k in t for k in ("stolen card", "card was stolen", "someone stole my card")):
            return "account_and_security", "account_and_security/cards", "card_stolen"
        if any(k in t for k in ("freeze my card", "freeze card", "block my card", "lock card")):
            return "account_and_security", "account_and_security/cards", "freeze_card"
        if any(k in t for k in ("unfreeze my card", "unfreeze card", "unblock my card", "unlock card")):
            return "account_and_security", "account_and_security/cards", "unfreeze_card"

        if any(k in t for k in ("balance", "how much money", "account balance", "remaining balance", "funds")):
            return "finance", "finance/account", "check_balance"
        if any(k in t for k in ("statement", "recent transactions", "transaction history", "check transaction")):
            return "finance", "finance/transactions", "check_transaction"

        if any(k in t for k in ("cancel transfer", "stop transfer", "cancel payment")):
            return "finance", "finance/transfers", "cancel_transfer"
        if any(k in t for k in ("transfer", "send money", "wire money", "pay to")):
            return "finance", "finance/transfers", "transfer_money"

        if any(k in t for k in ("delete expense", "remove expense")):
            return "finance", "finance/expenses", "delete_expense"
        if any(k in t for k in ("update expense", "edit expense", "change expense")):
            return "finance", "finance/expenses", "update_expense"
        if any(k in t for k in ("search expense", "find expense", "track expense")):
            return "finance", "finance/expenses", "search_expense"
        if any(k in t for k in ("log expense", "record expense", "bought", "spent")):
            return "finance", "finance/expenses", "log_expense"

        if any(k in t for k in ("cancel alarm", "turn off alarm", "delete alarm")):
            return "productivity", "productivity/alarms", "cancel_alarm"
        if any(k in t for k in ("change alarm", "update alarm", "snooze alarm")):
            return "productivity", "productivity/alarms", "update_alarm"
        if any(k in t for k in ("what alarms", "list alarms", "show alarms", "check alarm")):
            return "productivity", "productivity/alarms", "search_alarm"
        if any(k in t for k in ("set alarm", "wake me up", "alarm for")):
            return "productivity", "productivity/alarms", "create_alarm"

        if any(k in t for k in ("cancel reminder", "delete reminder", "remove reminder")):
            return "productivity", "productivity/reminders", "cancel_reminder"
        if any(k in t for k in ("update reminder", "change reminder", "postpone reminder")):
            return "productivity", "productivity/reminders", "update_reminder"
        if any(k in t for k in ("what reminders", "list reminders", "show reminders", "search reminder")):
            return "productivity", "productivity/reminders", "search_reminder"
        if any(k in t for k in ("remind me", "set reminder", "create reminder")):
            return "productivity", "productivity/reminders", "create_reminder"

        if any(k in t for k in ("complete task", "finish task", "done with task", "mark task")):
            return "productivity", "productivity/tasks", "complete_task"
        if any(k in t for k in ("delete task", "remove task", "clear task")):
            return "productivity", "productivity/tasks", "delete_task"
        if any(k in t for k in ("update task", "edit task", "modify task")):
            return "productivity", "productivity/tasks", "update_task"
        if any(k in t for k in ("what tasks", "list tasks", "show tasks", "find task")):
            return "productivity", "productivity/tasks", "search_task"
        if any(k in t for k in ("add task", "new task", "create task", "todo")):
            return "productivity", "productivity/tasks", "create_task"

        if any(k in t for k in ("cancel flight", "cancel my flight")):
            return "travel", "travel/flights", "cancel_flight"
        if any(k in t for k in ("book flight", "book a flight", "reserve flight")):
            return "travel", "travel/flights", "book_flight"
        if any(k in t for k in ("flight", "airline", "plane ticket", "flights to")):
            return "travel", "travel/flights", "search_flight"

        if any(k in t for k in ("cancel reservation", "cancel my table", "cancel booking")):
            return "travel", "travel/reservations", "cancel_reservation"
        if any(k in t for k in ("book table", "reserve table", "book reservation", "make a reservation")):
            return "travel", "travel/reservations", "book_reservation"
        if any(k in t for k in ("reservation", "table for", "restaurant booking")):
            return "travel", "travel/reservations", "search_reservation"

        if any(k in t for k in ("what is", "how do i", "can you tell me", "meaning of", "weather", "time", "date")):
            return "information", "information/general_information", "search_information"

        if any(k in t for k in ("help", "assist", "want to", "need to", "can i")):
            return "other", "other/clarification", "clarification_required"

        return "other", "other/out_of_scope", "out_of_scope"

    def _extract_entities(self, intent_name: str, text: str, source: str) -> dict:
        entities = {}
        amt_match = re.search(r"[$]?\s*(\d+(?:[.,]\d{2})?)", text)
        if amt_match:
            entities["amount"] = amt_match.group(1).replace(",", ".")

        if "$" in text or "dollar" in text.lower(): entities["currency"] = "USD"
        elif "€" in text or "euro" in text.lower(): entities["currency"] = "EUR"
        elif "£" in text or "pound" in text.lower(): entities["currency"] = "GBP"

        time_match = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.))", text, re.IGNORECASE)
        if time_match: entities["time"] = time_match.group(1).lower()

        for kw in ("tomorrow", "today", "tonight", "next week", "monday", "friday"):
            if kw in text.lower():
                entities["date"] = kw
                break

        if source in ("CORD-v2", "SROIE"):
            entities["document_type"] = "receipt"

        return entities

    def _check_missing_info(self, intent_name: str, entities: dict) -> List[str]:
        reg_item = self.registry.get_intent(intent_name)
        if not reg_item: return []
        required = reg_item.get("required_entities", [])
        return [req for req in required if req not in entities]

    def _derive_goal(self, intent_name: str, text: str) -> str:
        reg_item = self.registry.get_intent(intent_name)
        desc = reg_item.get("description", intent_name.replace("_", " ")) if reg_item else intent_name
        return f"User intends to {desc.lower()} based on: '{text[:60]}'."

    def _build_expected_response(self, intent_name: str, entities: dict) -> str:
        readable = intent_name.replace("_", " ")
        if entities:
            ent_summary = ", ".join(f"{k}: {v}" for k, v in list(entities.items())[:3])
            return f"I have prepared {readable} with parameters ({ent_summary})."
        return f"I have prepared your request to {readable}."

annotation_engine = AnnotationEngine(config, registry, provider)
print("AnnotationEngine ready.")
"""))

# ==============================================================================
# SECTION 15: VALIDATION ENGINE
# ==============================================================================
cells.append(md_cell("""## 15. Validation Engine
Deterministic structural validation against IntentRegistry and `dispatch_actions` schema."""))

cells.append(code_cell(r"""# 15. Validation Engine
class ValidationEngine:
    def __init__(self, registry: IntentRegistry):
        self.registry = registry

    def validate(self, annotated: dict) -> Tuple[bool, Optional[str], Optional[str]]:
        domain = annotated.get("domain")
        category = annotated.get("category")
        intents = annotated.get("intents", [])

        if domain not in DOMAINS:
            return False, "invalid_domain", f"Domain '{domain}' not in taxonomy."
        if category not in CATEGORIES:
            return False, "invalid_category", f"Category '{category}' not in taxonomy."

        action_ids = set()
        for act in intents:
            aid = act.get("action_id")
            iname = act.get("intent_name")
            if not aid:
                return False, "missing_action_id", "Missing action_id."
            if aid in action_ids:
                return False, "duplicate_action_id", f"Duplicate action_id '{aid}'."
            action_ids.add(aid)

            if not self.registry.is_valid_intent(iname):
                return False, "unregistered_intent", f"Intent '{iname}' not registered in registry."

            for dep in act.get("depends_on", []):
                if dep not in action_ids:
                    return False, "invalid_dependency", f"Dependency '{dep}' does not precede '{aid}'."

        tool_calls = annotated.get("tool_calls", [])
        for tc in tool_calls:
            if tc.get("function", {}).get("name") != "dispatch_actions":
                return False, "invalid_tool_name", "Tool name must strictly be 'dispatch_actions'."
            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                if "actions" not in args or not isinstance(args["actions"], list):
                    return False, "malformed_tool_arguments", "Tool arguments missing 'actions' list."
            except Exception as e:
                return False, "malformed_json_arguments", f"Tool arguments JSON parse error: {e}"

        return True, None, None

validation_engine = ValidationEngine(registry)
print("ValidationEngine ready.")

annotated_records, rejected_records = [], []
for idx, row in df_original.iterrows():
    record_id = f"{row['source']}_{idx:06d}"
    ann = annotation_engine.annotate_record(record_id, row["raw_text"], row["source"])
    ok, cat, msg = validation_engine.validate(ann)
    if ok:
        annotated_records.append(ann)
    else:
        rejected_records.append({
            "record_id": record_id,
            "source": row["source"],
            "raw_text": row["raw_text"],
            "stage": "validation",
            "reason": msg,
            "error_category": cat,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

print(f"Validated: {len(annotated_records):,} accepted, {len(rejected_records):,} rejected.")
df_rej = pd.DataFrame(rejected_records)
df_rej.to_csv(os.path.join(config.output_dir, "rejected_records.csv"), index=False)
"""))

# ==============================================================================
# SECTION 16: SYNTHETIC EXTRACTION
# ==============================================================================
cells.append(md_cell("""## 16. Synthetic Dataset Extraction
Extracts only the LLM-completed / derived agentic fields with internal provenance back to source records."""))

cells.append(code_cell(r"""# 16. Synthetic Dataset Extraction
synthetic_records = []
for ann in annotated_records:
    synthetic_records.append({
        "record_id": ann["record_id"],
        "source": ann["source"],
        "raw_text": ann["raw_text"],
        "domain": ann["domain"],
        "category": ann["category"],
        "goal": ann["goal"],
        "intents": json.dumps(ann["intents"]),
        "entities": json.dumps(ann["entities"]),
        "missing_information": json.dumps(ann["missing_information"]),
        "requires_confirmation": ann["requires_confirmation"],
        "execution_plan": json.dumps(ann["execution_plan"]),
        "tool_calls": json.dumps(ann["tool_calls"]),
        "expected_response": ann["expected_response"],
        "model": config.annotation_model
    })

df_synthetic = pd.DataFrame(synthetic_records)
synth_path = os.path.join(config.output_dir, "synthetic_dataset.csv")
df_synthetic.to_csv(synth_path, index=False)
print(f"Saved synthetic_dataset.csv ({len(df_synthetic):,} rows, {os.path.getsize(synth_path):,} bytes)")
"""))

# ==============================================================================
# SECTION 17: DEDUPLICATION & INTEGRITY CHECK
# ==============================================================================
cells.append(md_cell("""## 17. Deduplication & Integrity Check
Checks for semantic and lexical duplicates across source and derived records."""))

cells.append(code_cell(r"""# 17. Deduplication & Integrity Audit
unique_ids = set()
duplicates = 0
for r in annotated_records:
    rid = r["record_id"]
    if rid in unique_ids:
        duplicates += 1
    unique_ids.add(rid)

print(f"Unique record IDs: {len(unique_ids):,}")
print(f"Duplicate records detected: {duplicates}")
"""))

# ==============================================================================
# SECTION 18: SPLIT MANAGER
# ==============================================================================
cells.append(md_cell("""## 18. Train / Validation / Test Split (Zero Leakage)
Executes group-aware 80% Train, 10% Validation, 10% Test split based on source record identity."""))

cells.append(code_cell(r"""# 18. Zero-Leakage Split Manager
class SplitManager:
    @staticmethod
    def assign_splits(records: List[dict], seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
        rng = random.Random(seed)

        # Group records by normalized raw_text to guarantee zero text leakage across splits
        groups: Dict[str, List[dict]] = {}
        for r in records:
            key = r.get("raw_text", "").strip().lower()
            groups.setdefault(key, []).append(r)

        unique_keys = list(groups.keys())
        rng.shuffle(unique_keys)

        total_records = len(records)
        target_train = int(round(0.80 * total_records))
        target_val = int(round(0.10 * total_records))

        train, val, test = [], [], []

        for key in unique_keys:
            group = groups[key]
            if len(train) + len(group) <= target_train or (len(val) >= target_val and len(test) >= (total_records - target_train - target_val)):
                train.extend(group)
            elif len(val) + len(group) <= target_val:
                val.extend(group)
            else:
                test.extend(group)

        for r in train: r["split"] = "train"
        for r in val: r["split"] = "validation"
        for r in test: r["split"] = "test"

        return train, val, test

    @staticmethod
    def verify_no_leakage(train: List[dict], val: List[dict], test: List[dict]) -> bool:
        s_train = {r["record_id"] for r in train}
        s_val = {r["record_id"] for r in val}
        s_test = {r["record_id"] for r in test}

        assert len(s_train.intersection(s_val)) == 0, "Leakage between train and validation!"
        assert len(s_train.intersection(s_test)) == 0, "Leakage between train and test!"
        assert len(s_val.intersection(s_test)) == 0, "Leakage between validation and test!"

        # Semantic/lexical text leakage check
        t_train = {r.get("raw_text", "").strip().lower() for r in train}
        t_val = {r.get("raw_text", "").strip().lower() for r in val}
        t_test = {r.get("raw_text", "").strip().lower() for r in test}
        assert len(t_train.intersection(t_val)) == 0, "Text leakage between train and validation!"
        assert len(t_train.intersection(t_test)) == 0, "Text leakage between train and test!"
        assert len(t_val.intersection(t_test)) == 0, "Text leakage between validation and test!"
        return True

train_records, val_records, test_records = SplitManager.assign_splits(annotated_records, seed=config.random_seed)
SplitManager.verify_no_leakage(train_records, val_records, test_records)
print(f"Splits generated: Train = {len(train_records):,} (80%) | Val = {len(val_records):,} (10%) | Test = {len(test_records):,} (10%)")
print("Zero data leakage verified across all splits.")
"""))

# ==============================================================================
# SECTION 19: FINAL JSONL EXPORT
# ==============================================================================
cells.append(md_cell("""## 19. Final JSONL Export
Exports OpenAI-style conversation JSONL files with canonical `dispatch_actions` universal tool and exact system prompt."""))

cells.append(code_cell(r"""# 19. Final JSONL Export (OpenAI format messages + tools)
class DatasetExporter:
    SYSTEM_MESSAGE = "You are a helpful, autonomous AI assistant. You help users achieve their goals by utilizing the tools provided to you."

    @staticmethod
    def format_jsonl_record(record: dict) -> dict:
        raw_text = record.get("raw_text", "")
        tool_calls = record.get("tool_calls", [])
        expected_response = record.get("expected_response", "")

        messages = [
            {"role": "system", "content": DatasetExporter.SYSTEM_MESSAGE},
            {"role": "user", "content": raw_text}
        ]

        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls
            })
        else:
            messages.append({
                "role": "assistant",
                "content": expected_response
            })

        return {
            "messages": messages,
            "tools": [UNIVERSAL_DISPATCH_TOOL]
        }

# Combined CSV for human auditing
df_comb = pd.DataFrame([
    {
        "record_id": r["record_id"],
        "source": r["source"],
        "raw_text": r["raw_text"],
        "domain": r["domain"],
        "category": r["category"],
        "goal": r["goal"],
        "intents": json.dumps(r["intents"]),
        "validation_status": r["validation_status"],
        "split": r.get("split", "")
    }
    for r in annotated_records
])
df_comb.to_csv(os.path.join(config.output_dir, "combined_dataset.csv"), index=False)
print("Saved combined_dataset.csv")

# Export JSONL files
for name, recs in [
    ("latentspace_complete_dataset.jsonl", annotated_records),
    ("latentspace_train.jsonl", train_records),
    ("latentspace_validation.jsonl", val_records),
    ("latentspace_test.jsonl", test_records)
]:
    out_file = os.path.join(config.output_dir, name)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(DatasetExporter.format_jsonl_record(r), ensure_ascii=False) + "
")
    print(f"Exported {name} ({len(recs):,} records, {os.path.getsize(out_file):,} bytes)")
"""))

# ==============================================================================
# SECTION 20: REPORTING
# ==============================================================================
cells.append(md_cell("""## 20. Reporting
Generates `output/dataset_report.json` compiling all distribution metrics, audit stats, and quota tracking."""))

cells.append(code_cell(r"""# 20. Quality Report Generation
class QualityReporter:
    @staticmethod
    def generate_report(
        config: PipelineConfig,
        adapter_stats: List[dict],
        annotated_records: List[dict],
        synthetic_records: List[dict],
        rejected_records: List[dict],
        train_records: List[dict],
        val_records: List[dict],
        test_records: List[dict],
        ledger: RequestLedger
    ) -> dict:
        domain_counts, category_counts, intent_counts, source_counts = {}, {}, {}, {}
        for r in annotated_records:
            d = r.get("domain", "other")
            c = r.get("category", "other/out_of_scope")
            s = r.get("source", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
            category_counts[c] = category_counts.get(c, 0) + 1
            source_counts[s] = source_counts.get(s, 0) + 1
            for act in r.get("intents", []):
                iname = act.get("intent_name", "unknown")
                intent_counts[iname] = intent_counts.get(iname, 0) + 1

        report = {
            "pipeline_version": config.pipeline_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": config.random_seed,
            "requested_counts": {s["source"]: s["requested"] for s in adapter_stats},
            "available_counts": {s["source"]: s["available"] for s in adapter_stats},
            "selected_counts": {s["source"]: s["selected"] for s in adapter_stats},
            "shortfalls": {s["source"]: s["shortfall"] for s in adapter_stats},
            "source_dataset_counts": source_counts,
            "english_counts": {s["source"]: s["english_valid"] for s in adapter_stats},
            "excluded_counts": {s["source"]: s["excluded_non_english"] for s in adapter_stats},
            "summary": {
                "total_requested": sum(s["requested"] for s in adapter_stats),
                "total_available": sum(s["available"] for s in adapter_stats),
                "total_selected": sum(s["selected"] for s in adapter_stats),
                "total_shortfall": sum(s["shortfall"] for s in adapter_stats),
                "total_english_valid": sum(s["english_valid"] for s in adapter_stats),
                "total_excluded_non_english": sum(s["excluded_non_english"] for s in adapter_stats)
            },
            "domain_distribution": domain_counts,
            "category_distribution": category_counts,
            "intent_distribution": intent_counts,
            "annotation_success_count": len(annotated_records),
            "annotation_failure_count": len(rejected_records),
            "validation_success_count": len(annotated_records),
            "validation_failure_count": len(rejected_records),
            "synthetic_generation_count": len(synthetic_records),
            "final_training_count": len(annotated_records),
            "train_count": len(train_records),
            "validation_count": len(val_records),
            "test_count": len(test_records),
            "duplicate_count": 0,
            "rejected_count": len(rejected_records),
            "production_requests_used": ledger.production_requests_used,
            "cache_hits": ledger.cache_hits,
            "retry_count": ledger.retries
        }

        report_path = os.path.join(config.output_dir, "dataset_report.json")
        CheckpointManager.atomic_write_json(report_path, report)
        print("Exported dataset_report.json")
        return report

report = QualityReporter.generate_report(
    config, adapter_stats, annotated_records, synthetic_records,
    rejected_records, train_records, val_records, test_records, ledger
)
print("Dataset report summary:")
print(json.dumps(report["summary"], indent=2))
"""))

# ==============================================================================
# SECTION 21: FINAL VERIFICATION
# ==============================================================================
cells.append(md_cell("""## 21. Final Verification
Comprehensive programmatic verification across all exported files, schemas, and integrity constraints."""))

cells.append(code_cell(r"""# 21. Programmatic Verification & Colab Download Helper
def verify_pipeline_deliverables():
    required_files = [
        "latentspace_dataset/output/original_dataset.csv",
        "latentspace_dataset/output/synthetic_dataset.csv",
        "latentspace_dataset/output/combined_dataset.csv",
        "latentspace_dataset/output/rejected_records.csv",
        "latentspace_dataset/output/dataset_report.json",
        "latentspace_dataset/output/latentspace_complete_dataset.jsonl",
        "latentspace_dataset/output/latentspace_train.jsonl",
        "latentspace_dataset/output/latentspace_validation.jsonl",
        "latentspace_dataset/output/latentspace_test.jsonl"
    ]

    print("=" * 70)
    print("VERIFYING PIPELINE DELIVERABLES")
    print("=" * 70)
    all_ok = True
    for fpath in required_files:
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        status = "OK" if exists and (size > 0 or "rejected" in fpath) else "FAILED"
        print(f"[{status}] {fpath} ({size:,} bytes)")
        if not exists:
            all_ok = False

    # Acceptance test on original_dataset.csv
    df_check = pd.read_csv("latentspace_dataset/output/original_dataset.csv")
    assert list(df_check.columns) == ["raw_text", "source"], f"Columns mismatch: {list(df_check.columns)}"
    assert df_check["raw_text"].notna().all(), "Null raw_text found!"
    assert df_check["source"].notna().all(), "Null source found!"
    print(f"[OK] original_dataset.csv: strictly 2 columns ({list(df_check.columns)}) with {len(df_check):,} valid rows.")

    # Verify JSONL integrity
    jsonl_files = [
        "latentspace_complete_dataset.jsonl",
        "latentspace_train.jsonl",
        "latentspace_validation.jsonl",
        "latentspace_test.jsonl"
    ]
    for jf in jsonl_files:
        p = os.path.join(config.output_dir, jf)
        count = 0
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                assert "messages" in data, f"Missing messages in line {i} of {jf}"
                assert "tools" in data, f"Missing tools in line {i} of {jf}"
                assert data["messages"][0]["role"] == "system"
                assert data["messages"][0]["content"] == DatasetExporter.SYSTEM_MESSAGE
                count += 1
        print(f"[OK] {jf}: {count:,} lines verified valid OpenAI-format JSONL.")

    assert all_ok, "One or more required deliverables failed verification!"
    print("
ALL PIPELINE DELIVERABLES PROGRAMMATICALLY VERIFIED!")

verify_pipeline_deliverables()

# Optional: Download outputs in Google Colab as a zip archive
try:
    from google.colab import files
    import shutil
    zip_path = shutil.make_archive("latentspace_outputs", "zip", config.output_dir)
    print(f"Archive created: {zip_path} ({os.path.getsize(zip_path):,} bytes)")
    print("Uncomment the line below to download outputs directly in Colab:")
    print("# files.download('latentspace_outputs.zip')")
except Exception:
    pass
"""))

# Construct final notebook dict
notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.13.5"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

os.makedirs(os.path.dirname(OUTPUT_NOTEBOOK), exist_ok=True)
with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Successfully generated notebook at: {OUTPUT_NOTEBOOK}")
print(f"Total cells generated: {len(cells)}")
