# MASTER IMPLEMENTATION SPECIFICATION
## Latent Dataset Annotation + Synthetic Data Pipeline

**Date:** 2026-09-23  
**Target Assistant:** Latent Semantic Middleware / Offline Agentic Assistant  
**Pipeline Version:** 1.0.0  
**Status:** Approved & Locked  

---

### 1. Architectural Overview & System Contract

The goal of this pipeline is to ingest, normalize, annotate, synthesize, validate, and export a dataset for fine-tuning an offline semantic middleware assistant.
The assistant's primary responsibility is:
$$\text{Natural Language Input} \longrightarrow \text{Semantic Understanding} \longrightarrow \text{Intent Detection} \longrightarrow \text{Entity Extraction} \longrightarrow \text{Action Decomposition} \longrightarrow \text{dispatch\_actions}$$

The assistant never executes backend application logic directly. It emits structured actions via a single universal tool.

#### Canonical Universal Tool Contract
The pipeline enforces exactly **one universal function tool**: `dispatch_actions`. No individual per-intent tool schemas may ever be defined.
```json
{
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
                "description": "Unique identifier for this action within the request (e.g. 'a1', 'a2')."
              },
              "intent_name": {
                "type": "string",
                "description": "The classified intent matching the Intent Registry."
              },
              "depends_on": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional array of prior action_ids that this action depends on."
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
```

---

### 2. Directory Structure & Self-Contained Notebook Rule

The pipeline is completely self-contained in:
`latentspace_dataset/notebook/latentspace_dataset_builder.ipynb`

All classes (`PipelineConfig`, `IntentRegistry`, `QuotaManager`, `LLMProvider`, `OpenRouterProvider`, `DatasetAdapter`, `Banking77Adapter`, `Clinc150Adapter`, `Hwu64Adapter`, `Minds14Adapter`, `CordV2Adapter`, `SroieAdapter`, `FunsdAdapter`, `AnnotationEngine`, `SyntheticGenerator`, `ValidationEngine`, `DeduplicationEngine`, `SplitManager`, `DatasetExporter`, `QualityReporter`) are defined directly within clean notebook cells. Zero external local repository imports are permitted.

#### File Tree Architecture
```
latentspace_dataset/
├── notebook/
│   └── latentspace_dataset_builder.ipynb    # Completely self-contained notebook
├── config/
│   └── intent_registry.json                 # Extensible Intent Registry
├── prompts/
│   ├── annotation_prompt.txt                # System & batch annotation prompt
│   ├── synthetic_generation_prompt.txt       # Linguistic variation & controlled scenario prompt
│   └── validation_prompt.txt                # Tier-2 semantic validation filter
├── output/
│   ├── original_dataset.csv                 # Normalized original baseline records
│   ├── synthetic_dataset.csv                # Accepted synthetic records
│   ├── combined_dataset.csv                 # Original + synthetic combined
│   ├── rejected_records.csv                 # Rejected/flagged records with reasons
│   ├── dataset_report.json                  # Comprehensive audit & metrics report
│   ├── latentspace_complete_dataset.jsonl   # Complete fine-tuning dataset
│   ├── latentspace_train.jsonl              # 80% train split (grouped by record_group_id)
│   ├── latentspace_validation.jsonl         # 10% validation split
│   └── latentspace_test.jsonl               # 10% test split
└── cache/
    ├── normalized/                          # Cached normalized parquet datasets
    ├── annotations/                         # Record-level annotation cache
    ├── synthetic/                           # Generated synthetic records cache
    ├── validation/                          # Validation decisions cache
    ├── checkpoints/                         # Atomic resumable checkpoints
    └── request_ledger.jsonl                 # Immutable audit trail of every HTTP attempt
```

---

### 3. Source Datasets & Expected Counts

The requested baseline dataset consists of:
- **Text**:
  - `BANKING77`: Target 7,000 records (from PolyAI-LDN repository).
  - `CLINC150`: Target 8,000 records (`DeepPavlov/clinc150` train split).
  - `HWU64`: Target 5,000 records (`DeepPavlov/hwu64` train split).
- **Audio Transcript**:
  - `MINDS-14 US`: Target 563 records (`PolyAI/minds14`, config `en-US`).
  - `MINDS-14 EXT`: Target 1,246 records (`PolyAI/minds14`, configs `en-AU` [654] + `en-GB` [592]).
- **OCR Documents**:
  - `CORD-v2`: Target 800 records (`naver-clova-ix/cord-v2` train split).
  - `SROIE`: Target 626 records (`rth/sroie-2019-v2` train split).
  - `FUNSD`: Target 149 records (`nielsr/funsd` train split).
- **Total Baseline Expected**: 23,384 records.

#### Non-Duplication & Shortfall Policy
The notebook inspects each source and records:
- `raw_count`, `valid_count`, `english_count`, `selected_count`, `requested_count`, `shortfall`.
Records are **never duplicated** to artificially meet targets.

---

### 4. Normalized Data Schema & Modality Separation

Every normalized sample conforms to this schema:
```json
{
  "record_id": "BANKING77_000001",
  "source_record_id": "0",
  "record_group_id": "BANKING77_000001",
  "source_dataset": "BANKING77",
  "source_split": "train",
  "source_intent": "card_arrival",
  "intent_name": null,
  "mapping_status": "pending",
  "modality": "text",
  "original_text": "I ordered my card two weeks ago...",
  "document_text": null,
  "normalized_text": "I ordered my card two weeks ago...",
  "source_metadata": {},
  "language": "en",
  "is_original": true,
  "annotation_status": "pending",
  "entities": {},
  "action_count": 0,
  "scenario_type": null,
  "annotation_confidence": null
}
```

#### Modality Invariants
1. **Text & Audio Transcripts**:
   - `original_text`: Verbatim source text (untouched).
   - `document_text`: `null`.
   - `normalized_text`: `original_text`.
2. **OCR Documents**:
   - `original_text`: `null`.
   - `document_text`: Verbatim extracted document text/tokens from OCR.
   - `normalized_text`: Deterministic textual representation grounded strictly in source values.
3. **Identity**:
   - `source_record_id`: Native upstream row ID.
   - `record_id`: Deterministically derived as `{source_dataset}_{source_record_id}`.
   - `record_group_id`: Identical to `record_id` for original baseline records. All synthetic derivatives inherit the parent's `record_group_id`.
4. **Mapping Statuses**:
   `pending`, `direct`, `mapped`, `informational`, `out_of_scope`, `clarification_required`, `rejected`.

---

### 5. Intent Registry Architecture

The Intent Registry is the strict source of truth for valid `intent_name` values. It defines:
- `intent_name`: String identifier.
- `domain`: `financial`, `productivity`, `booking`, `information`, `control`.
- `description`: Clear semantic definition.
- `required_entities`: List of mandatory entity names.
- `optional_entities`: List of optional entity names with types.
- `execution_type`: `read`, `write`, `update`, `delete`, `control`.

Core intents include:
- Financial: `log_expense`, `delete_expense`, `search_expense`, `update_expense`, `transfer_money`, `check_balance`, `check_transaction`, `cancel_transfer`, `card_lost`, `card_stolen`, `freeze_card`, `unfreeze_card`.
- Productivity: `create_task`, `complete_task`, `delete_task`, `update_task`, `search_task`, `create_reminder`, `cancel_reminder`, `update_reminder`, `search_reminder`, `create_alarm`, `cancel_alarm`, `update_alarm`.
- Booking: `book_reservation`, `cancel_reservation`, `search_reservation`, `search_flight`, `book_flight`, `cancel_flight`.
- Meta/Control: `search_information`, `out_of_scope`, `clarification_required`.

---

### 6. Provider Abstraction, Quota Management & Checkpointing

#### Configurable Provider Contract
- Parameterized configuration:
  - `annotation_model = "qwen/qwen3.8-27b:free"`
  - `synthetic_model = "qwen/qwen3.8-27b:free"`
  - `validation_model = "qwen/qwen3.8-27b:free"`
  - `connectivity_test_model = "qwen/qwen3.8-27b:free"`
  - `allow_fallbacks = false` (strict free-only routing on OpenRouter)
  - `daily_request_limit = 50`
  - `stop_before_daily_limit = true`
  - `pipeline_version = "1.0.0"`
  - `random_seed = 42`

#### Quota Accounting & Request Ledger
- **Shared Budget**: Annotation, synthesis, validation, and retries share the single 50-request daily budget.
- **Accounting Rules**:
  - Initial HTTP attempt: 1 unit.
  - HTTP retry: +1 unit.
  - Successful HTTP response: 1 unit.
  - Failed HTTP response (network reached): 1 unit.
  - Cache hit: 0 units.
  - Pre-flight local validation failure: 0 units.
- **Ledger Fields**:
  - `timestamp`, `quota_date` (UTC `YYYY-MM-DD`), `attempt_id` (UUID), `request_hash`, `request_attempt`, `retry_number`, `stage`, `is_test`, `model`, `records_attempted`, `records_successful`, `http_status`, `error_category`, `latency_ms`.
  - **Zero secrets**: No API keys or authorization headers are ever logged.
- **Test Mode Separation**:
  - Pre-flight connectivity testing uses `is_test: true`.
  - Does not consume production budget, does not advance production checkpoints, and does not contaminate production caches.
- **Error Handling**:
  - Auth/Perm errors (401, 403, 404): Halt immediately.
  - Transient rate limits (HTTP 429 with retry-after): Exponential backoff up to `max_retries`.
  - Provider quota exhaustion / local daily limit reached: Atomic checkpoint and clean halt.
  - Malformed responses: Single-record retry fallback or rejection to `rejected_records.csv`. No aggressive/hallucinatory JSON repair.

#### Cryptographic Identity & Atomic Checkpoint
- `request_hash = SHA256(v + stage + model + sorted_record_ids + prompt_hash + schema_hash)`
- Checkpoints saved via temporary file + atomic `os.replace` to guarantee integrity during interruptions.
- Delivery Guarantee: Best-effort exactly-once processing using deterministic request identities, persistent caching, and atomic checkpoints.

---

### 7. Annotation, Synthesis, and Validation Engines

#### Annotation Engine
- Maps normalized inputs to valid `IntentRegistry` intents.
- Assigns scenario types: `SINGLE`, `MULTIPLE_SAME_INTENT`, `MULTIPLE_DIFFERENT_INTENT`, `PARALLEL`, `SEQUENTIAL`, `CONDITIONAL`, `MIXED`, `CLARIFICATION_REQUIRED`, `OUT_OF_SCOPE`.
- Extracts entities strictly grounded in source text.
- Formats structured tool calls into `dispatch_actions`.

#### Synthetic Generator (Controlled Grounding)
- Uses original record as semantic seed.
- **No Semantic Drift**: Multi-intent, sequential, or conditional scenarios are generated ONLY when supported by source semantics, grounded in document/source information, or specified by controlled scenario configuration. Never attaches unrelated intents (e.g. transfers/reminders) to unrelated requests.
- **Strict English-Only**: Generates 100% English queries; rejects non-English or mixed language tokens.
- **Linkage**: Retains `source_record_id`, `record_group_id`, `synthetic_group_id`, `generation_model`, `is_original: false`.

#### Validation Engine (Two-Tier Quality Gate)
- **Tier 1 (Deterministic)**: Schema check, `IntentRegistry` membership, required/optional entity validation, dependency consistency (`depends_on`), language code verification, lexical distinctness threshold.
- **Tier 2 (Semantic LLM)**: Intent preservation check, anti-hallucination evaluation, action alignment, English purity.
- Records failing validation are routed to `output/rejected_records.csv` with explicit `rejection_reason`.

---

### 8. Leakage Prevention, Splitting & Export

- **Grouping**: Dataset split is performed at the `record_group_id` level (80% train, 10% validation, 10% test). All synthetic variants remain in the exact same split as their source baseline record.
- **Outputs**:
  - `original_dataset.csv`
  - `synthetic_dataset.csv`
  - `combined_dataset.csv` (contains both, with `final_text` column)
  - `output/rejected_records.csv`
  - `output/dataset_report.json`
  - `output/latentspace_complete_dataset.jsonl`
  - `output/latentspace_train.jsonl`
  - `output/latentspace_validation.jsonl`
  - `output/latentspace_test.jsonl`

---

### 9. Pre-flight & Dry-Run Verification Plan

Before running production generation:
1. Validate directory creation (`latentspace_dataset/{config,prompts,output,cache}`).
2. Verify API key presence in environment (`OPENROUTER_API_KEY`).
3. Run test-mode connectivity ping to configured model with `is_test=True`.
4. Verify source datasets loading, counts, and normalization in dry-run mode without incurring production API calls.
5. Verify schema validation and audit reporting.
