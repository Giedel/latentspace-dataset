# Latent Dataset Annotation + Synthetic Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a complete, reproducible, self-contained Jupyter notebook and supporting configuration/prompt files implementing the Latent Dataset Annotation and Synthetic Data Pipeline according to the Master Implementation Specification.

**Architecture:** A 100% self-contained object-oriented pipeline residing in `latentspace_dataset/notebook/latentspace_dataset_builder.ipynb`, supported by `config/intent_registry.json`, prompt templates in `prompts/`, with an OpenRouter free-tier provider adapter (`qwen/qwen3.8-27b:free`), shared 50-request daily budget, audit ledger, atomic checkpointing, and dry-run preflight validation.

**Tech Stack:** Python 3.10+, Jupyter Notebook (`nbformat`), Pandas, Hugging Face `datasets`, Requests, Standard Library (`dataclasses`, `json`, `hashlib`, `uuid`, `time`, `os`, `re`).

**Spec:** `docs/superpowers/specs/2026-09-23-latent-dataset-pipeline-design.md`

## Global Constraints
- Pipeline version: `1.0.0`
- Universal tool: `dispatch_actions` only; no per-intent tool definitions.
- Default models: `qwen/qwen3.8-27b:free` (configurable, free-only routing, `allow_fallbacks = False`).
- Budget: Shared 50 requests/day production budget across annotation, synthesis, validation, and retries.
- API Key: Exclusively read from `OPENROUTER_API_KEY` env var; never logged, never printed, never stored.
- Self-contained: All Python classes defined directly in notebook cells.
- Zero duplication: Never duplicate records to artificially satisfy target counts; report shortfalls.
- Grouped splits: 80/10/10 split at `record_group_id` level; zero train/test leakage.
- Dry-run mode: Preflight validation tests environment, dataset availability, normalization, schema, ledger without consuming production API budget.

---

### Task 1: Scaffolding Directory Structure, Config & Prompts
- Create `latentspace_dataset/` directory tree: `config/`, `prompts/`, `output/`, `cache/{normalized,annotations,synthetic,validation,checkpoints}`.
- Create `latentspace_dataset/config/intent_registry.json`.
- Create `latentspace_dataset/prompts/annotation_prompt.txt`.
- Create `latentspace_dataset/prompts/synthetic_generation_prompt.txt`.
- Create `latentspace_dataset/prompts/validation_prompt.txt`.

### Task 2: Build the Self-Contained Jupyter Notebook (`latentspace_dataset_builder.ipynb`)
- Assemble all modular sections into 23 cleanly structured notebook cells:
  1. Title, Overview & Universal Tool Architecture Markdown
  2. Imports & Version Setup
  3. Pipeline Configuration (`PipelineConfig`)
  4. API Key Verification & Security Guard
  5. Quota Manager & Audit Request Ledger (`QuotaManager`)
  6. Universal Tool Contract & Intent Registry (`IntentRegistry`)
  7. Provider Abstraction & OpenRouter Adapter (`LLMProvider`, `OpenRouterProvider`)
  8. Connectivity Pre-flight Test Cell (`is_test=True`)
  9. Dataset Adapter Base Class (`DatasetAdapter`)
  10. Concrete Text Adapters (`Banking77Adapter`, `Clinc150Adapter`, `Hwu64Adapter`)
  11. Concrete Audio Transcript Adapters (`Minds14USAdapter`, `Minds14ExtAdapter`)
  12. Concrete OCR Adapters (`CordV2Adapter`, `SroieAdapter`, `FunsdAdapter`)
  13. Unified Dataset Ingestion & Normalization Runner (`DatasetNormalizer`)
  14. Annotation Engine & Multi-Intent Parser (`AnnotationEngine`)
  15. Synthetic Generator with Controlled Grounding (`SyntheticGenerator`)
  16. Two-Tier Validation Engine (`ValidationEngine`)
  17. Deduplication Engine (`DeduplicationEngine`)
  18. Split Manager with Zero-Leakage Grouping (`SplitManager`)
  19. Dataset Exporter (CSV + JSONL + Chat Tool-Call Formatter) (`DatasetExporter`)
  20. Comprehensive Quality Reporter (`QualityReporter`)
  21. Pre-flight Dry-Run Execution Cell (Tests full baseline without production API calls)
  22. Batch Annotation & Synthesis Execution Cell (Honors 50 req/day limit with checkpointing)
  23. Final Validation, Integrity Audit & Export Cell

### Task 3: Execute Pre-flight Dry-Run in Colab & Verify Baseline
- Transfer/execute the notebook cells in the connected Google Colab environment via `colab-mcp`.
- Verify dataset downloads and raw counts.
- Verify normalization, language validation, and shortfall accounting.
- Verify `IntentRegistry` loading and universal `dispatch_actions` validation.
- Verify connectivity test with test-mode request accounting.
- Verify atomic checkpointing and ledger persistence.
- Verify export outputs (`original_dataset.csv`, `combined_dataset.csv`, JSONLs, `dataset_report.json`).

### Task 4: Local Repository Verification & Final Commit
- Verify all generated local artifacts in `latentspace_dataset/`.
- Ensure git repository contains the complete reproducible codebase.
- Provide user walkthrough and clear operational instructions.
