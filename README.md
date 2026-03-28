---
title: DataCleanEnv
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# DataCleanEnv: Data Privacy & Compliance Steward

## Overview
**DataCleanEnv** is a high-fidelity RL environment designed for the OpenEnv Hackathon. It simulates the high-stakes work of Data Engineers and Compliance Officers who must manage, audit, and clean sensitive production databases. Agents interact with a live **SQLite** engine to enforce data privacy (GDPR/PII) and ensure schema integrity.

## Motivation
Modern AI agents must be capable of safely manipulating production data. This environment moves beyond standard "Read-Only" SQL tasks, challenging agents to **Modify** and **Sanitize** data while receiving dense reward signals based on their precision and compliance adherence.

## Action Space
Agents execute actions via JSON payloads. Supported actions include:
- `RUN_SQL_UPDATE`: Execute raw SQL queries (e.g., `UPDATE users SET...`).
- `APPLY_REGEX_MASK`: A specialized tool for redacting PII using SQLite's `REGEXP_REPLACE`.
  - *Parameters*: `column`, `pattern`, `replacement`.
- `DROP_COLUMN`: Remove restricted columns (e.g., `ssn`) to comply with data retention policies.
- `SUBMIT_FINAL`: Terminate the episode and trigger the final programmatic grader.

## Observation Space
Every step provides a rich context for reasoning:
- `schema_info`: A dictionary containing the **DDL statements** for every table.
- `sample_data`: A real-time preview of the first **5 rows** of each table.
- `last_execution_status`: Detailed feedback (Success or Traceback) from the SQLite engine.

## Tasks & Difficulty

| ID | Task Name | Difficulty | Objective |
| :--- | :--- | :--- | :--- |
| `easy_standardization` | Date Standardization | Easy | Convert varied date strings to ISO 8601 (`YYYY-MM-DD`). |
| `medium_pii_redaction` | PII Redaction | Medium | Mask Credit Card numbers found in unstructured message logs. |
| `hard_entity_resolution` | Entity Resolution | Hard | Merge disparate sales tables and deduplicate based on email. |
| `expert_pii_audit` | Compliance Audit | Expert | **Multi-table**: Redact emails in logs AND drop the `ssn` column. |

## Reward Shaping
- **Syntax Reward (+0.05)**: Awarded for any action that executes without a database error.
- **Progress Reward (+0.5 * Δ)**: Awarded for any improvement in the underlying grading metric (Dense Signal).
- **Safety Penalty (-0.5)**: Immediate episode termination for destructive actions (e.g., unauthorized `DROP TABLE`).

## Baseline Performance (GPT-4o-mini)
The following scores were achieved using the included `inference.py` script:
- **Easy**: 1.0 (100% precision)
- **Medium**: 1.0 (100% redaction)
- **Hard**: 0.8+ (depending on merge strategy)
- **Expert**: 1.0 (perfect compliance audit)

## API Endpoints
- **Root (`/`)**: Health check and list of endpoints.
- **`/tasks`**: Returns task metadata and the Action JSON schema.
- **`/grader`**: Returns the `final_score` (0.0-1.0) for the current episode.
- **`/baseline`**: Triggers the `inference.py` script and returns scores for all tasks.
- **`/reset`, `/step`, `/state`**: Standard OpenEnv API.

## Setup & Local Testing
```bash
# 1. Install dependencies
pip install -e .

# 2. Run unit tests
pytest tests/test_dataclean.py

# 3. Start local server
uv run server

# 4. Run inference baseline (requires OPENAI_API_KEY or HF_TOKEN)
python inference.py
```

**Submission Space**: [https://huggingface.co/spaces/grey8magic/DataCleanOpenEnv](https://huggingface.co/spaces/grey8magic/DataCleanOpenEnv)
