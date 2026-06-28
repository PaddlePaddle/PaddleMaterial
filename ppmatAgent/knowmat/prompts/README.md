# Prompt Templates Guide

This directory stores editable prompt templates used by KnowMat nodes.

## Files

- `extraction_system_template.txt`: extraction agent system prompt template.
- `extraction_user_template.txt`: extraction user prompt template.
- `subfield_detection.yaml`: sub-field detection prompt template.
- `evaluation.yaml`: evaluation prompt templates.
- `validator.yaml`: validation prompt templates.
- `flagging.yaml`: final quality flagging prompt templates.

## Conventions

- Keep templates as plain text/YAML with UTF-8 encoding.
- Use explicit placeholders like `{paper_text}` or `<<RUN_ID>>`.
- Avoid embedding runtime-only values in template files.
- If adding a new YAML template, update loader call sites to validate required keys.

## Safe Editing Workflow

1. Edit the template file.
2. Run lint/syntax checks.
3. Run a small extraction/evaluation sample to confirm no regressions.

