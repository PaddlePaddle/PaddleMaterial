# ppmatAgent

`ppmatAgent` collects agent workflows for PaddleMaterials.

## Modules

- `ppmatAgent.hea_crewai_agent`: high-entropy alloy optimization workflow migrated from the HEA CrewAI prototype. The original architecture is preserved.
- `ppmatAgent.knowmat`: materials literature cleaning and extraction workflow migrated from KnowMat. The OCR path is PaddleOCR-first: `auto` uses PaddleOCR API when `PADDLEOCR_API_TOKEN` is configured, otherwise local PaddleOCR-VL. OpenAI-compatible vision OCR is available as an explicit backend. MinerU remains only as a compatibility backend.

## Optional Dependencies

The agent stack is intentionally not added to the main PaddleMaterials
`requirements.txt`, because it pulls large LLM/OCR frameworks. Install optional
dependencies only when using this package:

```bash
pip install -r ppmatAgent/requirements-optional.txt
```

## KnowMat OCR Usage

Default backend selection:

```bash
python -m ppmatAgent.knowmat --input-folder data/raw --ocr-backend auto --ocr-only
```

PaddleOCR API:

```bash
export PADDLEOCR_API_TOKEN=...
python -m ppmatAgent.knowmat --input-folder data/raw --ocr-backend paddleocr-api --ocr-only
```

OpenAI-compatible vision OCR:

```bash
export KNOWMAT_OCR_API_KEY=...
export KNOWMAT_OCR_BASE_URL=https://aistudio.baidu.com/llm/lmapi/v3
export KNOWMAT_OCR_MODEL=ernie-5.0-thinking-preview
python -m ppmatAgent.knowmat --input-folder data/raw --ocr-backend openai --ocr-only
```

Local PaddleOCR-VL:

```bash
python -m ppmatAgent.knowmat --input-folder data/raw --ocr-backend paddleocr-local --ocr-only
```

MinerU compatibility backend:

```bash
export MINERU_API_KEY=...
python -m ppmatAgent.knowmat --input-folder data/raw --ocr-backend mineru --ocr-only
```

## HEA CrewAI Usage

```bash
python -m ppmatAgent.hea_crewai_agent.run_crewai \
  --elements Co,Cr,Fe,Ni,V \
  --requirement "design a high-strength high-entropy alloy"
```

HEA CrewAI uses either `llmone` or OpenAI-compatible APIs:

```bash
python -m ppmatAgent.hea_crewai_agent.run_crewai \
  --llm-api openai \
  --base-url https://aistudio.baidu.com/llm/lmapi/v3 \
  --model ernie-5.0-thinking-preview
```
