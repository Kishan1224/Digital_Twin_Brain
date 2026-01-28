# AI Personal Digital Twin (Local-Only Demo)

This is a **privacy-first Personal Digital Twin** demo you can run fully on your machine:

- Streamlit UI with tabs:
  - Chat with Twin
  - Memory Timeline
  - Decision Predictor
  - Daily Suggestions
  - Privacy Dashboard
- Persistent memory using **Chroma** (stored locally on disk)
- Embeddings via **Sentence Transformers**
- Modes:
  - **MIRROR**: respond like "you" based on retrieved memories
  - **ADVISOR**: give improved options aligned with stated goals/values
  - **PREDICTOR**: forecast likely next action + alternatives + recommendation

## Quickstart (Windows PowerShell)

```powershell
cd "C:\Users\puvvu\digital-twin"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Local LLM (Optional, Recommended)

This demo can call a local Ollama server if you have it installed and running.

1. Install Ollama and start it
2. Pull a model, e.g.:

```powershell
ollama pull llama3.1
```

3. In the app sidebar, enable **Local LLM via Ollama** and set model name (default: `llama3.1`).

If Ollama is not available, the app falls back to a lightweight heuristic response generator (still demonstrates memory + modes).

## Where data is stored

- Vector DB: `./data/chroma/` (local persistent directory)
- Uploaded / ingested text is stored as embedded documents in Chroma, and not sent anywhere.

## Privacy notes

- This project is designed to be **local-only by default**.
- If you enable an external model provider yourself, that would change the privacy properties (not included here).
