# AI Gaze studio (archived Streamlit notes)

The live studio now runs as **FastAPI** on Railway:

- App: `api_app.py` + `templates/studio.html`
- Engine: `engine/` (DeepGaze + PDF)
- Legacy Streamlit monolith: `archive/app_streamlit_legacy.py`

Local:

```bash
uvicorn api_app:app --host 0.0.0.0 --port 8080
```

Open http://127.0.0.1:8080/studio
