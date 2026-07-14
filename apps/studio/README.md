# AI Gaze™ Studio (Streamlit)

The interactive analysis tool — heat maps, hot spots, gaze path, clarity, PDF export.

## Why it's at the repo root

Streamlit Cloud currently deploys from the repository root (`app.py`, `requirements.txt`).
Those files remain the **live studio entrypoint**.

```
ai-gaze/
├── app.py                 ← Streamlit Cloud main file (studio)
├── requirements.txt
├── aigaze_logo.png
├── apps/
│   ├── web/               ← Next.js product marketing (Ethos+-style)
│   └── studio/            ← this folder (docs + future home)
└── package.json
```

## Run locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
# or: npm run studio
```

## Future

When Streamlit Cloud is reconfigured, move `app.py` + assets here and set Main file path to `apps/studio/app.py`.
