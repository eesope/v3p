Step by step for local test using model:

1. start server @localhost:
```bash
uvicorn model_api:app --reload --log-level debug
```

2. runtime command
```bash
uvicorn model_api:app --host 0.0.0.0 --port #
```

3. query via url
url/generate-txt?prompt=Three%20cats%20are