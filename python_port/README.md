# Python Port
## Run

```powershell
cd python_port
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.main:app --reload --port 8010
```

Open:

- `http://127.0.0.1:8010/`
- `http://127.0.0.1:8010/chat`
