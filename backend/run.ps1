$ErrorActionPreference = 'Stop'

$env:PYTHONPATH = "$PSScriptRoot"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
