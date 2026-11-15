# Project 2 - Advanced AI

## Setup Environment

Butuh Python 3.11.

```bash
python -m venv .venv
```

### Aktivasi

| Shell | Perintah |
|---|---|
| PowerShell | `.\.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

Kalau PowerShell menolak menjalankan script, jalankan sekali:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Install dependency

```bash
pip install -r requirements.txt
```

### Nonaktifkan

```bash
deactivate
```

## Code Convention
* pastikan parameter fungsi menggunakan type signature