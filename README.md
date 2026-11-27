# Project 2 - Advanced AI

Implementasi algoritma machine learning untuk klasifikasi Student Academic Success Prediction 

Algoritma yang diimplementasikan:
- Decision Tree Learning
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Artificial Neural Network (ANN)

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

Kalau PowerShell gabisa menjalankan script, jalankan sekali:

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

## Cara Menjalankan Notebook
1. Aktifkan virtual environment.
2. Buka `src/model.ipynb`
3. Pilih kernel
4. Jalankan semua cell

## Pembagian Tugas
Nama: Ruhullah Reza Muhammad Khan

NIM: 23526002

Kontribusi: 
- Mengimplementasikan algoritma Decision Tree, Random Forest, ANN
- Melakukan EDA, Data Cleaning, Dan Transformation
- Melatih ketiga Model diatas
- Membuat Laporan mengenai hal tersebut

## Code Convention
* pastikan parameter fungsi menggunakan type signature