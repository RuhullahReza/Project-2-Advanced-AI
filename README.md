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
| Nama | NIM | Pembagian Tugas |
| ------- | ------- | ------- |
| Ruhullah Reza Muhammad Khan | 23526002 | <ul><li>Mengimplementasikan algoritma Decision Tree, Random Forest, ANN</li><li>Melakukan EDA, Data Cleaning, dan Transformation</li><li>Melatih ketiga model di atas (from-scratch dan sklearn)</li><li>Membuat laporan mengenai hal tersebut</li></ul> |
| Ananda Myzza Marhelio | 23526008 | <ul><li>Mengimplementasikan algoritma Logistic Regression</li><li>Melatih model tersebut (from-scratch dan sklearn)</li><li>Membuat laporan mengenai hal tersebut</li></ul> |
| Muhammad Adam Dzulqarnain | 23526031 | <ul><li>Mengimplementasikan algoritma KNN dan Ensemble (5 Base Model)</li><li>Melatih kedua model tersebut (from-scratch dan sklearn)</li><li>Membuat laporan mengenai hal tersebut</li></ul> |


## Code Convention
* pastikan parameter fungsi menggunakan type signature