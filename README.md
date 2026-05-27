# Eksperimen SML - Feomita Ramadhany Fudiansah

Repository ini dibuat untuk memenuhi **Kriteria 1: Melakukan Eksperimen terhadap Dataset Pelatihan** pada kelas Membangun Sistem Machine Learning.

## Dataset

Dataset yang digunakan adalah **Breast Cancer Wisconsin Diagnostic Dataset** yang tersedia melalui `sklearn.datasets.load_breast_cancer`. Dataset raw telah diekspor ke file CSV agar dapat diproses ulang secara konsisten.

Target klasifikasi:

- `diagnosis = 0`: malignant
- `diagnosis = 1`: benign

## Struktur Repository

```text
Eksperimen_SML_Feomita_Ramadhany_Fudiansah
├── .github
│   └── workflows
│       └── preprocessing.yml
├── breast_cancer_raw
│   └── breast_cancer_raw.csv
├── preprocessing
│   ├── Eksperimen_Feomita_Ramadhany_Fudiansah.ipynb
│   ├── automate_Feomita_Ramadhany_Fudiansah.py
│   └── breast_cancer_preprocessing
│       └── breast_cancer_preprocessed.csv
├── requirements.txt
└── README.md
```

## Tahapan Eksperimen Manual

Tahapan eksperimen manual dilakukan di notebook `preprocessing/Eksperimen_Feomita_Ramadhany_Fudiansah.ipynb`, meliputi:

1. Import library.
2. Data loading.
3. Exploratory Data Analysis.
4. Data preprocessing.
5. Penyimpanan dataset hasil preprocessing.

## Tahapan Preprocessing Otomatis

Script otomatisasi tersedia pada:

```bash
preprocessing/automate_Feomita_Ramadhany_Fudiansah.py
```

Jalankan script dengan perintah berikut dari root repository:

```bash
python preprocessing/automate_Feomita_Ramadhany_Fudiansah.py   --input breast_cancer_raw/breast_cancer_raw.csv   --output preprocessing/breast_cancer_preprocessing
```

Output yang dihasilkan:

```text
preprocessing/breast_cancer_preprocessing/breast_cancer_preprocessed.csv
```

## GitHub Actions

Workflow GitHub Actions tersedia di:

```text
.github/workflows/preprocessing.yml
```

Workflow akan berjalan saat:

- `push` ke branch `main` atau `master`.
- `pull_request`.
- dijalankan manual melalui `workflow_dispatch`.

Workflow akan menjalankan script preprocessing dan mengunggah dataset hasil preprocessing sebagai artifact.
