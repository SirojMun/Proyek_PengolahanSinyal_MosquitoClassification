# Proyek Pengolahan Sinyal: Klasifikasi Suara Nyamuk (Mosquito Wingbeat Sound Classification)

Proyek ini mereplikasi dan mengembangkan metode klasifikasi spesies dan jenis kelamin nyamuk (8 kelas) berdasarkan sinyal dengung sayap (*wingbeat sound*) dari raw waveform 300 ms (sampling rate 8 kHz) di bawah kondisi *noise* lingkungan (outdoor noise).

Proyek ini mengacu pada paper:
> **Supratak et al. (2024)**, *"MosquitoSong+: A noise-robust deep learning model for mosquito classification from wingbeat sounds"*, *PLOS ONE*.

---

## 📊 Ringkasan Hasil Eksperimen

Evaluasi dilakukan menggunakan **10-Fold Cross-Validation** dengan metrik Accuracy (ACC) dan Macro-F1 (MF1) pada berbagai tingkat *gain* augmentasi ($G \in \{1, 1.5, 2\}$):

| Metode | Peran / Hipotesis | Fold | Akurasi (ACC) | Macro-F1 (MF1) | MF1 per Fold (Rerata ± SD) | $\Delta$ MF1 vs Replikasi MS+ | $\Delta$ MF1 vs Paper (0,911) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MosquitoSong+ (Replikasi)** | Baseline Utama | 10 | 93.76% | 0.9330 | 0.9327 ± 0.0233 | Acuan | +0.0220 |
| **MosquitoSong** | Tanpa augmentasi noise | 10 | 90.38% | 0.8991 | 0.8992 ± 0.0299 | -0.0339 | -0.0119 |
| **MFCC-CNN** | H2 (Representasi spektral) | 10 | 93.28% | 0.9313 | 0.9309 ± 0.0165 | -0.0017 | +0.0203 |
| **ECAPA-MFCC** | H2 (Arsitektur speaker recognition) | 10 | 93.77% | 0.9363 | 0.9356 ± 0.0238 | +0.0034 | +0.0253 |
| **DenseNet121-Spektrogram** | H2 (CNN 2D pra-latih) | 10 | 94.45% | 0.9428 | 0.9424 ± 0.0367 | +0.0099 | +0.0318 |
| **DenseNet121-ASP** | Eksploratori Attention Pooling | 10 | 94.50% | 0.9415 | 0.9412 ± 0.0160 | +0.0086 | +0.0305 |
| **MosquitoSong+ + MFCC** | **H3 (Dual-branch Feature Fusion)** | **10** | **97.21%** | **0.9711** | **0.9710 ± 0.0145** | **+0.0382** | **+0.0601** |
| **Ensemble** | **H4 (Ensemble Multi-Model)** | **10** | **97.04%** | **0.9707** | **0.9703 ± 0.0191** | **+0.0377** | **+0.0597** |

> **Temuan Utama:** Kombinasi representasi *raw waveform* dengan fitur spektral (*MosquitoSong+ + MFCC*) berhasil meningkatkan Macro-F1 secara signifikan dari baseline 93.30% menjadi **97.11%** ($p < 0.05$, terkoreksi dengan uji *corrected t-test* & *Holm-Bonferroni*), serta terbukti paling unggul pada pengujian bebas-kebocoran (*leakage-free evaluation*).

---

## 📁 Struktur Repositori

```text
├── 01_Replikasi/
│   ├── replikasi-01.ipynb             # Notebook replikasi MosquitoSong & MosquitoSong+
│   └── results/
│       ├── figures/                   # Grafik EDA, konfusi kurva, t-SNE, PSD sinyal
│       ├── logs/                      # Log training per fold
│       ├── runs/                      # Hasil prediksi per fold (preds.npz, history.csv)
│       ├── mosq_lib.py                # Library modular pemrosesan sinyal & arsitektur model
│       └── requirements.txt           # Dependensi modul replikasi
│
├── 02_Pembanding/
│   ├── pembanding-02.ipynb            # Notebook metode pembanding (MFCC, ECAPA, DenseNet, Fusi)
│   └── results/
│       ├── figures/                   # Kurva validasi perbandingan
│       ├── logs/                      # Log eksekusi model pembanding
│       ├── mosq_lib.py                # Library pendukung modul pembanding
│       └── requirements.txt
│
├── 03_Analisis/
│   ├── analisis-03.ipynb              # Notebook analisis statistik, uji hipotesis & biaya komputasi
│   ├── results.zip                    # Arsip hasil visualisasi dan ringkasan
│   └── results/
│       ├── figures/                   # Tabel CSV (uji hipotesis, biaya, F1) & visualisasi PNG
│       └── requirements.txt
│
├── .gitignore                         # Mengabaikan checkpoint besar (*.weights.h5), file zip, dan cache
└── README.md
```

---

## 🔬 Metodologi dan Data

1. **Kelas Sasaran (8 Kelas):**
   - *Aedes aegypti* (Betina & Jantan)
   - *Aedes albopictus* (Betina & Jantan)
   - *Anopheles dirus* (Betina & Jantan)
   - *Culex quinquefasciatus* (Betina & Jantan)
2. **Representasi Sinyal:**
   - Raw waveform 1D: durasi 300 ms (2400 sampel pada $f_s = 8000$ Hz)
   - Representasi spektral: Mel-Frequency Cepstral Coefficients (MFCC) & Log-Mel Spectrogram
3. **Augmentasi & Ketahanan Derau:**
   - Perturbasi waktu (*time shift*)
   - Skalasi amplitudo acak & derau Gaussian
   - Injeksi derau lingkungan outdoor (*N-OUTDOOR*) dengan probabilitas $p = 0.9$ dan faktor volume $G \sim \mathcal{U}(1, 2)$
4. **Validasi & Pengujian:**
   - 10-Fold Stratified Cross-Validation resmi
   - Pengujian signifikansi menggunakan *Nadeau-Bengio corrected repeated cross-validation t-test* dengan koreksi *Holm-Bonferroni*
   - Analisis *leakage-free* untuk memastikan generalisasi tidak bias

---

## 🚀 Cara Menjalankan

### Lingkungan Eksekusi
Notebook dirancang untuk dapat dijalankan baik di **Kaggle** (disarankan GPU T4 ×2) maupun di lingkungan **Lokal**.

1. **Clone repositori:**
   ```bash
   git clone https://github.com/SirojMun/Proyek_PengolahanSinyal_MosquitoClassification.git
   cd Proyek_PengolahanSinyal_MosquitoClassification
   ```

2. **Setup Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # atau .venv\Scripts\activate untuk Windows
   pip install -r 01_Replikasi/results/requirements.txt
   ```

3. **Urutan Eksekusi Notebook:**
   - Jalankan `01_Replikasi/replikasi-01.ipynb` untuk melatih baseline model MosquitoSong dan MosquitoSong+.
   - Jalankan `02_Pembanding/pembanding-02.ipynb` untuk melatih model-model pembanding (MFCC-CNN, ECAPA, DenseNet, Fusi Dual-Branch).
   - Jalankan `03_Analisis/analisis-03.ipynb` untuk mengumpulkan hasil metrik, menjalankan uji hipotesis statistik, dan memproduksi tabel serta grafik komparatif.

---

## 📜 Lisensi & Atribusi

Proyek ini dikembangkan untuk keperluan akademik dan penelitian Pengolahan Sinyal Digital.
Arsitektur dan data mengacu pada publikasi:
*Supratak et al., "MosquitoSong+: A noise-robust deep learning model for mosquito classification from wingbeat sounds", PLOS ONE, 2024.*
