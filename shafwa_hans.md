# Proses Analisis Sentimen Komentar TikTok Prabowo (PBAAnalysisSentiment)

Dokumen ini menjelaskan alur dari data mentah sampai skor model, beserta alasan di balik tiap keputusan dan batasan hasilnya.
Tugas: klasifikasi biner **negatif / positif** terhadap komentar TikTok tentang Prabowo.

## Ringkasan alur

```
Label 2 annotator (shafwa + hans)
   -> agregasi label (kappa 0.7449), netral dibuang
   -> 01 cleaning + split awal
   -> 02 embedding (bge-m3) + clustering
   -> 03 split per cluster, undersample negatif, oversample positif
   -> (03b) positif targeted untuk topik yang bias (mis. MBG)
   -> 04 fine-tune IndoBERTweet + evaluasi + eksperimen
```

| Tahap | Notebook | Keluaran |
|---|---|---|
| Agregasi label | - | `final_label` per komentar |
| Cleaning & split awal | 01 | `train_raw.csv`, `test_raw.csv` |
| Embedding & cluster | 02 | `clustered.csv` (+ `cluster_id`) |
| Split, resample, oversample | 03 | `train_balanced.csv`, `test_final.csv` |
| Positif targeted (opsional) | 03b | `targeted_positives.csv` |
| Fine-tune & evaluasi | 04 (v3) | folder `model_indobertweet/<run>/`, `metrics.json`, tabel perbandingan |

---

## 1. Pelabelan dan agregasi

- Dua annotator (shafwa dan hans) melabeli komentar.
- Kesepakatan antar-annotator: **Cohen's kappa = 0.7449** (substantial agreement).
- Label akhir (`final_label`) diambil dari hasil agregasi kedua annotator.
- Kelas **netral dibuang**, jadi masalahnya biner: `negatif` vs `positif`.

Catatan: karena netral dibuang, komentar yang sebenarnya cuma memuji program atau netral terhadap Prabowo bisa bernasib tidak jelas. Definisi "positif" sebaiknya tertulis eksplisit di laporan (positif = sikap mendukung/membela Prabowo, bukan sekadar nada kalimat positif).

## 2. Cleaning dan split awal (notebook 01)

- Membersihkan komentar (duplikat, kosong, dsb.) dan memisahkan data awal menjadi `train_raw.csv` dan `test_raw.csv`.
- Pembersihan di sini belum memotong isi komentar untuk model. Preprocessing khusus IndoBERTweet dilakukan di tahap 04 (lihat bagian 7).

## 3. Embedding dan clustering (notebook 02)

**Tujuan:** komentar yang hampir sama (copy-paste, spam, variasi kecil) dikelompokkan, supaya saat split tidak ada komentar yang "mirip" muncul di train dan test sekaligus (data leakage).

- **Embedding:** model `bge-m3` lewat Ollama (lokal), hasilnya di-cache supaya tidak diulang.
- **Clustering:** agglomerative clustering dengan ambang kemiripan (`SIM_THRESHOLD`) **0.70**, dilakukan per kelas.
- Percobaan awal dengan ambang 0.80 menghasilkan ±92% cluster singleton (hampir tidak ada yang dikelompokkan), jadi diturunkan ke **0.70**. Dengan 0.70, negatif menjadi ±884 cluster.
- Keluaran: `clustered.csv`, yaitu data asli + kolom `cluster_id`.

## 4. Split per cluster, undersample, oversample (notebook 03)

### 4.1 Split train/test per cluster
- `GroupShuffleSplit` dengan `groups=cluster_id`, `test_size=0.20`, `seed=42`.
- Satu cluster **seluruhnya** masuk train atau seluruhnya test. Ada assert: tidak ada `cluster_id` dan tidak ada `comment_text` yang sama di keduanya.
- **Test = distribusi asli** (tidak di-resample), supaya skor mencerminkan keadaan nyata. Hasil: `test_final.csv` = **446 komentar**, hanya **51 positif**.

### 4.2 Undersample negatif (khusus train)
- Negatif jauh lebih banyak daripada positif.
- Dipilih **750** negatif: 1 wakil per cluster dulu (agar variasi terjaga), sisa kuota diisi dari cluster besar dengan peluang proporsional ukuran cluster.

### 4.3 Oversample positif (khusus train)
- Positif asli di train hanya **338**.
- Kekurangan sampai 750 diisi **komentar sintetis** buatan LLM lokal (`qwen2.5:7b` via Ollama) memakai contoh positif asli sebagai few-shot.
- Dibuang jika sama persis dengan positif asli atau komentar test.
- Hasil `train_balanced.csv`: **1500 baris** (750 negatif; 338 positif asli + 412 positif sintetis).
- Kolom `sumber` menandai asal data: `asli` / `synthetic` (nanti juga `targeted`).

Catatan: setelah ini val tidak diambil dari test. Val diambil dari **data asli** di train (12% per kelas, `VAL_SEED=42`), dan test **tidak pernah** dipakai memilih model atau menentukan contoh few-shot.

## 5. Diagnosis awal: kenapa recall positif rendah

Setelah training pertama, recall positif hanya sekitar **0.70**. Analisis error menunjukkan:

- Model belajar **shortcut topik/prior**: kata `mbg` -> negatif, tanda `?` -> negatif.
  Di data asli, komentar yang menyebut MBG hanya **9% positif**, dan komentar ber-`?` hanya **9% positif**.
- Positif yang dilabel manusia ternyata **sikap terhadap Prabowo** (membela, humor, sarkas, bergantung konteks), bukan sekadar kata-kata bernada positif. Contoh: "tetep aja salah kalau di mata pembenci" itu positif, tapi mengandung kata negatif.
- Positif sintetis generik dari LLM menurunkan **presisi** (model jadi terlalu mudah bilang positif), sedangkan AUC hampir sama antar konfigurasi.

## 6. Positif targeted (notebook 03b dan `targeted_positives.csv`)

**Tujuan:** menambah variasi positif untuk topik yang bias negatif (khusus `mbg`) supaya model tidak langsung menjawab negatif hanya karena kata itu muncul.

- 03b menghitung tabel log-odds bias kata, lalu memilih keyword secara manual (bukan otomatis, karena kata seperti "gagal" atau "kecewa" memang negatif murni).
- Percobaan pertama memakai qwen2.5:7b gagal: menganggap MBG sebagai orang, mengarang fakta, dan menempelkan "?" secara paksa. Hasilnya ditolak.
- Perbaikan: prompt diberi glosarium (MBG = Makan Bergizi Gratis, sebuah program), aturan "tulis seperti orang biasa, bukan iklan", dan "?" tidak lagi dipaksakan.
- Dipakai juga file `targeted_positives.csv` berisi **80 komentar** MBG yang ditulis tangan oleh Claude. Ini **data sintetis** dan harus dilaporkan sebagai sintetis.
- Di notebook 04, baris targeted yang mirip val/test (Jaccard token >= 0.7) otomatis dibuang. Kolom `dipakai=0` bisa dipakai untuk mematikan baris yang jelek.

## 7. Fine-tuning IndoBERTweet (notebook 04 v3)

**Model:** `indolem/indobertweet-base-uncased`.

**Preprocessing (wajib untuk model ini):** lowercase, URL -> `HTTPURL`, mention -> `@USER`, emoji -> teks via `emoji.demojize(language="id")`.

**Setelan training:**

| Parameter | Nilai |
|---|---|
| MAX_LEN | 96 |
| Epoch | 3 |
| Learning rate | 2e-5 (AdamW dari `torch.optim`) |
| Batch | 8 x gradient accumulation 2 |
| Presisi | fp16 autocast + GradScaler |
| Padding | dinamis (collate) |
| Pemilihan checkpoint | val macro-F1 terbaik |
| Loss | CrossEntropy, opsional class weight |

**Pengaman GPU** (laptop RTX 2060 6GB): cek suhu via `nvidia-smi`; jeda di 78 C, lanjut di 72 C, berhenti di 86 C; jeda antar step, antar epoch (60 detik), dan antar run (90 detik).

**Desain eksperimen:**
- `SYN_MODE` in {`none`, `generic`, `targeted`, `both`} dikombinasi dengan `CLASS_WEIGHT` dan 3 seed (42, 43, 44).
- Nama run: `real` / `synth` / `targ` / `both`, `cw` / `nocw`, `s<seed>`.
- Sweep bisa dilanjutkan (resume); run yang sudah ada dilewati.
- Tiap run menyimpan `metrics.json`, `best/` checkpoint, dan `test_predictions.csv`.

## 8. Metrik dan cara membaca skor

- **Metrik utama:** macro-F1 pada test (adil untuk dua kelas yang tidak seimbang), plus precision, recall, dan F1 per kelas.
- **Confusion matrix:** train dan test, dalam jumlah dan persen per kelas.
- **Ketidakpastian:** rata-rata +- std dari 3 seed, dan bootstrap 95% CI.
- **Alat diagnosis:**
  - slice report (mis. subset komentar berisi `mbg` atau `?`),
  - minimal-pair (kalimat sama, topik diganti),
  - keyword leakage check,
  - threshold sweep,
  - daftar error (positif yang terlewat).

### Hasil yang sudah ada

| Konfigurasi | Macro-F1 (rata-rata +- std) | Recall positif | Presisi positif |
|---|---|---|---|
| `real_cw` (tanpa sintetis) | _isi dari tabel notebook_ | ~0.70 | _isi_ |
| `synth_nocw` (sintetis generik) | _isi_ | _isi_ | lebih rendah |
| `targ_cw` (asli + 80 targeted) | _isi_ | _isi_ | _isi_ |

Temuan kualitatif dari eksperimen:
- Sintetis generik **tidak menolong**: presisi positif turun, AUC sama.
- Targeted data menaikkan P(positif) di minimal-pair: untuk topik `mbg` rata-rata dari **0.07 menjadi 0.72**. Tetapi naik juga di topik lain (program ini 0.25 -> 0.63, ekonomi 0.39 -> 0.63), yang bisa berarti model belajar gaya tulisan rapi, bukan stance.
- Pada slice `mbg` di test (32 komentar, hanya 1 positif), recall negatif turun dari 100% menjadi 94/94/90% di tiga seed, artinya 2-3 negatif jadi salah. Perbaikan di data asli belum terbukti.

## 9. Batasan yang harus disebut di laporan

1. **Test kecil:** hanya 51 positif, jadi selisih 1-3 poin recall masih noise. Slice kecil (mis. `?` dengan 6 positif) sangat tidak stabil antar seed.
2. **Data sintetis:** 412 positif LLM dan 80 positif buatan Claude. Harus disebut sebagai sintetis, dan hasilnya jangan diklaim sebagai performa data nyata.
3. **Bias seleksi:** keyword MBG dipilih karena melihat error di test, jadi perbaikan di slice itu cenderung optimistis. Perlu holdout MBG yang segar (±30 positif asli) yang belum pernah dilihat siapa pun.
4. **Prior bergeser:** di data asli MBG hanya 9% positif; setelah targeted menjadi 51%. Itu bisa membuat model menganggap MBG netral padahal dunia nyata condong negatif.
5. **Minimal-pair adalah syarat perlu, bukan bukti cukup**, dan template-nya perlu dilengkapi kontrol negatif.
6. **Definisi label:** pujian pada program (tanpa menyebut Prabowo) bisa dilabel netral/negatif/positif tergantung aturan anotasi. Samakan dengan annotator kedua.

## 10. Langkah lanjutan yang disarankan

1. Tambah kontrol negatif pada minimal-pair dan tulis sebagian template dengan gaya TikTok yang berantakan.
2. Uji dosis targeted lebih kecil (mis. 20-30 baris), pilih lewat val, bukan test.
3. Buat holdout MBG asli: filter komentar berisi `mbg` + kata dukungan, lalu label bersama.
4. Tambah konfigurasi `("none", False)` untuk memisahkan efek class weight dari efek data.
5. Prioritaskan menambah positif asli (sarkas, membela) daripada sintetis.
6. Opsional: ensemble antar seed dan ambang keputusan yang dipilih dari val.