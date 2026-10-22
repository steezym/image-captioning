# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 22/10/2026
- **Anggota:** Anggota 2 - Data Pipeline & Preprocessing
- **Target Modul:** `src/vocab.py`, `src/dataset.py`, dan validasi Flickr8k
- **LLM yang dipakai:** Codex
- **Tautan Sesi Chat:** [lampirkan tautan/ekspor sesi ini]

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > Implementasikan dan validasi hanya tanggung jawab Anggota 2: pipeline data dan preprocessing Flickr8k sesuai kontrak README.
- **Alasan Khusus:** Dataset harus di-split berdasarkan gambar sebelum vocabulary dibangun agar caption validation/test tidak bocor ke vocabulary training.
- **Output Model:**
  - `Vocabulary` dengan tokenisasi lowercase deterministik, apostrof internal, `min_freq=5`, serta ID khusus 0-3.
  - Parser caption yang memisahkan hanya koma pertama.
  - Split gambar deterministik seed 42 menjadi 6.091 train, 1.000 validation, dan 1.000 test.
  - Transform gambar 128x128, float32, normalisasi [-1, 1], tanpa horizontal flip.
  - Dataset pasangan gambar-caption, DataLoader, tes sintetis, smoke test penuh, dan notebook bukti.

### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror dari lingkungan lokal:**
  ```
  ModuleNotFoundError: No module named 'torch'
  ```
- **Prompt Perbaikan ke LLM:**
  > Validasi implementasi secara nyata pada environment lokal dan jalankan smoke test terhadap archive.zip.
- **Analisis & Solusi Kelompok:** Environment lokal hanya memiliki Python 3.14 tanpa PyTorch/torchvision/pytest. Dependensi dipasang untuk validasi lokal, kemudian tes anggota 2 dijalankan ulang. Ini bukan kesalahan pipeline.

### 3. TAHAP OPTIMASI
- **Prompt:**
  > Jaga preprocessing ringan, reproducible, dan mudah digunakan di Google Colab.
- **Perubahan yang dilakukan:** Dataset diekstrak sekali ke `data/`; transform training hanya memakai `ColorJitter` ringan tanpa horizontal flip; validation dan test deterministik.
- **Hasil:** Seluruh tes sintetis dan smoke test Flickr8k penuh lulus.

## HASIL VALIDASI AKHIR

- `11 passed` untuk tes vocabulary, dataset sintetis, dan smoke test penuh.
- Gambar unik: 8.091; setiap gambar memiliki tepat 5 caption.
- Split: 6.091 train, 1.000 validation, 1.000 test; seluruh overlap bernilai 0.
- Pasangan gambar-caption: 30.455 train, 5.000 validation, 5.000 test.
- Vocabulary train-only: 2.579 token; 1.042 token khusus held-out tidak bocor ke vocabulary.
- Token khusus: `<pad>=0`, `<start>=1`, `<end>=2`, `<unk>=3`.
- Batch nyata: image `(8, 3, 128, 128)` `float32`, caption `(8, 30)` `int64`.
- Rentang batch gambar yang diuji: `[-0.992157, 0.992157]`, seluruh nilai finite.

## VALIDASI LOKAL WINDOWS

- Ditambahkan `run_data_local.py` agar pipeline dapat dijalankan dari direktori mana pun.
- Notebook mencari root repo secara otomatis dari current working directory atau environment variable `IMAGE_CAPTIONING_ROOT`.
- Notebook dieksekusi penuh secara lokal dan menghasilkan batch image `(64, 3, 128, 128)` serta caption `(64, 30)`.
- Dependensi lokal dicatat di `requirements-data.txt`.

---

**Commit terkait:** [isi setelah commit]
