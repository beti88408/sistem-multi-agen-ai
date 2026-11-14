# 🛡️ Integrated Multi-Agent HR, Payroll & Trade Governance Simulation System

Aplikasi Streamlit dengan 5 agen AI (Groq API) untuk audit SDM/Payroll dan data Ekspor-Impor/Kepabeanan.

| Agen | Tugas |
|---|---|
| 1. Data Sanitizer | Validasi kolom (pemetaan oleh AI), hapus kosong/duplikat, standarisasi angka & tanggal |
| 2. Compliance & Classification | Uji PPh21, BPJS, UMP, lembur / uji kecocokan HS Code |
| 3. Anomaly & Valuation | Outlier statistik, ghost employee, under/over-valuation, skor risiko |
| 4. Risk & Macro | Konsentrasi (HHI), narasi risiko, simulasi what-if |
| 5. Executive Reporting & Copilot | Kertas kerja audit + Chat with Data |

Aritmetika dan statistik dihitung pandas (akurat dan hemat token); LLM dipakai untuk penalaran. Tanpa API key, aplikasi tetap jalan dengan mode deterministik.

## Struktur
```
app.py  agents.py  sample_data.py  requirements.txt  README.md  .gitignore
.streamlit/secrets.toml   (template)
```

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Masukkan GROQ_API_KEY di sidebar (buat gratis di https://console.groq.com/keys), klik **Gunakan Data Sampel Bawaan**, lalu **Jalankan Analisis Multi-Agen**.

## Upload ke GitHub
1. Buat repository baru di https://github.com/new (mis. `hr-trade-governance`).
2. Di folder proyek:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/USERNAME/hr-trade-governance.git
git push -u origin main
```
3. ⚠️ **Jangan pernah menaruh API key asli di `.streamlit/secrets.toml` yang di-push.** File itu hanya template.

## Deploy ke Streamlit Community Cloud
1. Buka https://share.streamlit.io lalu login dengan GitHub.
2. Klik **Create app** → pilih repository, branch `main`, main file `app.py`.
3. Buka **Advanced settings → Secrets**, tempel:
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxx"
```
4. Klik **Deploy**. Untuk mengubah key kemudian: App → Settings → Secrets.

## Format data
- Dataset bawaan: letakkan file di folder yang sama dengan `app.py` dan sesuaikan `DEFAULT_FILES` di bagian atas `app.py`; file ini dibaca otomatis saat aplikasi dibuka (unggah manual tetap tersedia di sidebar).
- **Payroll** (wajib): `employee_id, nama, departemen, gaji_pokok, tunjangan, lembur_jam, lembur_rp, bpjs_kesehatan, bpjs_jht, pph21, gaji_bersih, no_rekening`; opsional: `hari_hadir, skor_kinerja, tanggal_masuk`.
- **Trade** (wajib): `id_transaksi, pemasok, negara_asal, deskripsi_barang, kode_hs, kuantitas, nilai_usd`; opsional: `tanggal, satuan, tarif_bm`.
- Jika nama kolom berbeda, Agen 1 mencoba memetakannya otomatis (butuh API key).

## Catatan
- Estimasi PPh21 memakai pendekatan progresif sederhana (TK/0), bukan skema TER/PTKP individual. Sesuaikan bila perlu.
- Model `mixtral-8x7b-32768` bisa sudah dihentikan Groq; kode otomatis beralih ke `llama-3.1-8b-instant`.
- Rate limit Groq ditangani dengan jeda antar-panggilan, retry, dan exponential backoff.
