"""agents.py - 5 Agen AI Spesialis (Groq API) + logika deterministik pendukung.
Prinsip hemat token: aritmetika/statistik dikerjakan pandas; LLM hanya untuk penalaran
(pemetaan kolom, klasifikasi HS, rekomendasi, narasi, chat) dengan konteks ringkas."""
import json
import re
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
FALLBACK = "llama-3.1-8b-instant"
_LAST_CALL = [0.0]

ID = {"payroll": "employee_id", "trade": "id_transaksi"}
GRP = {"payroll": "departemen", "trade": "pemasok"}
VAL = {"payroll": "gaji_bersih", "trade": "nilai_usd"}
CUR = {"payroll": "Rp", "trade": "USD"}
REQ = {"payroll": ["employee_id", "nama", "departemen", "gaji_pokok", "tunjangan", "lembur_jam", "lembur_rp",
                   "bpjs_kesehatan", "bpjs_jht", "pph21", "gaji_bersih", "no_rekening"],
       "trade": ["id_transaksi", "pemasok", "negara_asal", "deskripsi_barang", "kode_hs", "kuantitas", "nilai_usd"]}
CRIT = {"payroll": ["employee_id", "gaji_pokok"], "trade": ["id_transaksi", "kuantitas", "nilai_usd"]}
NUM = {"payroll": ["gaji_pokok", "tunjangan", "lembur_jam", "lembur_rp", "bpjs_kesehatan", "bpjs_jht", "pph21",
                   "gaji_bersih", "hari_hadir", "skor_kinerja"], "trade": ["kuantitas", "nilai_usd", "tarif_bm"]}
FILL0 = {"payroll": ["tunjangan", "lembur_jam", "lembur_rp", "bpjs_kesehatan", "bpjs_jht", "pph21", "gaji_bersih"], "trade": []}
KW_HS = {"kopi": "09", "beras": "10", "sawit": "15", "obat": "30", "karet": "40", "kain": "52", "sepatu": "64",
         "laptop": "84", "pompa": "84", "mesin": "84", "telepon": "85", "smartphone": "85", "baterai": "85", "kursi": "94"}


# ------------------------------------------------------------------ LLM client
class LLM:
    """Pembungkus Groq: throttle antar-panggilan, retry+backoff saat rate-limit, fallback model."""

    def __init__(self, api_key=None, model=MODELS[0], min_interval=1.2, max_retries=4):
        self.model, self.min_interval, self.max_retries = model, min_interval, max_retries
        self.errors, self.calls, self.client = [], 0, None
        if api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
            except Exception as e:
                self.errors.append(f"Inisialisasi Groq gagal: {e}")

    @property
    def active(self):
        return self.client is not None

    def chat(self, system, user, json_mode=False, max_tokens=800, temperature=0.1, history=None):
        if not self.active:
            return ""
        msgs = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": user}]
        kw = dict(model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - _LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            try:
                _LAST_CALL[0] = time.time()
                self.calls += 1
                r = self.client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as e:
                m = str(e).lower()
                if "401" in m or "invalid api key" in m:
                    self.errors.append("API key tidak valid.")
                    return ""
                if "429" in m or "rate" in m:
                    time.sleep(min(2 ** attempt * 3, 30))
                    continue
                if any(k in m for k in ("decommission", "not found", "does not exist")) and kw["model"] != FALLBACK:
                    self.errors.append(f"Model {kw['model']} tidak tersedia, beralih ke {FALLBACK}.")
                    kw["model"] = FALLBACK
                    continue
                self.errors.append(str(e)[:200])
                time.sleep(1.5)
        self.errors.append("Panggilan Groq gagal setelah beberapa percobaan.")
        return ""

    def json(self, system, user, **kw):
        t = self.chat(system, user, json_mode=True, **kw)
        if not t:
            return {}
        try:
            return json.loads(t)
        except Exception:
            m = re.search(r"\{.*\}", t, re.S)
            try:
                return json.loads(m.group(0)) if m else {}
            except Exception:
                return {}


# ------------------------------------------------------------------ util
def est_pph21_month(gross_month, ptkp=54_000_000):
    """Estimasi PPh21 bulanan (progresif UU HPP, TK/0, biaya jabatan 5% maks Rp6 jt/thn). Pendekatan sederhana."""
    a = np.asarray(gross_month, float) * 12
    pkp = np.maximum(a - np.minimum(a * 0.05, 6_000_000) - ptkp, 0)
    tax, lo = np.zeros_like(pkp), 0.0
    for hi, rate in [(60e6, .05), (250e6, .15), (500e6, .25), (5e9, .30), (np.inf, .35)]:
        tax += np.clip(pkp - lo, 0, hi - lo) * rate
        lo = hi
    return tax / 12


def to_num(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = re.sub(r"[^\d.,\-]", "", str(v))
    if s in ("", "-"):
        return np.nan
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def robust_z(x, g):
    x = pd.Series(np.asarray(x, float))
    g = pd.Series(np.asarray(g))
    med = x.groupby(g).transform("median")
    mad = (x - med).abs().groupby(g).transform("median")
    mad = mad.where(mad > 0, med.abs() * 0.1 + 1e-9)
    return (0.6745 * (x - med) / mad).values, med.values


def md(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


class Findings:
    """Penampung temuan per baris dari semua agen: skor, alasan, kategori, dampak."""

    def __init__(self, n):
        self.n, self.score, self.impact = n, np.zeros(n), np.zeros(n)
        self.reasons, self.cats, self.extra = [[] for _ in range(n)], [[] for _ in range(n)], {}

    def add(self, mask, pts, text, cat, imp=None):
        mask = np.asarray(mask, bool)
        imp = np.zeros(self.n) if imp is None else np.nan_to_num(np.abs(np.asarray(imp, float)))
        for i in np.flatnonzero(mask):
            self.score[i] += pts
            self.impact[i] = max(self.impact[i], imp[i])
            self.reasons[i].append(text if isinstance(text, str) else text[i])
            if cat not in self.cats[i]:
                self.cats[i].append(cat)


# ------------------------------------------------------------------ Agen 1
class SanitizerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, df, mode):
        rep = {"baris_awal": len(df)}
        df = df.copy()
        df.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        req = REQ[mode]
        miss = [c for c in req if c not in df.columns]
        mapping = {}
        if miss and self.llm.active:
            out = self.llm.json(
                'Anda Data Sanitizer. Petakan kolom dataset ke nama kolom standar. Balas HANYA JSON: {"mapping":{"standar":"kolom_asli atau null"}}',
                f"Kolom standar yang hilang: {miss}\nKolom dataset: {list(df.columns)}\nContoh: {df.head(2).astype(str).to_dict('records')}",
                max_tokens=500)
            for k, v in (out.get("mapping") or {}).items():
                if k in miss and v in df.columns and v not in req:
                    df = df.rename(columns={v: k})
                    mapping[k] = v
        miss = [c for c in req if c not in df.columns]
        rep["kolom_hilang"], rep["pemetaan_kolom_ai"] = miss, mapping
        if miss:
            return df, rep
        n0 = len(df)
        df = df.dropna(how="all")
        rep["baris_kosong_dihapus"] = n0 - len(df)
        for c in df.select_dtypes("object").columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan, "None": np.nan, "NaT": np.nan})
        for c in NUM[mode]:
            if c in df:
                df[c] = df[c].map(to_num)
        for c in df.columns:
            if "tanggal" in c or c.startswith("tgl") or "date" in c:
                df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
        n1 = len(df)
        df = df.dropna(subset=CRIT[mode])
        rep["baris_data_kritis_kosong"] = n1 - len(df)
        for c in FILL0[mode]:
            df[c] = df[c].fillna(0)
        for c in req:
            if c not in NUM[mode]:
                df[c] = df[c].fillna("N/A")
        n2 = len(df)
        df = df.drop_duplicates().drop_duplicates(subset=[ID[mode]]).reset_index(drop=True)
        rep["duplikat_dihapus"], rep["baris_akhir"] = n2 - len(df), len(df)
        rep["kolom_opsional_terdeteksi"] = [c for c in NUM[mode] if c in df and c not in req]
        return df, rep


# ------------------------------------------------------------------ Agen 2
class ComplianceAgent:
    SYS_HS = ('Anda ahli klasifikasi HS (Harmonized System/BTKI). Nilai apakah kode HS sesuai deskripsi barang. '
              'Balas HANYA JSON: {"items":[{"i":<idx>,"ok":true|false,"hs_saran":"4 digit","alasan":"maks 12 kata"}]}')

    def __init__(self, llm):
        self.llm = llm

    def run(self, d, mode, F, params):
        return self.payroll(d, F, params.get("ump", 3_500_000)) if mode == "payroll" else self.trade(d, F)

    def payroll(self, d, F, ump):
        gp, tj, lr, lj = (d[c].values for c in ("gaji_pokok", "tunjangan", "lembur_rp", "lembur_jam"))
        pph, bk, bj, gn = (d[c].values for c in ("pph21", "bpjs_kesehatan", "bpjs_jht", "gaji_bersih"))
        ep = est_pph21_month(gp + tj + lr)
        F.add((pph < ep * .6) & (ep - pph > 100_000), 25, "PPh21 dipotong jauh di bawah estimasi progresif", "PPh21", ep - pph)
        F.add((pph > ep * 1.6) & (pph - ep > 100_000), 20, "PPh21 dipotong melebihi estimasi progresif", "PPh21", pph - ep)
        ek, ej = np.minimum(gp, 12e6) * .01, gp * .02
        F.add(np.abs(bk - ek) > .05 * ek + 1000, 15, "Iuran BPJS Kesehatan tidak sesuai 1% (batas upah Rp12 jt)", "BPJS", bk - ek)
        F.add(np.abs(bj - ej) > .05 * ej + 1000, 15, "Iuran BPJS JHT tidak sesuai 2%", "BPJS", bj - ej)
        F.add(gp < ump, 40, f"Gaji pokok di bawah UMP/UMK acuan (Rp{ump:,.0f})".replace(",", "."), "Upah Minimum", ump - gp)
        en = gp + tj + lr - bk - bj - pph
        F.add(np.abs(en - gn) > 1000, 30, "Gaji bersih tidak cocok dengan komponen (bruto - potongan)", "Kalkulasi Gaji", gn - en)
        F.add(lj > 72, 35, "Jam lembur melebihi batas wajar 72 jam/bulan", "Lembur")
        el = gp / 173 * lj * 2.2
        F.add(lr > el + 1000, 25, "Upah lembur melebihi perhitungan maksimal (1/173 x jam x 2,2)", "Lembur", lr - el)

    def trade(self, d, F):
        hs = d["kode_hs"].astype(str).str.replace(r"\D", "", regex=True)
        F.add(((hs.str.len() < 6) | (hs.str.len() > 10)).values, 20, "Format kode HS tidak valid (harus 6-10 digit)", "HS Code")
        pairs = d.assign(hs_clean=hs)[["deskripsi_barang", "hs_clean"]].drop_duplicates().reset_index(drop=True).head(60)
        verdict = {}
        if self.llm.active:
            for s in range(0, len(pairs), 20):
                ch = pairs.iloc[s:s + 20]
                lines = "\n".join(f"{i}|{r['deskripsi_barang']}|{r['hs_clean']}" for i, r in ch.iterrows())
                out = self.llm.json(self.SYS_HS, "Format: idx|deskripsi|kode_hs\n" + lines, max_tokens=1500)
                for it in out.get("items", []):
                    try:
                        r = pairs.loc[int(it["i"])]
                        verdict[(r["deskripsi_barang"], r["hs_clean"])] = (bool(it.get("ok", True)), str(it.get("hs_saran", "")), str(it.get("alasan", "")))
                    except Exception:
                        pass
        if not verdict:  # fallback offline berbasis kata kunci
            for _, r in pairs.iterrows():
                for kw, chp in KW_HS.items():
                    if kw in str(r["deskripsi_barang"]).lower():
                        verdict[(r["deskripsi_barang"], r["hs_clean"])] = (r["hs_clean"][:2] == chp, chp, f"Deskripsi mengindikasikan bab HS {chp}")
                        break
        mask, texts = np.zeros(len(d), bool), [""] * len(d)
        for j, (ds, h) in enumerate(zip(d["deskripsi_barang"], hs)):
            v = verdict.get((ds, h))
            if v and not v[0]:
                mask[j], texts[j] = True, f"HS {h} tidak sesuai deskripsi; saran HS {v[1]}: {v[2]}"
        F.add(mask, 45, texts, "HS Misclassification")


# ------------------------------------------------------------------ Agen 3
class AnomalyAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, d, mode, F):
        if mode == "payroll":
            gp, tj, gn = d["gaji_pokok"].values, d["tunjangan"].values, d["gaji_bersih"].values
            z, med = robust_z(gp, d["departemen"])
            F.add(z > 3.5, 30, "Gaji pokok outlier ekstrem dibanding departemennya", "Outlier Gaji", gp - med)
            F.add(tj > gp, 30, "Tunjangan melebihi gaji pokok (rasio >100%)", "Tunjangan Anomali", tj - gp)
            dup = (d.duplicated("no_rekening", keep=False) & d["no_rekening"].ne("N/A")).values
            F.add(dup, 60, "No. rekening sama dipakai >1 karyawan (indikasi ghost employee)", "Ghost Employee", gn)
            if "hari_hadir" in d:
                F.add(((d["hari_hadir"] == 0) & (d["gaji_bersih"] > 0)).values, 60, "Digaji penuh tanpa kehadiran (indikasi ghost employee)", "Ghost Employee", gn)
        else:
            q, v = d["kuantitas"].values, d["nilai_usd"].values
            F.add((q <= 0) | (v <= 0), 40, "Kuantitas/nilai transaksi nol atau negatif", "Data Tidak Wajar")
            price = np.where(q > 0, v / np.where(q > 0, q, 1), np.nan)
            grp = d["deskripsi_barang"].values
            z, med = robust_z(price, grp)
            cnt = pd.Series(grp).map(pd.Series(grp).value_counts()).values
            ok = (cnt >= 4) & np.isfinite(z)
            gap = np.nan_to_num(np.abs(med - price) * q)
            under, over = ok & (z < -3.5), ok & (z > 3.5)
            tx = lambda w: [f"Harga satuan {p:.2f} vs median {m:.2f} ({w})" for p, m in zip(price, med)]
            F.add(under, 50, tx("indikasi UNDER-valuation"), "Under-Valuation", gap)
            F.add(over, 40, tx("indikasi OVER-valuation"), "Over-Valuation", gap)
            F.extra["under_gap"] = np.where(under, gap, 0.0)

    def enrich(self, res, mode):
        recs = {}
        top = res[res["risk_score"] > 0].nlargest(12, "risk_score")
        if self.llm.active and len(top):
            lines = "\n".join(f"{r[ID[mode]]}|{int(r['risk_score'])}|{str(r['temuan'])[:200]}" for _, r in top.iterrows())
            out = self.llm.json('Anda auditor senior. Untuk tiap temuan beri tindakan tindak lanjut singkat (maks 15 kata). '
                                'Balas HANYA JSON: {"items":[{"id":"...","tindakan":"..."}]}', "id|skor|temuan\n" + lines, max_tokens=1200)
            recs = {str(i.get("id")): i.get("tindakan") for i in out.get("items", []) if isinstance(i, dict)}
        default = "Verifikasi dokumen pendukung dan konfirmasi ke unit terkait"
        return [recs.get(str(i), default if s > 0 else "") for i, s in zip(res[ID[mode]], res["risk_score"])]


# ------------------------------------------------------------------ Agen 4
class RiskAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, res, mode, kpis):
        g, v = GRP[mode], VAL[mode]
        t = res.groupby(g).agg(nilai=(v, "sum"), jumlah=(ID[mode], "count"),
                               temuan=("risk_score", lambda x: int((x > 0).sum())), dampak=("estimasi_dampak", "sum")
                               ).sort_values("nilai", ascending=False)
        t["pangsa_%"] = (t["nilai"] / t["nilai"].sum() * 100).round(1)
        t["rasio_temuan_%"] = (t["temuan"] / t["jumlah"] * 100).round(1)
        if mode == "payroll" and "skor_kinerja" in res:
            perf = res.groupby(g)["skor_kinerja"].mean().reindex(t.index).replace(0, np.nan)
            t["biaya_per_poin_kinerja"] = (t["nilai"] / t["jumlah"] / perf).round(0)
        hhi = float(((t["pangsa_%"] / 100) ** 2).sum() * 10000)
        lvl = "Tinggi" if hhi > 2500 else "Sedang" if hhi > 1500 else "Rendah"
        info = {"hhi": round(hhi), "level_konsentrasi": lvl, "teratas": t.index[0], "pangsa_teratas_%": float(t["pangsa_%"].iloc[0])}
        if mode == "trade":
            info["pangsa_negara_%"] = (res.groupby("negara_asal")[v].sum().nlargest(3) / res[v].sum() * 100).round(1).to_dict()
        payload = json.dumps({"info": info, "kpi": kpis, "grup_teratas": t.head(5).round(1).to_dict("index")}, default=str, ensure_ascii=False)
        txt = self.llm.chat("Anda Risk & Macro Intelligence Agent. Tulis analisis risiko makro ringkas (maks 170 kata, Bahasa Indonesia, Markdown): "
                            "(1) konsentrasi, (2) dampak fiskal/keuangan & tren, (3) 3 rekomendasi bernomor.", payload, max_tokens=600, temperature=0.3)
        if not txt:
            txt = (f"- Konsentrasi {GRP[mode]}: HHI **{info['hhi']}** ({lvl}); terbesar **{info['teratas']}** ({info['pangsa_teratas_%']}%).\n"
                   f"- Total estimasi dampak: **{CUR[mode]} {kpis['dampak']:,.0f}** dari {kpis['anomali']} temuan.\n"
                   "- Rekomendasi: (1) prioritaskan temuan Tinggi, (2) diversifikasi/pengawasan pada kelompok dominan, (3) perkuat kontrol otomatis.")
        return t, info, txt

    # --- simulasi what-if
    @staticmethod
    def simulate_payroll(df, gaji_pct, tunj_pct, ump_base, ump_new):
        gp, tj, lr = df["gaji_pokok"].values, df["tunjangan"].values, df["lembur_rp"].values
        ngp = np.maximum(gp * (1 + gaji_pct / 100), ump_new)
        ntj = tj * (1 + tunj_pct / 100)
        emp = lambda g: (0.04 * np.minimum(g, 12e6) + 0.037 * g).sum()
        b0, b1 = (gp + tj + lr).sum(), (ngp + ntj + lr).sum()
        return {"bruto0": b0, "bruto1": b1, "bpjs0": emp(gp), "bpjs1": emp(ngp),
                "pph0": est_pph21_month(gp + tj + lr).sum(), "pph1": est_pph21_month(ngp + ntj + lr).sum(),
                "ump0": int((gp < ump_base).sum()), "ump1": int((ngp < ump_new).sum())}

    @staticmethod
    def simulate_trade(df, vol_pct, bm_delta, kurs, koreksi, under_gap):
        t0 = df["tarif_bm"].fillna(10).values if "tarif_bm" in df else np.full(len(df), 10.0)
        v0 = df["nilai_usd"].values
        t1, v1 = np.clip(t0 + bm_delta, 0, 100), v0 * (1 + vol_pct / 100)
        rec = float((np.asarray(under_gap) * (1 + vol_pct / 100) * t1 / 100).sum() * kurs) if koreksi else 0.0
        return {"nilai0": v0.sum(), "nilai1": v1.sum(), "bm0": (v0 * t0 / 100).sum() * kurs,
                "bm1": (v1 * t1 / 100).sum() * kurs, "recover": rec}


# ------------------------------------------------------------------ Agen 5
class ReportingAgent:
    @staticmethod
    def summary(llm, res, mode, kpis, risk_txt):
        cats = Counter(c for s in res["kategori"] for c in s.split(", ") if c)
        payload = json.dumps({"kpi": kpis, "kategori": dict(cats.most_common(6)), "risiko": risk_txt[:600]}, ensure_ascii=False, default=str)
        txt = llm.chat("Anda Executive Reporting Agent. Tulis ringkasan eksekutif audit (maks 140 kata, Bahasa Indonesia, Markdown, "
                       "tanpa heading): temuan utama, dampak, prioritas tindakan.", payload, max_tokens=450, temperature=0.3)
        return txt or (f"Dari **{kpis['total_baris']}** baris diuji ditemukan **{kpis['anomali']}** anomali ({kpis['pct']:.1f}%), "
                       f"**{kpis['tinggi']}** berisiko Tinggi, dengan estimasi dampak **{CUR[mode]} {kpis['dampak']:,.0f}**. "
                       f"Kategori dominan: {', '.join(k for k, _ in cats.most_common(3)) or '-'}. (Ringkasan otomatis tanpa AI.)")

    @staticmethod
    def context(res, mode, kpis, table):
        cols = [ID[mode], GRP[mode], "risk_score", "risk_level", "temuan"]
        top = res[res["risk_score"] > 0].nlargest(15, "risk_score")[list(dict.fromkeys(cols))].copy()
        top["temuan"] = top["temuan"].str.slice(0, 160)
        return (f"Mode: {mode}\nKolom data: {list(res.columns)}\nKPI: {json.dumps(kpis, default=str)}\n"
                f"Agregat per {GRP[mode]}:\n{md(table.head(10).round(1).reset_index())}\nTemuan teratas:\n{md(top)}")

    @staticmethod
    def chat(llm, question, r, history):
        if not llm.active:
            return "⚠️ AI tidak aktif. Isi GROQ_API_KEY di sidebar untuk memakai Copilot."
        extra = ""
        hits = [i for i in r["results"][r["idcol"]].astype(str) if i in question][:5]
        if hits:
            rows = r["results"][r["results"][r["idcol"]].astype(str).isin(hits)].astype(str).to_dict("records")
            extra = "\nRecord yang disebut:\n" + json.dumps(rows, ensure_ascii=False)[:2500]
        hist = [{"role": m["role"], "content": m["content"][:600]} for m in history[-6:]]
        out = llm.chat("Anda AI Audit Copilot. Jawab berdasarkan KONTEKS (ringkasan & sampel, bukan seluruh data). "
                       "Bila informasi tidak ada, jelaskan keterbatasannya. Singkat, Bahasa Indonesia, Markdown.",
                       f"KONTEKS:\n{r['context']}{extra}\n\nPERTANYAAN: {question}", max_tokens=700, temperature=0.2, history=hist)
        return out or "⚠️ Gagal mendapat respons: " + "; ".join(llm.errors[-2:])

    @staticmethod
    def report_md(res, mode, kpis, srep, risk_txt, summary, table):
        c = CUR[mode]
        top = res[res["risk_score"] > 0].nlargest(15, "risk_score")
        top = top[[ID[mode], GRP[mode], "risk_score", "risk_level", "temuan", "rekomendasi_ai"]].copy()
        top["temuan"] = top["temuan"].str.slice(0, 220)
        title = "SDM & Payroll" if mode == "payroll" else "Ekspor-Impor & Kepabeanan"
        return f"""# Kertas Kerja Audit - {title}
_Dibuat: {datetime.now():%d %B %Y %H:%M} | Sistem: Integrated Multi-Agent HR, Payroll & Trade Governance Simulation_

## 1. Ringkasan Eksekutif
{summary}

## 2. Lingkup & Kualitas Data (Agen Sanitizer)
- Baris awal: {srep['baris_awal']} | Baris diuji: {srep['baris_akhir']}
- Baris kosong dihapus: {srep['baris_kosong_dihapus']} | Data kritis kosong: {srep['baris_data_kritis_kosong']} | Duplikat: {srep['duplikat_dihapus']}
- Pemetaan kolom oleh AI: {srep['pemetaan_kolom_ai'] or '-'}

## 3. KPI Pengujian
| Indikator | Nilai |
|---|---|
| Total baris diuji | {kpis['total_baris']} |
| Anomali ditemukan | {kpis['anomali']} ({kpis['pct']:.1f}%) |
| Risiko Tinggi / Sedang / Rendah | {kpis['tinggi']} / {kpis['sedang']} / {kpis['rendah']} |
| Estimasi dampak finansial | {c} {kpis['dampak']:,.0f} |

## 4. Temuan Prioritas (Top 15)
{md(top)}

## 5. Analisis Risiko & Konsentrasi (Agen Risk & Macro)
{risk_txt}

{md(table.head(10).round(1).reset_index())}

## 6. Keterbatasan
Estimasi PPh21 bersifat pendekatan (bukan perhitungan TER/PTKP individual). Temuan adalah indikasi awal dan wajib diverifikasi dokumen sumber sebelum disimpulkan.
"""


# ------------------------------------------------------------------ Orkestrasi
def run_pipeline(df_raw, mode, llm, params=None, progress=None):
    params, p = params or {}, progress or (lambda f, t: None)
    p(.05, "🧹 Agen 1 - Data Sanitizer & Ingestion...")
    clean, srep = SanitizerAgent(llm).run(df_raw, mode)
    if srep["kolom_hilang"]:
        raise ValueError(f"Kolom wajib tidak ditemukan: {srep['kolom_hilang']}")
    if clean.empty:
        raise ValueError("Tidak ada baris valid setelah pembersihan data.")
    F = Findings(len(clean))
    p(.25, "⚖️ Agen 2 - Compliance & Classification...")
    ComplianceAgent(llm).run(clean, mode, F, params)
    p(.45, "🔎 Agen 3 - Anomaly & Valuation Audit...")
    anom = AnomalyAgent(llm)
    anom.run(clean, mode, F)
    res = clean.copy()
    sc = np.minimum(F.score, 100)
    res["risk_score"] = sc.round(0)
    res["risk_level"] = np.select([sc >= 60, sc >= 30, sc > 0], ["Tinggi", "Sedang", "Rendah"], "Normal")
    res["temuan"] = ["; ".join(x) for x in F.reasons]
    res["kategori"] = [", ".join(x) for x in F.cats]
    res["estimasi_dampak"] = F.impact.round(2)
    res["selisih_undervaluasi"] = F.extra.get("under_gap", np.zeros(len(res)))
    res["rekomendasi_ai"] = anom.enrich(res, mode)
    n_flag = int((sc > 0).sum())
    kpis = {"total_baris": len(res), "anomali": n_flag, "pct": n_flag / len(res) * 100,
            "tinggi": int((res.risk_level == "Tinggi").sum()), "sedang": int((res.risk_level == "Sedang").sum()),
            "rendah": int((res.risk_level == "Rendah").sum()), "dampak": float(res["estimasi_dampak"].sum()),
            "nilai_total": float(res[VAL[mode]].sum()), "mata_uang": CUR[mode]}
    p(.7, "🌐 Agen 4 - Risk & Macro Intelligence...")
    table, info, risk_txt = RiskAgent(llm).run(res, mode, kpis)
    p(.88, "📝 Agen 5 - Executive Reporting & Copilot...")
    summary = ReportingAgent.summary(llm, res, mode, kpis, risk_txt)
    report = ReportingAgent.report_md(res, mode, kpis, srep, risk_txt, summary, table)
    ctx = ReportingAgent.context(res, mode, kpis, table)
    p(1.0, "Selesai")
    return {"results": res, "kpis": kpis, "sanitizer": srep, "risk_table": table, "risk_info": info, "risk_md": risk_txt,
            "summary": summary, "report_md": report, "context": ctx, "idcol": ID[mode], "grpcol": GRP[mode],
            "llm_errors": list(dict.fromkeys(llm.errors)), "llm_calls": llm.calls}
