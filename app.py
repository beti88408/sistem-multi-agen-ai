"""app.py - Streamlit UI: Integrated Multi-Agent HR, Payroll & Trade Governance Simulation System."""
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import agents as ag
from sample_data import generate_payroll, generate_trade

st.set_page_config(page_title="Multi-Agent HR, Payroll & Trade Governance", page_icon="🛡️", layout="wide")
MODES = {"Tata Kelola SDM & Payroll Simulation": "payroll", "Audit Data Ekspor-Impor & Kepabeanan": "trade"}
COLORS = {"Tinggi": "#d62728", "Sedang": "#ff7f0e", "Rendah": "#f2c744", "Normal": "#2ca02c"}


def money(v, cur):
    return f"{cur} {v:,.0f}".replace(",", ".") if cur == "Rp" else f"{cur} {v:,.0f}"


def secret_key():
    try:
        k = st.secrets["GROQ_API_KEY"]
        return "" if "ISI_" in k else k
    except Exception:
        return ""


def load_file(f):
    if f.name.lower().endswith(".csv"):
        f.seek(0)
        return pd.read_csv(f, sep=None, engine="python")
    return pd.read_excel(f)


# ------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    key_in = st.text_input("GROQ_API_KEY", type="password", help="Kosongkan untuk memakai st.secrets['GROQ_API_KEY']")
    api_key = key_in or secret_key()
    st.caption("🔑 API key terdeteksi" if api_key else "⚠️ API key belum diisi - agen AI nonaktif (mode deterministik)")
    model = st.selectbox("Model Groq", ag.MODELS)
    mode = MODES[st.radio("Mode Pengujian", list(MODES))]
    ump = st.number_input("UMP/UMK acuan (Rp)", value=3_500_000, step=50_000) if mode == "payroll" else None
    st.divider()
    up = st.file_uploader("Unggah data (.csv / .xlsx)", type=["csv", "xlsx"])
    sample_btn = st.button("🧪 Gunakan Data Sampel Bawaan")
    run_btn = st.button("🚀 Jalankan Analisis Multi-Agen", type="primary")

ss = st.session_state
if ss.get("mode") != mode:
    ss.update(mode=mode, df_raw=None, result=None, chat=[], source="", upload_sig=None)
if up is not None and ss.get("upload_sig") != (up.name, up.size):
    try:
        ss.df_raw, ss.source, ss.upload_sig, ss.result = load_file(up), up.name, (up.name, up.size), None
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")
if sample_btn:
    ss.df_raw = generate_payroll() if mode == "payroll" else generate_trade()
    ss.source, ss.result, ss.chat = "Data sampel bawaan", None, []

if run_btn:
    if ss.get("df_raw") is None:
        st.sidebar.warning("Unggah file atau pilih data sampel dulu.")
    else:
        llm = ag.LLM(api_key, model)
        bar = st.progress(0.0, text="Memulai...")
        try:
            ss.result = ag.run_pipeline(ss.df_raw, mode, llm, {"ump": ump}, lambda f, t: bar.progress(f, text=t))
            ss.chat = []
        except Exception as e:
            st.error(f"Analisis gagal: {e}")
        finally:
            bar.empty()

# ------------------------------------------------------------ Main
st.title("🛡️ Integrated Multi-Agent HR, Payroll & Trade Governance Simulation System")
r = ss.get("result")
if r is None:
    st.info("Pilih mode, unggah data atau klik **Gunakan Data Sampel Bawaan**, lalu klik **Jalankan Analisis Multi-Agen**.")
    if ss.get("df_raw") is not None:
        st.subheader(f"Pratinjau data: {ss.source} ({len(ss.df_raw)} baris)")
        st.dataframe(ss.df_raw.head(50))
    st.stop()

res, k = r["results"], r["kpis"]
cur = k["mata_uang"]
tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan Eksekutif & KPI", "🔍 Hasil Audit & Anomali", "🎛️ Simulasi What-If", "🤖 AI Copilot & Export"])

with tab1:
    c = st.columns(5)
    c[0].metric("Baris diuji", f"{k['total_baris']:,}")
    c[1].metric("Anomali ditemukan", k["anomali"], f"{k['pct']:.1f}%", delta_color="off")
    c[2].metric("Risiko Tinggi", k["tinggi"])
    c[3].metric("Estimasi dampak", money(k["dampak"], cur))
    c[4].metric("Total nilai diuji", money(k["nilai_total"], cur))
    a, b = st.columns(2)
    lv = res["risk_level"].value_counts().reindex(list(COLORS)).dropna().reset_index()
    lv.columns = ["Tingkat", "Jumlah"]
    a.plotly_chart(px.pie(lv, names="Tingkat", values="Jumlah", hole=.45, color="Tingkat", color_discrete_map=COLORS,
                          title="Distribusi Tingkat Risiko"))
    imp = res.groupby(r["grpcol"])["estimasi_dampak"].sum().nlargest(10).reset_index()
    b.plotly_chart(px.bar(imp, x="estimasi_dampak", y=r["grpcol"], orientation="h", title=f"Estimasi Dampak per {r['grpcol']} (Top 10)"))
    a, b = st.columns(2)
    cats = Counter(x for s in res["kategori"] for x in s.split(", ") if x)
    if cats:
        cd = pd.DataFrame(cats.most_common(), columns=["Kategori", "Jumlah"])
        a.plotly_chart(px.bar(cd, x="Kategori", y="Jumlah", title="Temuan per Kategori"))
    b.plotly_chart(px.histogram(res[res.risk_score > 0], x="risk_score", nbins=20, title="Sebaran Skor Risiko (baris beranomali)"))
    st.subheader("Ringkasan Eksekutif")
    st.markdown(r["summary"])
    with st.expander("🌐 Analisis Risiko & Konsentrasi (Agen 4)"):
        st.markdown(r["risk_md"])
        st.dataframe(r["risk_table"])
    with st.expander("🧹 Laporan Data Sanitizer (Agen 1)"):
        st.json(r["sanitizer"])
    if r["llm_errors"]:
        st.warning("Catatan LLM: " + " | ".join(r["llm_errors"]))
    elif r["llm_calls"] == 0:
        st.caption("ℹ️ Berjalan tanpa Groq (API key kosong): narasi & rekomendasi memakai templat otomatis.")

with tab2:
    f1, f2 = st.columns([2, 3])
    levels = f1.multiselect("Tingkat risiko", list(COLORS), default=["Tinggi", "Sedang", "Rendah"])
    q = f2.text_input("Cari (ID / nama / alasan temuan)")
    df = res[res["risk_level"].isin(levels)]
    if q:
        df = df[df.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False)).any(axis=1)]
    df = df.sort_values("risk_score", ascending=False)
    st.caption(f"{len(df)} baris ditampilkan")
    st.dataframe(df, height=520, column_config={
        "risk_score": st.column_config.ProgressColumn("Skor Risiko", min_value=0, max_value=100, format="%d")})
    st.download_button("⬇️ Unduh tampilan terfilter (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "hasil_audit_filter.csv", "text/csv")

with tab3:
    st.subheader("Simulasi Skenario (What-If)")
    if mode == "payroll":
        s1, s2, s3 = st.columns(3)
        gpct = s1.slider("Kenaikan gaji pokok (%)", 0, 30, 5)
        tpct = s2.slider("Perubahan tunjangan (%)", -20, 30, 0)
        unew = s3.slider("UMP baru (Rp)", 3_000_000, 6_000_000, int(ump), step=50_000)
        s = ag.RiskAgent.simulate_payroll(res, gpct, tpct, ump, unew)
        t0, t1 = s["bruto0"] + s["bpjs0"], s["bruto1"] + s["bpjs1"]
        m = st.columns(4)
        m[0].metric("Total beban SDM/bulan", money(t1, "Rp"), money(t1 - t0, "Rp"), delta_color="inverse")
        m[1].metric("PPh21 (estimasi)", money(s["pph1"], "Rp"), money(s["pph1"] - s["pph0"], "Rp"), delta_color="off")
        m[2].metric("BPJS pemberi kerja", money(s["bpjs1"], "Rp"), money(s["bpjs1"] - s["bpjs0"], "Rp"), delta_color="inverse")
        m[3].metric("Karyawan < UMP", s["ump1"], s["ump1"] - s["ump0"], delta_color="inverse")
        st.caption("Gaji pokok otomatis dinaikkan minimal ke UMP baru. BPJS pemberi kerja diasumsikan 4% Kesehatan (cap Rp12 jt) + 3,7% JHT.")
        fig = go.Figure([go.Bar(name="Baseline", x=["Beban SDM", "PPh21", "BPJS pemberi"], y=[t0, s["pph0"], s["bpjs0"]]),
                         go.Bar(name="Skenario", x=["Beban SDM", "PPh21", "BPJS pemberi"], y=[t1, s["pph1"], s["bpjs1"]])])
        st.plotly_chart(fig.update_layout(barmode="group", title="Baseline vs Skenario (Rp/bulan)"))
    else:
        s1, s2, s3 = st.columns(3)
        vol = s1.slider("Perubahan volume transaksi (%)", -50, 100, 10)
        bm = s2.slider("Perubahan tarif Bea Masuk (poin %)", -10, 10, 0)
        kurs = s3.slider("Kurs USD/IDR", 12000, 20000, 16000, step=100)
        fix = st.checkbox("Simulasikan koreksi under-valuation (penerimaan yang dapat dipulihkan)", value=True)
        s = ag.RiskAgent.simulate_trade(res, vol, bm, kurs, fix, res["selisih_undervaluasi"].values)
        m = st.columns(3)
        m[0].metric("Nilai impor", money(s["nilai1"], "USD"), money(s["nilai1"] - s["nilai0"], "USD"))
        m[1].metric("Estimasi Bea Masuk", money(s["bm1"], "Rp"), money(s["bm1"] - s["bm0"], "Rp"))
        m[2].metric("Potensi pemulihan (under-valuation)", money(s["recover"], "Rp"))
        fig = go.Figure(go.Bar(x=["Baseline", "Skenario", "Skenario + Koreksi"], y=[s["bm0"], s["bm1"], s["bm1"] + s["recover"]]))
        st.plotly_chart(fig.update_layout(title="Estimasi Penerimaan Bea Masuk (Rp)"))
        st.caption("Tarif BM diambil dari kolom tarif_bm (default 10% bila kosong).")

with tab4:
    st.subheader("💬 AI Audit Copilot")
    ss.setdefault("chat", [])
    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if prompt := st.chat_input("Tanyakan sesuatu tentang data, mis. 'Departemen mana paling berisiko?'"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Copilot menganalisis..."):
                ans = ag.ReportingAgent.chat(ag.LLM(api_key, model), prompt, r, ss.chat)
            st.markdown(ans)
        ss.chat += [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}]
    st.divider()
    st.subheader("📥 Export Laporan")
    e = st.columns(3)
    e[0].download_button("⬇️ Kertas Kerja (Markdown)", r["report_md"], "kertas_kerja_audit.md", "text/markdown")
    e[1].download_button("⬇️ Kertas Kerja (TXT)", r["report_md"], "kertas_kerja_audit.txt", "text/plain")
    e[2].download_button("⬇️ Seluruh hasil (CSV)", res.to_csv(index=False).encode("utf-8-sig"), "hasil_audit_lengkap.csv", "text/csv")
    with st.expander("Pratinjau laporan"):
        st.markdown(r["report_md"])
