"""sample_data.py - Generator data sampel realistis (dengan anomali tersuntik) untuk demo."""
import numpy as np
import pandas as pd
from agents import est_pph21_month

FIRST = ["Budi", "Siti", "Agus", "Dewi", "Rudi", "Ani", "Joko", "Rina", "Eko", "Maya", "Hendra", "Lestari", "Fajar", "Putri", "Andi"]
LAST = ["Santoso", "Wijaya", "Pratama", "Hidayat", "Saputra", "Lubis", "Nasution", "Siregar", "Harahap", "Utami", "Kusuma", "Ramadhan"]


def generate_payroll(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    depts = {"Keuangan": (7e6, 15e6), "SDM": (6e6, 13e6), "Produksi": (4e6, 9e6),
             "Logistik": (4e6, 8e6), "IT": (8e6, 20e6), "Pemasaran": (6e6, 14e6)}
    dn = rng.choice(list(depts), n, p=[.12, .1, .3, .18, .12, .18])
    lo = np.array([depts[x][0] for x in dn]); hi = np.array([depts[x][1] for x in dn])
    gp = np.round(rng.uniform(lo, hi), -3)
    tj = np.round(gp * rng.uniform(.1, .35, n), -3)
    jam = rng.choice([0, 0, 0, 4, 8, 12, 16, 20, 24, 30], n).astype(float)
    p = rng.permutation(n)
    sl = lambda a, b: p[a:b]
    gp[sl(0, 4)] = np.round(rng.uniform(2.4e6, 3.3e6, 4), -3)          # di bawah UMP
    gp[sl(4, 7)] = np.round(rng.uniform(55e6, 70e6, 3), -3)            # outlier gaji
    tj[sl(7, 11)] = np.round(gp[sl(7, 11)] * 1.3, -3)                  # tunjangan > gaji pokok
    jam[sl(11, 17)] = rng.integers(90, 141, 6)                         # lembur berlebih
    jam[sl(17, 21)] = rng.choice([8, 12, 16, 20], 4)
    lr = np.round(gp / 173 * jam * 1.75, -2)
    bk = np.round(np.minimum(gp, 12e6) * .01); bj = np.round(gp * .02)
    pph = np.round(est_pph21_month(gp + tj + lr))
    lr[sl(17, 21)] *= 3                                                # upah lembur kemahalan
    pph[sl(21, 29)] = np.round(np.where(rng.random(8) < .5, pph[sl(21, 29)] * .15, pph[sl(21, 29)] * 2.2))
    bk[sl(29, 34)] = 0; bj[sl(34, 38)] = np.round(bj[sl(34, 38)] * .4)
    gn = gp + tj + lr - bk - bj - pph
    gn[sl(38, 42)] += rng.integers(500_000, 2_000_000, 4)              # gaji bersih tidak cocok
    hadir = rng.integers(20, 23, n).astype(float)
    kin = np.clip(rng.normal(78, 9, n), 40, 100)
    ghost = sl(42, 48); hadir[ghost] = 0; kin[ghost] = rng.uniform(20, 45, 6)
    rek = np.array([f"BCA-{x:010d}" for x in rng.integers(10**9, 10**10, n, dtype=np.int64)])
    others = np.setdiff1d(np.arange(n), ghost)
    rek[ghost[:3]] = rek[rng.choice(others, 3, replace=False)]         # rekening ganda
    masuk = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 3800, n), unit="D")
    return pd.DataFrame({
        "employee_id": [f"EMP{i + 1:04d}" for i in range(n)],
        "nama": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)],
        "departemen": dn, "tanggal_masuk": masuk.strftime("%Y-%m-%d"),
        "gaji_pokok": gp, "tunjangan": tj, "lembur_jam": jam, "lembur_rp": lr,
        "bpjs_kesehatan": bk, "bpjs_jht": bj, "pph21": pph, "gaji_bersih": gn,
        "no_rekening": rek, "hari_hadir": hadir, "skor_kinerja": kin.round(1)})


def generate_trade(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prods = [("Kopi arabika biji mentah", "09011110", 4.5, "kg", 5), ("Beras putih curah", "10063099", .6, "kg", 0),
             ("Sepatu olahraga kulit", "64039990", 22, "pasang", 20), ("Laptop notebook 14 inci", "84713010", 520, "unit", 0),
             ("Telepon seluler smartphone", "85171300", 210, "unit", 0), ("Kain tenun katun", "52083100", 3.2, "meter", 10),
             ("Minyak sawit mentah CPO", "15111000", .95, "kg", 5), ("Karet alam SIR 20", "40012210", 1.6, "kg", 5),
             ("Mesin pompa air sentrifugal", "84137010", 340, "unit", 5), ("Obat tablet paracetamol", "30049099", 1.1, "boks", 5),
             ("Kursi kayu jati", "94016900", 85, "unit", 15), ("Baterai lithium-ion", "85076000", 7.5, "unit", 10)]
    sup = [("Shenzhen Hightech Co", "China"), ("Hanoi Export JSC", "Vietnam"), ("Osaka Machinery Ltd", "Jepang"),
           ("Bangkok Agro Co", "Thailand"), ("Munich Industrie GmbH", "Jerman"), ("Mumbai Textile Pvt", "India"),
           ("Penang Resources Bhd", "Malaysia"), ("Busan Electronics", "Korea Selatan")]
    si = rng.choice(len(sup), n, p=[.35, .15, .12, .1, .1, .08, .05, .05])
    pi = rng.integers(0, len(prods), n)
    price = np.array([prods[i][2] for i in pi]) * rng.lognormal(0, .08, n)
    big = np.array([prods[i][3] in ("kg", "meter") for i in pi])
    qty = np.where(big, rng.integers(500, 20000, n), rng.integers(20, 800, n)).astype(float)
    hs = np.array([prods[i][1] for i in pi], dtype=object)
    val = qty * price
    p = rng.permutation(n)
    val[p[:15]] = qty[p[:15]] * price[p[:15]] * rng.uniform(.2, .4, 15)       # under-valuation
    val[p[15:24]] = qty[p[15:24]] * price[p[15:24]] * rng.uniform(3, 4, 9)    # over-valuation
    for j in p[24:42]:                                                        # salah klasifikasi HS
        hs[j] = prods[(pi[j] + rng.integers(3, 9)) % len(prods)][1]
    tanggal = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    return pd.DataFrame({
        "id_transaksi": [f"TRX-{i + 1:04d}" for i in range(n)], "tanggal": tanggal.strftime("%Y-%m-%d"),
        "pemasok": [sup[i][0] for i in si], "negara_asal": [sup[i][1] for i in si],
        "deskripsi_barang": [prods[i][0] for i in pi], "kode_hs": hs, "kuantitas": qty,
        "satuan": [prods[i][3] for i in pi], "nilai_usd": val.round(2),
        "tarif_bm": [prods[i][4] for i in pi]})
