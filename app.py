import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Prediksi Durasi Pengeringan Kayu",
    page_icon="🪵",
    layout="centered"
)

TARGET_DURASI_HARI = 14  # target durasi standar perusahaan

@st.cache_resource
def load_model():
    pipeline = joblib.load('model_kiln.joblib')
    with open('feature_order.json') as f:
        feature_order = json.load(f)
    with open('metadata.json') as f:
        metadata = json.load(f)
    return pipeline, feature_order, metadata

pipeline, FEATURE_ORDER, METADATA = load_model()
MAE_MODEL = METADATA['evaluation']['mae_test']

st.title("🪵 Prediksi Durasi Pengeringan Kayu Kiln")
st.markdown("Isi form di bawah, lalu klik **Hitung Prediksi**.")

TEBAL_BINS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]

BATAS = {
    'vol_total_m3'   : (0.0, 15.0),
    'total_lembar'   : (0, 1500),
    'curah_hujan_mm' : (0.0, 40.0),
    'suhu_maks_c'    : (25.0, 35.0),
    'suhu_min_c'     : (15.0, 26.0),
}

with st.form("input_form"):
    st.subheader("📅 Jadwal")
    tanggal_mulai = st.date_input(
        "Tanggal Mulai Pengeringan",
        value=pd.Timestamp.now(),
        help="Tanggal batch kayu mulai dimasukkan ke kiln."
    )

    st.subheader("📋 Informasi Batch")
    col1, col2 = st.columns(2)
    with col1:
        jenis_kayu = st.selectbox("Jenis Kayu", ["MAHONI", "JATI"])
        no_kiln    = st.selectbox("Nomor Kiln", [str(i) for i in range(1, 7)])
        bulan_in   = st.selectbox("Bulan Masuk", list(range(1, 13)),
                        format_func=lambda x: ["Januari","Februari","Maret","April",
                            "Mei","Juni","Juli","Agustus","September",
                            "Oktober","November","Desember"][x-1])
    with col2:
        vol_total_m3 = st.number_input("Volume Total (m³)", min_value=BATAS['vol_total_m3'][0],
                          max_value=BATAS['vol_total_m3'][1], value=10.0, step=0.5,
                          help="Total volume kayu dalam batch, maksimal 15 m³.")
        total_lembar = st.number_input("Total Lembar", min_value=BATAS['total_lembar'][0],
                          max_value=BATAS['total_lembar'][1], value=500, step=10)
        jumlah_asal  = st.slider("Jumlah Asal Kayu", min_value=1, max_value=5, value=1,
                          help="Jumlah daerah asal kayu berbeda dalam satu batch.")

    st.subheader("🌤️ Kondisi Cuaca")
    col3, col4 = st.columns(2)
    with col3:
        kelembaban_pct = st.number_input("Kelembaban (%)", min_value=40.0, max_value=100.0,
                          value=75.0, step=0.5)
        curah_hujan_mm = st.number_input("Curah Hujan (mm)", min_value=BATAS['curah_hujan_mm'][0],
                          max_value=BATAS['curah_hujan_mm'][1], value=3.0, step=0.5)
    with col4:
        suhu_maks_c = st.number_input("Suhu Maksimum (°C)", min_value=BATAS['suhu_maks_c'][0],
                          max_value=BATAS['suhu_maks_c'][1], value=32.0, step=0.5)
        suhu_min_c  = st.number_input("Suhu Minimum (°C)", min_value=BATAS['suhu_min_c'][0],
                          max_value=BATAS['suhu_min_c'][1], value=24.0, step=0.5)

    st.subheader("🎯 Target Kadar Air Akhir")
    mc_akhir_ket_max = st.number_input(
        "Target MC Akhir untuk Papan Tertebal (%)", min_value=5.0, max_value=19.0,
        value=12.0, step=0.5,
        help="Batas maksimal kadar air yang diinginkan pada papan dengan ketebalan terbesar dalam batch."
    )

    st.subheader("📏 Komposisi Ketebalan Papan")
    st.caption("Isi jumlah lembar tiap ketebalan. Kosongkan (isi 0) jika tidak ada. Total harus sama dengan Total Lembar di atas.")
    cols_tebal = st.columns(len(TEBAL_BINS))
    komposisi  = {}
    for col, t in zip(cols_tebal, TEBAL_BINS):
        with col:
            n = st.number_input(f"{t} cm", min_value=0, max_value=int(BATAS['total_lembar'][1]),
                                 value=0, step=10, key=f"t_{t}")
            if n > 0:
                komposisi[t] = n

    total_input_sementara = sum(komposisi.values())
    if total_input_sementara > 0:
        st.caption(f"Total lembar terisi: **{total_input_sementara}** / {total_lembar}")

    submitted = st.form_submit_button("🔍 Hitung Prediksi", use_container_width=True, type="primary")

if submitted:
    if not komposisi:
        st.error("⚠️ Isi minimal satu ketebalan papan.")
        st.stop()
    if suhu_maks_c <= suhu_min_c:
        st.error("⚠️ Suhu maksimum harus lebih besar dari suhu minimum.")
        st.stop()

    total_komposisi = sum(komposisi.values())
    if total_komposisi != total_lembar:
        st.error(f"⚠️ Total komposisi ketebalan ({total_komposisi}) harus sama dengan Total Lembar ({total_lembar}).")
        st.stop()

    props = {t: komposisi.get(t, 0) / total_komposisi for t in TEBAL_BINS}
    tebal_arr = np.repeat(list(komposisi.keys()), list(komposisi.values())).astype(float)

    ket_mean   = float(np.mean(tebal_arr))
    ket_max    = float(np.max(tebal_arr))
    ket_min    = float(np.min(tebal_arr))
    ket_std    = float(np.std(tebal_arr)) if len(komposisi) > 1 else 0.0
    musim      = 1 if bulan_in in [11, 12, 1, 2, 3, 4] else 0
    delta_suhu = float(suhu_maks_c) - float(suhu_min_c)

    row = {
        'no_kiln'          : float(no_kiln),
        'vol_total_m3'     : float(vol_total_m3),
        'kelembaban_pct'   : float(kelembaban_pct),
        'curah_hujan_mm'   : float(curah_hujan_mm),
        'suhu_maks_c'      : float(suhu_maks_c),
        'suhu_min_c'       : float(suhu_min_c),
        'ket_mean'         : ket_mean,
        'ket_max'          : ket_max,
        'ket_min'          : ket_min,
        'ket_std'          : ket_std,
        'n_ketebalan'      : float(len(komposisi)),
        'total_lembar'     : float(total_lembar),
        'jumlah_asal'      : float(jumlah_asal),
        'vol_per_lembar'   : float(vol_total_m3) / max(float(total_lembar), 1),
        'bulan_in'         : float(bulan_in),
        'mc_akhir_ket_max' : float(mc_akhir_ket_max),
        'prop_2.0'         : props[2.0],
        'prop_2.5'         : props[2.5],
        'prop_3.0'         : props[3.0],
        'prop_3.5'         : props[3.5],
        'prop_4.0'         : props[4.0],
        'prop_5.0'         : props[5.0],
        'prop_8.0'         : props[8.0],
        'delta_suhu'       : delta_suhu,
        'musim'            : float(musim),
        'lembab_x_tebal'   : float(kelembaban_pct) * ket_max,
        'hujan_x_lembab'   : float(curah_hujan_mm) * float(kelembaban_pct),
        'tebal_x_vol'      : ket_mean * float(vol_total_m3),
        'prop_tipis'       : props[2.0] + props[2.5],
        'prop_tebal_ext'   : props[5.0] + props[8.0],
        'rasio_tebal'      : (props[5.0] + props[8.0]) / (props[2.0] + props[2.5] + 1e-6),
        'rasio_lembab_suhu': float(kelembaban_pct) / (float(suhu_maks_c) + 1e-6),
        'hujan_x_tebal'    : float(curah_hujan_mm) * ket_mean,
        'mc_akhir_x_tebal' : float(mc_akhir_ket_max) * ket_max,
        'jenis_kayu_MAHONI': 1.0 if jenis_kayu == 'MAHONI' else 0.0,
    }

    df_input = pd.DataFrame([row])
    df_input = df_input[FEATURE_ORDER]

    with st.spinner("Menghitung prediksi..."):
        try:
            durasi = round(float(pipeline.predict(df_input)[0]), 1)
        except Exception as e:
            st.error(f"Error prediksi: {e}")
            st.stop()

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    selesai = pd.Timestamp(tanggal_mulai) + pd.Timedelta(days=durasi)
    hari_id = {'Monday':'Senin','Tuesday':'Selasa','Wednesday':'Rabu','Thursday':'Kamis',
               'Friday':'Jumat','Saturday':'Sabtu','Sunday':'Minggu'}

    hari_mulai   = hari_id.get(pd.Timestamp(tanggal_mulai).strftime("%A"), "")
    hari_selesai = hari_id.get(selesai.strftime("%A"), "")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("🚀 Tanggal Mulai", f"{hari_mulai}, {pd.Timestamp(tanggal_mulai).strftime('%d %B %Y')}")
    with c2:
        st.metric("📅 Perkiraan Selesai", f"{hari_selesai}, {selesai.strftime('%d %B %Y')}")

    selisih_target = durasi - TARGET_DURASI_HARI

    c3, c4 = st.columns(2)
    with c3:
        st.metric("⏱️ Estimasi Durasi", f"{durasi} hari")
    with c4:
        st.metric(
            "🎯 Selisih dari Target",
            f"{'+' if selisih_target > 0 else ''}{selisih_target:.1f} hari",
            delta=f"{selisih_target:.1f} hari dari target {TARGET_DURASI_HARI} hari",
            delta_color="inverse"
        )

    if selisih_target > 0:
        st.warning(
            f"⚠️ Estimasi durasi melebihi target standar ({TARGET_DURASI_HARI} hari). "
            "Pertimbangkan penyesuaian jadwal produksi atau prioritas kiln."
        )
    else:
        st.success(
            f"✅ Estimasi durasi berada dalam atau di bawah target standar ({TARGET_DURASI_HARI} hari)."
        )

    durasi_min = max(0, durasi - MAE_MODEL)
    durasi_max = durasi + MAE_MODEL
    st.caption(f"Rentang estimasi (± MAE model): **{durasi_min:.1f} – {durasi_max:.1f} hari**")

    with st.expander("📋 Detail input"):
        st.dataframe(df_input.T.rename(columns={0: "Nilai"}), use_container_width=True)
