import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Prediksi Kebangkrutan Perusahaan",
    page_icon="📊",
    layout="wide"
)

# ===============================
# LOAD MODEL & DATA
# ===============================
model = joblib.load("decision_tree_bankruptcy.joblib")
scaler = joblib.load("scaler.joblib")

df = pd.read_csv("data.csv")
X = df.drop("Bankrupt?", axis=1)

feature_names = X.columns.tolist()
mean_values = X.mean()

# Contoh data dari dataset
contoh_bangkrut = X[df["Bankrupt?"] == 1].iloc[0]
contoh_tidak_bangkrut = X[df["Bankrupt?"] == 0].iloc[0]

# ===============================
# HEADER
# ===============================
st.image(
    "https://images.unsplash.com/photo-1520607162513-77705c0f0d4a",
    use_container_width=True
)

st.title("📊 Prediksi Kebangkrutan Perusahaan")

st.markdown("""
Aplikasi ini memprediksi **potensi kebangkrutan perusahaan**
berdasarkan **95 indikator keuangan** menggunakan **Decision Tree**.

💡 *Nilai contoh diambil langsung dari dataset untuk mempermudah demonstrasi.*
""")

st.divider()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("ℹ️ Informasi Aplikasi")
st.sidebar.write("""
- Model : Decision Tree  
- Fitur : 95 indikator keuangan  
- Dataset : Laporan keuangan perusahaan  
- Tujuan : Prediksi risiko kebangkrutan
""")

# ===============================
# MODE INPUT
# ===============================
st.subheader("🧪 Mode Input Data")

mode = st.radio(
    "Pilih skenario pengujian:",
    (
        "📉 Contoh Perusahaan Berisiko Bangkrut",
        "📈 Contoh Perusahaan Tidak Bangkrut",
        "✍️ Input Manual (Lengkap 95 Indikator)"
    )
)

# ===============================
# GROUPING FEATURES
# ===============================
groups = {
    "📈 Profitabilitas": feature_names[0:15],
    "💧 Likuiditas": feature_names[15:30],
    "🏦 Struktur Utang": feature_names[30:50],
    "⚙️ Efisiensi Operasional": feature_names[50:70],
    "💰 Arus Kas & Stabilitas": feature_names[70:95]
}

# ===============================
# INPUT DATA LOGIC
# ===============================
if mode == "📉 Contoh Perusahaan Berisiko Bangkrut":
    input_data = contoh_bangkrut.copy()

    st.warning("Menggunakan contoh perusahaan dengan kondisi keuangan BURUK dari dataset.")

    for group_name, features in groups.items():
        with st.expander(group_name):
            st.dataframe(
                input_data[features].to_frame(name="Nilai Indikator"),
                use_container_width=True
            )

elif mode == "📈 Contoh Perusahaan Tidak Bangkrut":
    input_data = contoh_tidak_bangkrut.copy()

    st.success("Menggunakan contoh perusahaan dengan kondisi keuangan SEHAT dari dataset.")

    for group_name, features in groups.items():
        with st.expander(group_name):
            st.dataframe(
                input_data[features].to_frame(name="Nilai Indikator"),
                use_container_width=True
            )

else:
    input_data = mean_values.copy()

    st.info("Silakan masukkan indikator keuangan atau gunakan nilai default.")

    for group_name, features in groups.items():
        with st.expander(group_name):
            cols = st.columns(3)
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    input_data[feature] = st.number_input(
                        label=feature,
                        value=float(mean_values[feature]),
                        help="Rasio keuangan berdasarkan laporan perusahaan"
                    )

# ===============================
# PREDICTION
# ===============================
st.divider()
st.subheader("🔍 Hasil Prediksi")

if st.button("🚀 Prediksi Status Perusahaan", use_container_width=True):
    scaled_input = scaler.transform([input_data.values])
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ **Perusahaan diprediksi BERISIKO BANGKRUT**")
    else:
        st.success("✅ **Perusahaan diprediksi TIDAK bangkrut**")

    st.info("""
    ⚠️ **Disclaimer Akademik**  
    Hasil prediksi ini merupakan keluaran model Machine Learning dan
    tidak dapat dijadikan keputusan mutlak dalam dunia nyata.
    """)

# ===============================
# FOOTER
# ===============================
st.caption("📘 Project Data Mining | Decision Tree + Streamlit Deployment")