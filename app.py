import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(
    page_title="Smart Stock Decision System",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Smart Stock Decision System")
st.markdown("""
Aplikasi ini membantu Manajer Produk mengambil keputusan stok berdasarkan **Prediksi AI**.
Alur: `Input Spesifikasi` -> `Auto-Segmentasi (K-Means)` -> `Prediksi Permintaan (XGBoost)` -> `Keputusan`.
""")
st.markdown("---")

# ==========================================
# 2. LOAD SEMUA "OTAK" (MODEL)
# ==========================================
@st.cache_resource
def load_brains():
    """
    Memuat model Scaler, K-Means, dan XGBoost Pipeline.
    """
    try:
        models = {}
        # [cite_start]Model Clustering [cite: 7]
        models['kmeans'] = joblib.load('kmeans_model.pkl')
        models['scaler'] = joblib.load('scaler_clustering.pkl')
        
        # [cite_start]Model Regresi (Pipeline) [cite: 32]
        models['xgboost'] = joblib.load('xgboost_model.pkl')
        return models, None
    except FileNotFoundError as e:
        return None, str(e)

# Muat model
brains, error_msg = load_brains()

if error_msg:
    st.error(f"❌ **System Error:** File model tidak ditemukan.\n\nDetail: `{error_msg}`")
    st.stop()

# Mapping Nama Cluster (Untuk Labeling)
CLUSTER_NAMES = {
    0: 'Reliable Daily Drivers (Standard)',
    1: 'Silent Newcomers (Produk Baru/Stagnan)',
    2: 'The Popular One (Laris)',
    3: 'High-End Luxury (Premium)',
    4: 'The Legends (Viral)'
}

# ==========================================
# 3. INPUT DATA (SIDEBAR)
# ==========================================
st.sidebar.header("📝 Spesifikasi Produk Baru")

with st.sidebar.form("product_form"):
    product_name = st.text_input("Nama Produk", value="Super Glow Serum Viral")
    
    # Input Fitur Utama
    category = st.selectbox("Kategori", ['Toner', 'Serum', 'Moisturizer', 'Facial Wash', 'Mask', 'Sunscreen'])
    price = st.number_input("Harga (Price Clean)", min_value=10000, value=120000, step=5000)
    rating = st.slider("Target Rating", 1.0, 5.0, 4.7)
    
    st.markdown("---")
    st.markdown("**Estimasi Pasar (Untuk Clustering)**")
    st.caption("Nilai di bawah digunakan untuk mendeteksi segmen pasar. Untuk produk baru, gunakan estimasi dari produk sejenis/kompetitor.")
    
    reviews = st.number_input("Estimasi Total Reviews", min_value=0, value=50)
    # Ini input untuk clustering saja, bukan target regresi
    repurchase_est = st.number_input("Estimasi Repurchase (Historis)", min_value=0, value=2000) 
    points = st.number_input("Beauty Points Earned", min_value=0, value=20)
    
    submit_btn = st.form_submit_button("🔍 Analisis & Prediksi Stok")

# ==========================================
# 4. LOGIKA UTAMA (RUN ON CLICK)
# ==========================================
if submit_btn:
    # Tampilkan Data Input
    st.subheader(f"Analisis Produk: {product_name}")
    
    col_proc1, col_proc2 = st.columns(2)
    
    # --- TAHAP A: CLUSTERING (MENCARI SEGMEN) ---
    with col_proc1:
        st.info("🔄 Tahap 1: Auto-Segmentation (K-Means)")
        
        # 1. Siapkan fitur sesuai urutan notebook clustering
        features_cluster = np.array([[price, rating, reviews, repurchase_est, points]])
        
        # 2. Scaling (Wajib sama dengan training)
        features_scaled = brains['scaler'].transform(features_cluster)
        
        # 3. Prediksi Cluster
        cluster_id = brains['kmeans'].predict(features_scaled)[0]
        cluster_label = CLUSTER_NAMES.get(cluster_id, 'Unknown')
        
        st.success(f"**Hasil Segmentasi:**\n\n### Cluster {cluster_id}\n**({cluster_label})**")
        st.caption("Produk diklasifikasikan berdasarkan kemiripan Harga, Rating, dan Popularitas.")

    # --- TAHAP B: REGRESI (MEMPREDIKSI JUMLAH) ---
    with col_proc2:
        st.info("📈 Tahap 2: Prediction Engine (XGBoost)")
        
        # 1. Siapkan DataFrame untuk regresi
        # Memasukkan 'Cluster' hasil dari Tahap A sebagai fitur input
        df_reg = pd.DataFrame([{
            "price_clean": price,
            "average_rating": rating,
            "default_category": category,
            "Cluster": cluster_id  # <--- INTEGRASI MODEL DI SINI
        }])
        
        # 2. Prediksi Jumlah Repurchase
        # Pipeline akan otomatis mengurus OneHotEncoding kategori
        predicted_raw = brains['xgboost'].predict(df_reg)[0]
        predicted_qty = int(max(0, predicted_raw)) # Hindari nilai negatif
        
        st.success(f"**Prediksi Pembelian Ulang:**\n\n### {predicted_qty:,} User\n**(Per Bulan)**")
        st.caption("Dihitung berdasarkan Kategori, Harga, dan Cluster produk.")

    # --- TAHAP C: KEPUTUSAN BISNIS (BUSINESS LOGIC) ---
    st.markdown("---")
    st.header("📋 Keputusan Bisnis (Rekomendasi AI)")
    
    # Logika If-Else sesuai script Anda
    if predicted_qty >= 5000:
        st.error("🚀 **KEPUTUSAN: STOK BESAR (PRIORITAS TINGGI)**")
        st.write("**Alasan:** Produk ini diprediksi sangat laku (Viral/Legend). Potensi Stock-out tinggi jika suplai kurang.")
        st.metric("Saran Order Awal", f"{predicted_qty + 1000} Pcs", delta="Aggressive Stock")
        
    elif 1000 <= predicted_qty < 5000:
        st.success("✅ **KEPUTUSAN: STOK NORMAL**")
        st.write("**Alasan:** Performa penjualan sehat dan stabil. Masuk kategori aman.")
        st.metric("Saran Order Awal", f"{int(predicted_qty * 1.2)} Pcs", delta="Safe Stock")
        
    elif 200 <= predicted_qty < 1000:
        st.warning("⚠️ **KEPUTUSAN: STOK HATI-HATI (TEST MARKET)**")
        st.write("**Alasan:** Permintaan ada, tapi belum masif. Jangan terlalu agresif.")
        st.metric("Saran Order Awal", f"{predicted_qty} Pcs", delta="Conservative")
        
    else:
        st.info("🛑 **KEPUTUSAN: JANGAN STOK / PRE-ORDER SAJA**")
        st.write("**Alasan:** Prediksi permintaan sangat rendah. Risiko barang mati (Dead Stock).")
        st.metric("Saran Order Awal", "0 - 50 Pcs", delta="Minimize Risk", delta_color="inverse")

else:
    # Tampilan Awal (Kosong)
    st.info("👈 Silakan masukkan data produk di sidebar dan klik tombol **Analisis**.")