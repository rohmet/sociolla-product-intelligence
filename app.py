import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN & UTILS
# ==========================================
st.set_page_config(
    page_title="Sociolla Product Intelligence",
    page_icon="💄",
    layout="wide"
)

# Custom CSS untuk mempercantik
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Dictionary Nama Cluster (Sesuai kode Anda)
CLUSTER_NAMES = {
    0: 'Reliable Daily Drivers',
    1: 'Silent Newcomers',
    2: 'The Popular One',
    3: 'High-End Luxury',
    4: 'The Legends'
}

# ==========================================
# 2. LOAD MODELS & DATA
# ==========================================
@st.cache_resource
def load_assets():
    try:
        kmeans = joblib.load('artifacts/kmeans_model.pkl')
        scaler = joblib.load('artifacts/scaler_clustering.pkl')
        pca = joblib.load('artifacts/pca_model.pkl')
        reg_pipeline = joblib.load('artifacts/xgboost_model.pkl')
        df = pd.read_csv('data/skincare_segmented.csv')
        return kmeans, scaler, pca, reg_pipeline, df
    except Exception as e:
        st.error(f"Gagal memuat file model/data. Pastikan file .pkl dan .csv ada di folder yang sama.\nError: {e}")
        return None, None, None, None, None

kmeans, scaler, pca, reg_pipeline, df = load_assets()

if df is not None:
    # Pastikan nama cluster tersedia di dataframe untuk visualisasi
    if 'Segment_Name' not in df.columns:
        df['Segment_Name'] = df['Cluster'].map(CLUSTER_NAMES)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Sociolla_Logo.png/640px-Sociolla_Logo.png", width=200) # Placeholder Logo
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Pilih Menu:", ["Dashboard Segmentasi", "Simulator Produk Baru", "Strategi Bisnis"])

st.sidebar.markdown("---")
st.sidebar.info("System v1.0 | Created by Data Science Team")

# ==========================================
# 4. HALAMAN 1: DASHBOARD SEGMENTASI
# ==========================================
if menu == "Dashboard Segmentasi":
    st.title("📊 Dashboard Segmentasi Produk")
    st.markdown("Analisis persebaran produk berdasarkan Harga, Rating, dan Popularitas.")

    # Top Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Produk", f"{len(df):,}")
    c2.metric("Total Cluster", "5 Segmen")
    c3.metric("Avg Repurchase Rate", f"{int(df['total_repurchase_yes_count'].mean()):,} User")

    # Visualisasi PCA (Scatter Plot)
    st.subheader("Peta Persebaran Produk (PCA Visualization)")
    
    col_chart, col_desc = st.columns([2, 1])
    
    with col_chart:
        # Kita perlu menghitung koordinat PCA ulang agar plotnya akurat sesuai model
        # Mengambil fitur numerik yang digunakan saat training
        features_clustering = ['price_clean', 'average_rating', 'total_reviews', 'total_repurchase_yes_count', 'beauty_point_earned']
        X_data = df[features_clustering].dropna()
        
        # Transformasi data untuk plot
        X_scaled_plot = scaler.transform(X_data)
        pca_coords = pca.transform(X_scaled_plot)
        
        df_plot = pd.DataFrame(pca_coords, columns=['PCA1', 'PCA2'])
        df_plot['Cluster'] = df.loc[X_data.index, 'Cluster']
        df_plot['Segment'] = df_plot['Cluster'].map(CLUSTER_NAMES)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='PCA1', y='PCA2', 
            hue='Segment', 
            palette='viridis', 
            data=df_plot, 
            s=80, alpha=0.7, edgecolor='w', ax=ax
        )
        plt.title('Peta Segmentasi Produk (2D PCA)')
        st.pyplot(fig)

    with col_desc:
        st.markdown("### Profil Cluster")
        selected_cluster = st.selectbox("Pilih Cluster untuk Detail:", list(CLUSTER_NAMES.values()))
        
        # Filter data
        cluster_id = [k for k, v in CLUSTER_NAMES.items() if v == selected_cluster][0]
        subset = df[df['Cluster'] == cluster_id]
        
        st.write(f"**Jumlah Produk:** {len(subset)}")
        st.write(f"**Rata-rata Harga:** Rp {subset['price_clean'].mean()*1000000:.0f} (Estimasi)") # Sesuaikan denormalisasi jika perlu
        st.write(f"**Rata-rata Rating:** {subset['average_rating'].mean():.2f}")
        st.write(f"**Rata-rata Review:** {int(subset['total_reviews'].mean())}")

# ==========================================
# 5. HALAMAN 2: SIMULATOR (CORE FEATURE)
# ==========================================
elif menu == "Simulator Produk Baru":
    st.title("🤖 Simulator Prediksi Produk")
    st.markdown("""
    Simulasikan peluncuran produk baru. Sistem akan melakukan 2 tahap prediksi:
    1. **Prediksi Cluster:** Produk ini masuk kategori "Legends" atau "Silent Newcomer"?
    2. **Prediksi Loyalitas:** Berapa estimasi user yang akan membeli ulang (Repurchase)?
    """)

    st.markdown("---")

    # Input User
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Atribut Produk")
        input_name = st.text_input("Nama Produk (Opsional)", "New Serum X")
        
        # Ambil daftar kategori dari data
        categories = df['default_category'].unique().tolist() if 'default_category' in df.columns else ['Toner', 'Serum', 'Cream', 'Mask']
        input_category = st.selectbox("Kategori Produk", categories)
        
        input_price = st.number_input("Harga Produk (Rp)", min_value=1000, value=150000, step=10000)
        input_rating = st.slider("Target Rating (Ekspektasi)", 1.0, 5.0, 4.5)

    with col2:
        st.subheader("2. Skenario Pasar (Untuk Clustering)")
        st.info("Karena ini produk baru, kita perlu mengasumsikan kondisinya untuk menentukan Cluster.")
        
        scenario = st.radio("Pilih Skenario:", [
            "Cold Start (Produk Baru Rilis - Review 0)",
            "Rising Star (Mulai Viral - Review Sedang)",
            "Viral Hit (Sangat Populer - Review Tinggi)"
        ])
        
        # Logika Skenario (Dummy Values untuk Clustering)
        if scenario == "Cold Start":
            dummy_reviews = 0
            dummy_repurchase = 0
            dummy_points = 0
        elif scenario == "Rising Star":
            dummy_reviews = 500
            dummy_repurchase = 100
            dummy_points = 5000
        else: # Viral Hit
            dummy_reviews = 5000
            dummy_repurchase = 1000
            dummy_points = 50000

    # Tombol Prediksi
    if st.button("Jalankan Prediksi 🚀"):
        
        # --- TAHAP 1: PREDIKSI CLUSTER (K-MEANS) ---
        # Kita perlu normalisasi harga dulu agar sesuai skala training
        # Asumsi: price_clean di data training adalah hasil StandardScaler harga rupiah asli
        # Tapi scaler.fit_transform dilakukan pada 5 fitur sekaligus.
        # Maka kita harus buat array 5 fitur untuk di-transform.
        
        # Fitur urut: price_clean (raw price here?), rating, reviews, repurchase, points
        # CATATAN PENTING: Scaler dilatih pada data RAW atau data yang sudah dibersihkan?
        # Biasanya scaler menerima data mentah. Jika input model adalah 'price_clean' (sudah diskalakan),
        # maka kita gunakan scaler untuk mengubah Rupiah -> Standardized Value.
        
        input_features = np.array([[input_price, input_rating, dummy_reviews, dummy_repurchase, dummy_points]])
        input_scaled = scaler.transform(input_features) # Output: Array 5 kolom yang sudah diskalakan
        
        # Ambil nilai scaled untuk dipakai di regression nanti
        price_scaled_val = input_scaled[0][0] # Ini adalah 'price_clean'
        
        # Prediksi Cluster
        pred_cluster_id = kmeans.predict(input_scaled)[0]
        pred_cluster_name = CLUSTER_NAMES[pred_cluster_id]
        
        # --- TAHAP 2: PREDIKSI REPURCHASE (REGRESSION) ---
        # Model Regression butuh input: Price Clean, Average Rating, Default Category, Cluster ID
        # Kita siapkan DataFramenya
        
        reg_input_df = pd.DataFrame({
            'price_clean': [price_scaled_val],
            'average_rating': [input_rating],
            'default_category': [input_category],
            'Cluster': [pred_cluster_id]
        })
        
        # Prediksi Repurchase
        pred_repurchase = reg_pipeline.predict(reg_input_df)[0]
        
        # --- TAMPILKAN HASIL ---
        st.success("Prediksi Selesai!")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("### Hasil Segmentasi")
            st.metric("Masuk ke Cluster", pred_cluster_name)
            if pred_cluster_id == 4:
                st.balloons()
                st.success("Produk ini berpotensi menjadi LEGEND! 🌟")
            elif pred_cluster_id == 1:
                st.warning("Hati-hati, produk ini masuk kategori Silent Newcomers.")
                
        with res_col2:
            st.markdown("### Prediksi Loyalitas")
            st.metric("Estimasi Repurchase (Beli Ulang)", f"{int(pred_repurchase)} User")
            st.caption(f"Menggunakan algoritma XGBoost dengan input Cluster {pred_cluster_id}")

# ==========================================
# 6. HALAMAN 3: STRATEGI BISNIS
# ==========================================
elif menu == "Strategi Bisnis":
    st.title("💡 Rekomendasi Strategi")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "The Legends", "The Popular One", "Reliable Daily", "High-End Luxury", "Silent Newcomers"
    ])
    
    with tab1: # Cluster 4
        st.header("Strategi: The Legends (Cluster 4)")
        st.success("**Karakter:** Review Masif, Repurchase Tinggi, Harga Murah")
        st.markdown("""
        * **Zero Out-of-Stock:** Jangan sampai stok kosong karena ini penarik traffic utama.
        * **Loss Leader:** Gunakan sebagai produk pancingan di halaman depan.
        """)
        
    with tab2: # Cluster 2
        st.header("Strategi: The Popular One (Cluster 2)")
        st.info("**Karakter:** Review Tinggi, Rating Bagus, Harga Terjangkau")
        st.markdown("""
        * **Bundling:** Paket hemat (Cluster 4 + Cluster 2) untuk naikkan nilai keranjang belanja.
        * **Ads Push:** Iklan sedikit lagi untuk menjadikannya 'Legend'.
        """)
        
    with tab5: # Cluster 1
        st.header("Strategi: Silent Newcomers (Cluster 1)")
        st.error("**Karakter:** Rating/Review Nol, Stok Berisiko Mati")
        st.markdown("""
        * **Review Generation:** Berikan sampel gratis/diskon besar demi ulasan pertama.
        * **Clearance Sale:** Jika > 6 bulan, lakukan cuci gudang.
        """)
    
    with tab3:
        st.write("Isi strategi untuk Reliable Daily Drivers sesuai laporan Anda...")
        
    with tab4:
        st.write("Isi strategi untuk High-End Luxury sesuai laporan Anda...")