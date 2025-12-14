# 💄 Sociolla Product Intelligence: Segmentation & Prediction App

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

**Sociolla Product Intelligence** adalah aplikasi Business Intelligence berbasis Web yang dirancang untuk membantu tim manajemen produk dalam menganalisis posisi pasar dan memprediksi loyalitas pelanggan.

Proyek ini menggabungkan dua pendekatan Machine Learning:

1.  **Unsupervised Learning (Clustering):** Mengelompokkan produk skincare berdasarkan kemiripan harga, rating, dan popularitas untuk strategi pemasaran yang terarah.
2.  **Supervised Learning (Regression):** Memprediksi jumlah pembelian ulang (_Repurchase Count_) untuk produk baru guna mengoptimalkan stok.

## Key Features

Aplikasi Streamlit ini memiliki 3 modul utama:

### 1. Dashboard Segmentasi (Market Overview)

Visualisasi interaktif menggunakan **PCA (Principal Component Analysis)** untuk memetakan produk ke dalam 5 segmen karakteristik:

- **The Legends:** Produk murah dengan review masif dan repurchase tinggi (Traffic Generator).
- **The Popular One:** Produk terjangkau dengan rating bagus yang menopang penjualan (Growth Driver).
- **Reliable Daily Drivers:** Produk 'tulang punggung' dengan variasi terbanyak.
- **High-End Luxury:** Produk premium (mahal) dengan risiko kepuasan pelanggan.
- **Silent Newcomers:** Produk baru atau _dead stock_ dengan interaksi minim.

### 2. Product Simulator (Prediction Engine)

Fitur simulasi untuk memprediksi performa produk baru sebelum diluncurkan.

- **Input:** Harga, Kategori, dan Estimasi Rating.
- **Output:** Prediksi Cluster & Prediksi Angka Repurchase (Loyalitas).
- **Model:** Menggunakan **XGBoost Regressor** yang terbukti memiliki performa terbaik dengan $R^2$ Score ~79%.

### 3. Strategy Recommendation

Menyediakan panduan strategi bisnis berbasis data ("Actionable Insights"), seperti strategi _bundling_ untuk produk populer atau _flash sale_ untuk produk baru.

## Tech Stack

- **Language:** Python
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (K-Means, PCA), XGBoost
- **Visualization:** Matplotlib, Seaborn
