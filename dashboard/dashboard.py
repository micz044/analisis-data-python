import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# buat relative path
day_data_path = 'submission/data/day.csv'
hour_data_path = 'submission/data/hour.csv'

# Load data
day_data = pd.read_csv('submission/data/day.csv')
hour_data = pd.read_csv(hour_data_path)

# Grouping data
weather_group = day_data.groupby('weathersit')['cnt'].mean().reset_index()
weekday_group = day_data.groupby('weekday')['cnt'].mean().reset_index()

# kita asumsikan bahwa' instant' mewakili hari yang unik, kita bisa men simulasikan menggunakan feature user_id
day_data['user_id'] = day_data.index + 1

# RFM Analysis
rfm_table = day_data.groupby('user_id').agg({
    'dteday': 'max',  # Recency
    'instant': 'count',  # Frequency
    'cnt': 'sum'  # Monetary
}).reset_index()

rfm_table.rename(columns={'dteday': 'Recency', 'instant': 'Frequency', 'cnt': 'Monetary'}, inplace=True)
last_date = pd.to_datetime(day_data['dteday']).max()
rfm_table['Recency'] = (last_date - pd.to_datetime(rfm_table['Recency'])).dt.days

# Clustering
# Pastikan semua kolom yang relevan adalah numerik
day_data = day_data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']]
day_data = day_data.dropna()  # hapus baris dengan nilai yang hilang

# Standarisasi feature
scaler = StandardScaler()
scaled_features = scaler.fit_transform(day_data)

# mengaplikasikan KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
day_data['cluster'] = kmeans.fit_predict(scaled_features)
cluster_summary = day_data.groupby('cluster').mean().reset_index()

# Title
st.title('Dashboard Analisis Penggunaan Sepeda')
st.write(
    """
    - **Nama:** Michael Vincent Efren Malamo
    - **Email:** alvaromichael044@gmail.com
    - **ID Dicoding:**alvaromichael044@gmail.com
    """
)
# RFM Analysis Section
st.header('RFM Analysis')
st.write("""
RFM (Recency, Frequency, Monetary) analysis adalah metode yang digunakan untuk mengelompokkan pengguna berdasarkan tiga metrik utama:
- **Recency**: Mengukur seberapa baru pengguna terakhir kali menggunakan layanan (berapa hari sejak terakhir kali pengguna menyewa sepeda).
- **Frequency**: Mengukur seberapa sering pengguna menggunakan layanan (jumlah hari aktif pengguna).
- **Monetary**: Mengukur total nilai dari transaksi yang dilakukan oleh pengguna (total jumlah penyewaan sepeda oleh pengguna).
""")
st.write("Tabel berikut menunjukkan hasil RFM analysis pada data pengguna:")
st.dataframe(rfm_table.head())

# Clustering Section
st.header('Clustering Analysis')
st.write("""
Clustering adalah teknik analisis yang digunakan untuk mengelompokkan data berdasarkan kesamaan. Dalam konteks ini, kita menggunakan teknik KMeans untuk mengelompokkan data penyewaan sepeda berdasarkan variabel-variabel berikut:
- **Suhu (temp)**: Suhu lingkungan.
- **Suhu Nyata (atemp)**: Suhu yang dirasakan.
- **Kelembaban (hum)**: Kelembaban udara.
- **Kecepatan Angin (windspeed)**: Kecepatan angin.
- **Total Penyewaan Sepeda (cnt)**: Jumlah total penyewaan sepeda.
""")
st.write("Tabel berikut menunjukkan hasil clustering pada data pengguna:")
st.dataframe(cluster_summary)

# Visualization 1: Weather Situation
st.header('Pengaruh Kondisi Cuaca terhadap Jumlah Pengguna')
st.write("""
Jumlah pengguna sepeda cenderung lebih tinggi pada kondisi cuaca cerah. 
Pengguna menurun pada kondisi cuaca mendung dan sangat sedikit pada kondisi cuaca hujan.
""")
fig, ax = plt.subplots()
sns.barplot(x='weathersit', y='cnt', data=weather_group, ax=ax)
ax.set_title('Rata-rata Jumlah Pengguna Berdasarkan Kondisi Cuaca')
ax.set_xlabel('Kondisi Cuaca')
ax.set_ylabel('Rata-rata Jumlah Pengguna')
ax.set_xticks(ticks=[0, 1, 2], labels=['Cerah', 'Mendung', 'Hujan'])
st.pyplot(fig)


# Visualization 2: Weekday
st.header('Pola Penggunaan Sepeda Berdasarkan Hari dalam Seminggu')
st.write("""
Penggunaan sepeda cenderung lebih rendah pada hari Minggu dan Senin. 
Jumlah pengguna meningkat dari Selasa hingga Jumat, dengan puncaknya pada hari Jumat. 
Pada hari Sabtu, jumlah pengguna sedikit menurun dibandingkan hari Jumat.
""")
fig, ax = plt.subplots()
sns.lineplot(x='weekday', y='cnt', data=weekday_group, marker='o', ax=ax)
ax.set_title('Rata-rata Jumlah Pengguna Berdasarkan Hari dalam Seminggu')
ax.set_xlabel('Hari dalam Seminggu')
ax.set_ylabel('Rata-rata Jumlah Pengguna')
ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Minggu', 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu'])
st.pyplot(fig)

