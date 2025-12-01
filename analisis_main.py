import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Fungsi utama aplikasi
# -----------------------------------
def main():
    st.title("Aplikasi Analisis Survei – Statistik Deskriptif")
    st.write(
        """
        Aplikasi ini membantu menghitung:
        - Mean, Median, Mode
        - Minimum dan Maximum
        - Standard Deviation
        - Tabel Frekuensi dan Persentase
        - Histogram dan Boxplot (opsional)
        """
    )

    # -----------------------------------
    # Upload file data
    # -----------------------------------
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload file data (CSV atau Excel)", 
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        # Baca file sesuai jenisnya
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Preview Data")
        st.dataframe(df.head())

        # Pilih kolom numerik (misal item Likert 1–5)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            st.warning("Tidak ditemukan kolom numerik. Pastikan data Likert bertipe angka (int/float).")
            return

        st.sidebar.markdown("---")
        st.sidebar.subheader("Pengaturan Analisis")
        pilih_semua = st.sidebar.checkbox("Gunakan semua kolom numerik untuk statistik deskriptif", value=True)

        if pilih_semua:
            selected_cols = numeric_cols
        else:
            selected_cols = st.sidebar.multiselect(
                "Pilih kolom untuk dianalisis:",
                options=numeric_cols,
                default=numeric_cols[:1]
            )

        if not selected_cols:
            st.warning("Silakan pilih minimal satu kolom untuk dianalisis.")
            return

        # -----------------------------------
        # Statistik Deskriptif: Mean, Median, Mode, Min, Max, Std
        # -----------------------------------
        st.subheader("1. Statistik Deskriptif")

        stats = pd.DataFrame(
            index=["Mean", "Median", "Mode", "Min", "Max", "Std Dev"],
            columns=selected_cols
        )

        for col in selected_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            stats.loc["Mean", col] = series.mean()
            stats.loc["Median", col] = series.median()
            modes = series.mode()
            stats.loc["Mode", col] = modes.iloc[0] if not modes.empty else np.nan
            stats.loc["Min", col] = series.min()
            stats.loc["Max", col] = series.max()
            stats.loc["Std Dev", col] = series.std()

        # Biar lebih rapi dibulatkan 3 desimal
        stats = stats.astype(float).round(3)

        st.write("Tabel ringkasan statistik deskriptif:")
        st.dataframe(stats.T)  # .T agar tiap baris = 1 variabel

        # -----------------------------------
        # Tabel Frekuensi & Persentase
        # -----------------------------------
        st.subheader("2. Tabel Frekuensi & Persentase")

        all_cols = df.columns.tolist()
        kolom_freq = st.selectbox(
            "Pilih kolom untuk dibuat tabel frekuensi:",
            options=all_cols,
            index=0
        )

        if kolom_freq:
            series = df[kolom_freq]
            freq = series.value_counts(dropna=False).rename("Frekuensi")
            percent = (series.value_counts(normalize=True, dropna=False) * 100).rename("Persentase (%)")

            freq_table = pd.concat([freq, percent], axis=1)
            freq_table["Persentase (%)"] = freq_table["Persentase (%)"].round(2)

            st.write(f"Tabel frekuensi dan persentase untuk kolom: **{kolom_freq}**")
            st.dataframe(freq_table)

        # -----------------------------------
        # Histogram
        # -----------------------------------
        st.subheader("3. Histogram (Opsional)")

        kolom_hist = st.selectbox(
            "Pilih kolom numerik untuk histogram:",
            options=numeric_cols,
            index=0
        )

        if kolom_hist:
            fig, ax = plt.subplots()
            ax.hist(df[kolom_hist].dropna(), bins=10)
            ax.set_title(f"Histogram {kolom_hist}")
            ax.set_xlabel(kolom_hist)
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)

        # -----------------------------------
        # Boxplot
        # -----------------------------------
        st.subheader("4. Boxplot (Opsional)")

        kolom_box = st.selectbox(
            "Pilih kolom numerik untuk boxplot:",
            options=numeric_cols,
            index=0,
            key="boxplot_select"
        )

        if kolom_box:
            fig2, ax2 = plt.subplots()
            ax2.boxplot(df[kolom_box].dropna(), vert=True)
            ax2.set_title(f"Boxplot {kolom_box}")
            ax2.set_ylabel(kolom_box)
            st.pyplot(fig2)

    else:
        st.info("Silakan upload file data pada panel sebelah kiri untuk memulai analisis.")


if __name__ == "__main__":
    main()
