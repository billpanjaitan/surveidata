from typing import List, Optional

import itertools
import os
import base64
import string
import time
from collections import Counter
from io import BytesIO

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from scipy.stats import pearsonr, spearmanr, chi2_contingency, normaltest

st.set_page_config(page_title="Survey Data", layout="wide")

# --------------------------- NLTK INIT ---------------------------
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
EN_STOPWORDS = set(stopwords.words("english"))
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

# ---------- VIDEO BACKGROUND (full-screen) ----------
def set_video_background(video_path: str) -> None:
    """Set an mp4 video as full-screen background using HTML/CSS (base64)."""
    if not os.path.exists(video_path):
        st.warning(f"Video background tidak ditemukan: {video_path}")
        return

    with open(video_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    video_data_url = f"data:video/mp4;base64,{b64}"

    st.markdown(
        f"""
        <style>
        .video-bg {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
        }}
        .stApp {{
            background: transparent !important;
        }}
        </style>
        <video class="video-bg" autoplay muted loop playsinline>
            <source src="{video_data_url}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Multi-language texts (EN, ID, JP, KR, CN)
# ------------------------------------------------------------
TEXTS = {
    "EN": {
        "title": "Survey Analysis Dashboard 📊",
        "subtitle": "Upload your survey data to see clear statistics, visual insights, and multi-language PDF reports ready to share 📈.",
        "dark_mode": "Dark mode 🌙",
        "language": "Language 🌐",
        "upload_label": "Upload CSV or Excel file 📂",
        "no_file": "Please upload a CSV or Excel file to get started 🚀.",
        "invalid_file_type": "This file type is not supported, please upload a CSV, XLS, or XLSX file ⚠️.",
        "preview_title": "Data preview 👀",
        "summary_title": "Dataset overview 📂",
        "rows": "Rows 🔢",
        "cols": "Columns 🔢",
        "num_cols": "Numeric columns 🔢",
        "cat_cols": "Categorical columns 🧩",
        "text_cols": "Text columns 📝",
        "tab_desc": "Descriptive statistics 📌",
        "tab_visual": "Visualizations 📊",
        "tab_corr": "Correlations & tests 🔗",
        "tab_text": "Text analysis 💬",
        "select_numeric_col": "Select a numeric column 🎯",
        "select_numeric_col_x": "Select numeric variable X 📈",
        "select_numeric_col_y": "Select numeric variable Y 📉",
        "select_cat_col1": "Select categorical variable 1 🧩",
        "select_cat_col2": "Select categorical variable 2 🧩",
        "select_cat_col": "Select a categorical column 🧩",
        "select_text_col": "Select a text column 📝",
        "desc_stats_title": "Summary statistics for your data 📌",
        "normaltest_title": "Normality test (D’Agostino–Pearson) 📏",
        "normaltest_not_enough": "There are not enough valid observations for a normality test (need at least 8) ⚠️.",
        "statistic": "Statistic 📊",
        "pvalue": "p-value 📉",
        "alpha_note": "Using significance level α = 0.05 🎯.",
        "normal_interpret": "The data is likely consistent with a normal distribution (fail to reject H₀) ✅.",
        "not_normal_interpret": "The data is unlikely to follow a normal distribution (reject H₀) ⚠️.",
        "hist_title": "Histogram 📊",
        "box_title": "Boxplot 📦",
        "freq_table_title": "Frequency table 📋",
        "count": "Count 🔢",
        "percent": "Percent (%) 📈",
        "visual_hist_title": "Histogram for the selected numeric column 📊",
        "visual_box_title": "Boxplot for the selected numeric column 📦",
        "scatter_title": "Scatter plot 🔍",
        "scatter_x": "X axis ➡️",
        "scatter_y": "Y axis ⬆️",
        "bar_title": "Bar chart (top 20 categories) 📊",
        "corr_matrix_title": "Pearson correlation matrix 🧮",
        "pearson_title": "Pearson correlation 📐",
        "spearman_title": "Spearman correlation 📐",
        "r_label": "Correlation (r) 🔗",
        "strength": "Strength 💪",
        "direction": "Direction ➡️",
        "p_label": "p-value 📉",
        "strength_very_weak": "Very weak 💧",
        "strength_weak": "Weak 🌱",
        "strength_moderate": "Moderate ⚖️",
        "strength_strong": "Strong 💪",
        "strength_very_strong": "Very strong 🔥",
        "direction_positive": "Positive 📈",
        "direction_negative": "Negative 📉",
        "direction_none": "None 🚫",
        "chi_square_title": "Chi-square test of independence 🧪",
        "chi2_label": "Chi-square (χ²) 🧮",
        "df_label": "Degrees of freedom 🎚️",
        "expected_title": "Expected frequencies 📊",
        "observed_title": "Observed frequencies 📊",
        "text_preview_title": "Sample tokens from your text 👀",
        "top_words_title": "Top 10 most frequent words 🔝",
        "pdf_title": "Export PDF report 📄",
        "pdf_button": "Create PDF report 🖨️",
        "pdf_ready": "Your PDF report is ready, use the button below to download it ✅.",
        "pdf_download": "Download PDF report 📥",
        "pdf_filename": "survey_report_en.pdf",
        "no_numeric": "No numeric columns were detected in this dataset ⚠️.",
        "no_categorical": "No categorical columns were detected in this dataset ⚠️.",
        "no_text": "No text columns were detected in this dataset ⚠️.",
        "loading_pdf": "Building your PDF report, please wait ⏳.",
        "scatter_note": "The scatter plot only uses rows where both selected columns have valid values ✅.",
        "matrix_note": "The correlation matrix is computed using the Pearson method for all numeric columns 📐.",
        "text_processing_note": "Text is lowercased, punctuation is removed, and English stopwords are filtered out 🧹.",
        "app_footer": "Built with Streamlit · Survey analysis assistant 💡.",
        "team_members_title": "Team members 👥",
        "team_members_box_title": "Project team 👥",
        "team_member_1": "Regina Vinta Amanullah (004202400133) 🎓",
        "team_member_2": "Bill Christian Panjaitan (004202400058) 🎓",
        "team_member_3": "Putri Lasrida Malau (004202400132) 🎓",
        "team_member_4": "Elizabeth Kurniawan (004202400001) 🎓",
        "pdf_generated_on": "Generated on %Y-%m-%d %H:%M:%S 🕒",
        "pdf_dataset_metadata": "Dataset metadata ℹ️",
        "pdf_numeric_stats": "Numeric column statistics 🔢",
        "pdf_scatter_plots": "Scatter plots for numeric pairs �",
        "pdf_ready": "Your PDF report is ready, use the button below to download it ✅.",
        "pdf_download": "Download PDF report 📥",
        "pdf_filename": "survey_report_en.pdf",
        "no_numeric": "No numeric columns were detected in this dataset ⚠️.",
        "no_categorical": "No categorical columns were detected in this dataset ⚠️.",
        "no_text": "No text columns were detected in this dataset ⚠️.",
        "loading_pdf": "Building your PDF report, please wait ⏳.",
        "scatter_note": "The scatter plot only uses rows where both selected columns have valid values ✅.",
        "matrix_note": "The correlation matrix is computed using the Pearson method for all numeric columns 📐.",
        "text_processing_note": "Text is lowercased, punctuation is removed, and English stopwords are filtered out 🧹.",
        "app_footer": "Built with Streamlit · Survey analysis assistant 💡.",
        "team_members_title": "Team members 👥",
        "team_members_box_title": "Project team 👥",
        "team_member_1": "Regina Vinta Amanullah (004202400133) 🎓",
        "team_member_2": "Bill Christian Panjaitan (004202400058) 🎓",
        "team_member_3": "Putri Lasrida Malau (004202400132) 🎓",
        "team_member_4": "Elizabeth Kurniawan (004202400001) 🎓",
        "pdf_generated_on": "Generated on %Y-%m-%d %H:%M:%S 🕒",
        "pdf_dataset_metadata": "Dataset metadata ℹ️",
        "pdf_numeric_stats": "Numeric column statistics 🔢",
        "pdf_scatter_plots": "Scatter plots for numeric pairs 🔍",
        "pdf_cat_cols": "Categorical columns (top 10 categories) 🧩",
        "pdf_text_summary": "Text analysis summary (top 10 words per column) 💬",
        "pdf_column": "Column 📁",
        "pdf_text_column": "Text column 📝",
        "pdf_normaltest_stat_label": "Normality statistic 📏",
        "pdf_p_value_label": "p-value 📉",
        "pdf_count": "Count 🔢",
        "pdf_mean": "Mean 📊",
        "pdf_median": "Median 📊",
        "pdf_mode": "Mode 📊",
        "pdf_min": "Min 🔽",
        "pdf_max": "Max 🔼",
        "pdf_std": "Std. deviation 📊",
        "pdf_normaltest_not_enough": "Normality test: not enough data (n < 8) ⚠️.",
        "no_valid_data": "There are no valid values in the selected column ⚠️.",
        "select_two_diff_numeric": "Please select two different numeric columns 🙂.",
        "not_enough_corr": "There is not enough data to compute a reliable correlation ⚠️.",
        "not_enough_scatter": "There is not enough complete data to draw a scatter plot ⚠️.",
        "select_two_diff_categorical": "Please select two different categorical columns 🙂.",
        "not_enough_chi": "There is not enough data to run a Chi-square test ⚠️.",
        "quick_interp_title": "Quick interpretation 💡",
        "quick_interp_hist_1": "The histogram shows how often values fall into each range, revealing the overall shape of the distribution 📊.",
        "quick_interp_hist_2": "The boxplot summarizes the median, spread, and possible outliers in the selected numeric column 📦.",
        "quick_interp_scatter_1": "An upward pattern suggests a positive relationship between the two variables 📈.",
        "quick_interp_scatter_2": "A downward pattern suggests a negative relationship, while a cloud of points suggests little or no linear relationship 📉.",
        "quick_interp_corr_1": "Correlations close to +1 or -1 indicate a strong linear relationship between the variables 📐.",
        "quick_interp_corr_2": "Correlations near 0 suggest little or no linear relationship ⚖️.",
        "x_total": "X Total",
        "y_total": "Y Total",
        "x_total_interp": "X Total is the sum of all values in the 'x' column.",
        "y_total_interp": "Y Total is the sum of all values in the 'y' column.",
        "rows_interp": "Number of rows in the dataset.",
        "cols_interp": "Number of columns in the dataset.",
        "num_cols_interp": "Number of numeric columns.",
        "cat_cols_interp": "Number of categorical columns.",
    },
    "ID": {
        "title": "Dasbor Analisis Survei 📊",
        "subtitle": "Unggah data survei Anda untuk melihat statistik, visualisasi, dan laporan PDF multi-bahasa yang siap dibagikan 📈.",
        "dark_mode": "Mode gelap 🌙",
        "language": "Bahasa 🌐",
        "upload_label": "Unggah file CSV atau Excel 📂",
        "no_file": "Silakan unggah file CSV atau Excel terlebih dahulu 🚀.",
        "invalid_file_type": "Tipe file tidak didukung, unggah file CSV, XLS, atau XLSX ⚠️.",
        "preview_title": "Pratinjau data 👀",
        "summary_title": "Ringkasan dataset 📂",
        "rows": "Jumlah baris 🔢",
        "cols": "Jumlah kolom 🔢",
        "num_cols": "Kolom numerik 🔢",
        "cat_cols": "Kolom kategorikal 🧩",
        "text_cols": "Kolom teks 📝",
        "tab_desc": "Statistik deskriptif 📌",
        "tab_visual": "Visualisasi 📊",
        "tab_corr": "Korelasi & uji 🔗",
        "tab_text": "Analisis teks 💬",
        "select_numeric_col": "Pilih satu kolom numerik 🎯",
        "select_numeric_col_x": "Pilih variabel numerik X 📈",
        "select_numeric_col_y": "Pilih variabel numerik Y 📉",
        "select_cat_col1": "Pilih variabel kategorikal 1 🧩",
        "select_cat_col2": "Pilih variabel kategorikal 2 🧩",
        "select_cat_col": "Pilih kolom kategorikal 🧩",
        "select_text_col": "Pilih kolom teks 📝",
        "desc_stats_title": "Statistik ringkas untuk data Anda 📌",
        "normaltest_title": "Uji normalitas (D’Agostino–Pearson) 📏",
        "normaltest_not_enough": "Data valid belum cukup untuk uji normalitas (minimal 8) ⚠️.",
        "statistic": "Statistik 📊",
        "pvalue": "p-value 📉",
        "alpha_note": "Menggunakan taraf signifikansi α = 0,05 🎯.",
        "normal_interpret": "Data kemungkinan mengikuti distribusi normal (gagal menolak H₀) ✅.",
        "not_normal_interpret": "Data kemungkinan tidak berdistribusi normal (menolak H₀) ⚠️.",
        "hist_title": "Histogram 📊",
        "box_title": "Boxplot 📦",
        "freq_table_title": "Tabel frekuensi 📋",
        "count": "Frekuensi 🔢",
        "percent": "Persentase (%) 📈",
        "visual_hist_title": "Histogram untuk kolom numerik terpilih 📊",
        "visual_box_title": "Boxplot untuk kolom numerik terpilih 📦",
        "scatter_title": "Scatter plot 🔍",
        "scatter_x": "Sumbu X ➡️",
        "scatter_y": "Sumbu Y ⬆️",
        "bar_title": "Diagram batang (20 kategori teratas) 📊",
        "corr_matrix_title": "Matriks korelasi Pearson 🧮",
        "pearson_title": "Korelasi Pearson 📐",
        "spearman_title": "Korelasi Spearman 📐",
        "r_label": "Korelasi (r) 🔗",
        "strength": "Kekuatan 💪",
        "direction": "Arah ➡️",
        "p_label": "p-value 📉",
        "strength_very_weak": "Sangat lemah 💧",
        "strength_weak": "Lemah 🌱",
        "strength_moderate": "Sedang ⚖️",
        "strength_strong": "Kuat 💪",
        "strength_very_strong": "Sangat kuat 🔥",
        "direction_positive": "Positif 📈",
        "direction_negative": "Negatif 📉",
        "direction_none": "Tidak ada 🚫",
        "chi_square_title": "Uji Chi-square keterkaitan 🧪",
        "chi2_label": "Chi-square (χ²) 🧮",
        "df_label": "Derajat bebas 🎚️",
        "expected_title": "Frekuensi harapan 📊",
        "observed_title": "Frekuensi teramati 📊",
        "text_preview_title": "Contoh token dari teks 👀",
        "top_words_title": "10 kata paling sering muncul 🔝",
        "pdf_title": "Ekspor laporan PDF 📄",
        "pdf_button": "Buat laporan PDF 🖨️",
        "pdf_ready": "Laporan PDF siap, gunakan tombol di bawah untuk mengunduh ✅.",
        "pdf_download": "Unduh laporan PDF 📥",
        "pdf_filename": "laporan_survei_id.pdf",
        "no_numeric": "Tidak ada kolom numerik yang terdeteksi di dataset ini ⚠️.",
        "no_categorical": "Tidak ada kolom kategorikal yang terdeteksi di dataset ini ⚠️.",
        "no_text": "Tidak ada kolom teks yang terdeteksi di dataset ini ⚠️.",
        "loading_pdf": "Sedang membuat laporan PDF, harap tunggu ⏳.",
        "scatter_note": "Scatter plot hanya menggunakan baris dengan data lengkap pada kedua kolom ✅.",
        "matrix_note": "Matriks korelasi dihitung dengan metode Pearson untuk semua kolom numerik 📐.",
        "text_processing_note": "Teks diubah ke huruf kecil, tanda baca dihapus, dan stopword bahasa Inggris dihilangkan 🧹.",
        "app_footer": "Dibangun dengan Streamlit · Asisten analisis survei 💡.",
        "team_members_title": "Anggota tim 👥",
        "team_members_box_title": "Tim proyek 👥",
        "team_member_1": "Regina Vinta Amanullah (004202400133) 🎓",
        "team_member_2": "Bill Christian Panjaitan (004202400058) 🎓",
        "team_member_3": "Putri Lasrida Malau (004202400132) 🎓",
        "team_member_4": "Elizabeth Kurniawan (004202400001) 🎓",
        "pdf_generated_on": "Dihasilkan pada %Y-%m-%d %H:%M:%S 🕒",
        "pdf_dataset_metadata": "Metadata dataset ℹ️",
        "pdf_numeric_stats": "Statistik kolom numerik 🔢",
        "pdf_scatter_plots": "Scatter plot untuk pasangan numerik 🔍",
        "pdf_cat_cols": "Kolom kategorikal (20 kategori teratas) 🧩",
        "pdf_text_summary": "Ringkasan analisis teks (10 kata teratas per kolom) 💬",
        "pdf_column": "Kolom 📁",
        "pdf_text_column": "Kolom teks 📝",
        "pdf_normaltest_stat_label": "Statistik normalitas 📏",
        "pdf_p_value_label": "p-value 📉",
        "pdf_count": "Jumlah 🔢",
        "pdf_mean": "Rata-rata 📊",
        "pdf_median": "Median 📊",
        "pdf_mode": "Modus 📊",
        "pdf_min": "Min 🔽",
        "pdf_max": "Maks 🔼",
        "pdf_std": "Simpangan baku 📊",
        "pdf_normaltest_not_enough": "Uji normalitas: data belum cukup (n < 8) ⚠️.",
        "no_valid_data": "Tidak ada nilai valid pada kolom yang dipilih ⚠️.",
        "select_two_diff_numeric": "Pilih dua kolom numerik yang berbeda 🙂.",
        "not_enough_corr": "Data belum cukup untuk menghitung korelasi yang andal ⚠️.",
        "not_enough_scatter": "Data lengkap belum cukup untuk membuat scatter plot ⚠️.",
        "select_two_diff_categorical": "Pilih dua kolom kategorikal yang berbeda 🙂.",
        "not_enough_chi": "Data belum cukup untuk menjalankan uji Chi-square ⚠️.",
        "quick_interp_title": "Interpretasi singkat 💡",
        "quick_interp_hist_1": "Histogram menunjukkan seberapa sering nilai muncul pada tiap rentang sehingga bentuk distribusi data terlihat 📊.",
        "quick_interp_hist_2": "Boxplot merangkum median, sebaran, dan kemungkinan outlier pada kolom numerik terpilih 📦.",
        "quick_interp_scatter_1": "Pola yang cenderung naik menunjukkan hubungan positif antara dua variabel 📈.",
        "quick_interp_scatter_2": "Pola yang cenderung turun menunjukkan hubungan negatif, sedangkan titik menyebar acak menandakan hubungan linear yang lemah atau tidak ada 📉.",
        "quick_interp_corr_1": "Korelasi mendekati +1 atau -1 menandakan hubungan linear yang kuat antara variabel 📐.",
        "quick_interp_corr_2": "Korelasi mendekati 0 menandakan hubungan linear yang lemah atau hampir tidak ada ⚖️.",
        "x_total": "Total X",
        "y_total": "Total Y",
        "x_total_interp": "Total X adalah jumlah semua nilai di kolom 'x'.",
        "y_total_interp": "Total Y adalah jumlah semua nilai di kolom 'y'.",
        "rows_interp": "Jumlah baris dalam dataset.",
        "cols_interp": "Jumlah kolom dalam dataset.",
        "num_cols_interp": "Jumlah kolom numerik.",
        "cat_cols_interp": "Jumlah kolom kategorikal.",
    },
    "JP": {
        "title": "アンケート分析ダッシュボード 📊",
        "subtitle": "アンケートデータをアップロードして、多言語PDFレポート付きの統計と可視化を確認できます 📈。",
        "dark_mode": "ダークモード 🌙",
        "language": "言語 🌐",
        "upload_label": "CSV または Excel ファイルをアップロード 📂",
        "no_file": "はじめに CSV または Excel ファイルをアップロードしてください 🚀。",
        "invalid_file_type": "このファイル形式はサポートされていません。CSV・XLS・XLSX をアップロードしてください ⚠️。",
        "preview_title": "データプレビュー 👀",
        "summary_title": "データセット概要 📂",
        "rows": "行数 🔢",
        "cols": "列数 🔢",
        "num_cols": "数値列 🔢",
        "cat_cols": "カテゴリ列 🧩",
        "text_cols": "テキスト列 📝",
        "tab_desc": "記述統計 📌",
        "tab_visual": "可視化 📊",
        "tab_corr": "相関・検定 🔗",
        "tab_text": "テキスト分析 💬",
        "select_numeric_col": "数値列を選択してください 🎯",
        "select_numeric_col_x": "数値変数 X を選択 📈",
        "select_numeric_col_y": "数値変数 Y を選択 📉",
        "select_cat_col1": "カテゴリ変数 1 を選択 🧩",
        "select_cat_col2": "カテゴリ変数 2 を選択 🧩",
        "select_cat_col": "カテゴリ列を選択 🧩",
        "select_text_col": "テキスト列を選択 📝",
        "desc_stats_title": "データの要約統計量 📌",
        "normaltest_title": "正規性検定（D’Agostino–Pearson）📏",
        "normaltest_not_enough": "正規性検定を行うには有効なデータが 8 件以上必要です ⚠️。",
        "statistic": "統計量 📊",
        "pvalue": "p 値 📉",
        "alpha_note": "有意水準 α = 0.05 を使用します 🎯。",
        "normal_interpret": "データは正規分布とみなせる可能性があります（帰無仮説を棄却しません）✅。",
        "not_normal_interpret": "データは正規分布から外れている可能性があります（帰無仮説を棄却します）⚠️。",
        "hist_title": "ヒストグラム 📊",
        "box_title": "箱ひげ図 📦",
        "freq_table_title": "度数表 📋",
        "count": "件数 🔢",
        "percent": "割合 (%) 📈",
        "visual_hist_title": "選択した数値列のヒストグラム 📊",
        "visual_box_title": "選択した数値列の箱ひげ図 📦",
        "scatter_title": "散布図 🔍",
        "scatter_x": "X 軸 ➡️",
        "scatter_y": "Y 軸 ⬆️",
        "bar_title": "棒グラフ（上位 20 カテゴリ）📊",
        "corr_matrix_title": "ピアソン相関行列 🧮",
        "pearson_title": "ピアソン相関係数 📐",
        "spearman_title": "スピアマン相関係数 📐",
        "r_label": "相関係数 (r) 🔗",
        "strength": "強さ 💪",
        "direction": "方向 ➡️",
        "p_label": "p 値 📉",
        "strength_very_weak": "とても弱い 💧",
        "strength_weak": "弱い 🌱",
        "strength_moderate": "中程度 ⚖️",
        "strength_strong": "強い 💪",
        "strength_very_strong": "非常に強い 🔥",
        "direction_positive": "正の相関 📈",
        "direction_negative": "負の相関 📉",
        "direction_none": "相関なし 🚫",
        "chi_square_title": "カイ二乗検定（独立性）🧪",
        "chi2_label": "カイ二乗値 (χ²) 🧮",
        "df_label": "自由度 🎚️",
        "expected_title": "期待度数 📊",
        "observed_title": "観測度数 📊",
        "text_preview_title": "テキストからのサンプルトークン 👀",
        "top_words_title": "出現頻度トップ 10 の単語 🔝",
        "pdf_title": "PDF レポートをエクスポート 📄",
        "pdf_button": "PDF レポートを作成 🖨️",
        "pdf_ready": "PDF レポートの準備ができました。下のボタンからダウンロードできます ✅。",
        "pdf_download": "PDF レポートをダウンロード 📥",
        "pdf_filename": "survey_report_jp.pdf",
        "no_numeric": "このデータセットには数値列がありません ⚠️。",
        "no_categorical": "このデータセットにはカテゴリ列がありません ⚠️。",
        "no_text": "このデータセットにはテキスト列がありません ⚠️。",
        "loading_pdf": "PDF レポートを作成しています。しばらくお待ちください ⏳。",
        "scatter_note": "散布図は両方の列に有効な値がある行のみを使用します ✅。",
        "matrix_note": "相関行列はすべての数値列に対してピアソン法で計算されます 📐。",
        "text_processing_note": "テキストは小文字化され、句読点が削除され、英語ストップワードが除去されます 🧹。",
        "app_footer": "Streamlit で構築されたアンケート分析アシスタントです 💡。",
        "team_members_title": "チームメンバー 👥",
        "team_members_box_title": "プロジェクトチーム 👥",
        "team_member_1": "Regina Vinta Amanullah (004202400133) 🎓",
        "team_member_2": "Bill Christian Panjaitan (004202400058) 🎓",
        "team_member_3": "Putri Lasrida Malau (004202400132) 🎓",
        "team_member_4": "Elizabeth Kurniawan (004202400001) 🎓",
        "pdf_generated_on": "%Y-%m-%d %H:%M:%S に生成 🕒",
        "pdf_dataset_metadata": "データセットのメタデータ ℹ️",
        "pdf_numeric_stats": "数値列の統計量 🔢",
        "pdf_scatter_plots": "数値ペアの散布図 🔍",
        "pdf_cat_cols": "カテゴリ列（上位 10 カテゴリ）🧩",
        "pdf_text_summary": "テキスト分析サマリー（各列の上位 10 単語）💬",
        "pdf_column": "列 📁",
        "pdf_text_column": "テキスト列 📝",
        "pdf_normaltest_stat_label": "正規性統計量 📏",
        "pdf_p_value_label": "p 値 📉",
        "pdf_count": "件数 🔢",
        "pdf_mean": "平均 📊",
        "pdf_median": "中央値 📊",
        "pdf_mode": "最頻値 📊",
        "pdf_min": "最小値 🔽",
        "pdf_max": "最大値 🔼",
        "pdf_std": "標準偏差 📊",
        "pdf_normaltest_not_enough": "正規性検定：データが不足しています（n < 8）⚠️。",
        "no_valid_data": "選択した列には有効な値がありません ⚠️。",
        "select_two_diff_numeric": "異なる 2 つの数値列を選択してください 🙂。",
        "not_enough_corr": "相関を計算するにはデータが不足しています ⚠️。",
        "not_enough_scatter": "散布図を描くには十分なデータがありません ⚠️。",
        "select_two_diff_categorical": "異なる 2 つのカテゴリ列を選択してください 🙂。",
        "not_enough_chi": "カイ二乗検定を行うにはデータが不足しています ⚠️。",
        "quick_interp_title": "かんたんな読み取りポイント 💡",
        "quick_interp_hist_1": "ヒストグラムは値がどの範囲にどれくらい出現するかを示し、分布の形を直感的に確認できます 📊。",
        "quick_interp_hist_2": "箱ひげ図は中央値、ばらつき、および外れ値の有無を一度に把握するのに役立ちます 📦。",
        "quick_interp_scatter_1": "右上がりのパターンは 2 つの変数の間に正の関係があることを示します 📈。",
        "quick_interp_scatter_2": "右下がりのパターンは負の関係を示し、点が雲のように散らばっている場合は線形な関係が弱いかほとんどないことを示します 📉。",
        "quick_interp_corr_1": "相関係数が +1 や -1 に近いほど、2 つの変数の線形関係は強くなります 📐。",
        "quick_interp_corr_2": "相関係数が 0 に近い場合は、線形な関係が弱いかほとんどないことを意味します ⚖️。",
    },
    "KR": {
        "title": "설문 분석 대시보드 📊",
        "subtitle": "설문 데이터를 업로드하고 통계, 시각화, 다국어 PDF 보고서를 한눈에 확인하세요 📈.",
        "dark_mode": "다크 모드 🌙",
        "language": "언어 🌐",
        "upload_label": "CSV 또는 Excel 파일 업로드 📂",
        "no_file": "먼저 CSV 또는 Excel 파일을 업로드해 주세요 🚀.",
        "invalid_file_type": "지원되지 않는 파일 형식입니다. CSV, XLS 또는 XLSX 파일을 업로드해 주세요 ⚠️.",
        "preview_title": "데이터 미리보기 👀",
        "summary_title": "데이터셋 개요 📂",
        "rows": "행 수 🔢",
        "cols": "열 수 🔢",
        "num_cols": "수치형 열 🔢",
        "cat_cols": "범주형 열 🧩",
        "text_cols": "텍스트 열 📝",
        "tab_desc": "기술 통계 📌",
        "tab_visual": "시각화 📊",
        "tab_corr": "상관관계 및 검정 🔗",
        "tab_text": "텍스트 분석 💬",
        "select_numeric_col": "수치형 열을 선택하세요 🎯",
        "select_numeric_col_x": "수치 변수 X 선택 📈",
        "select_numeric_col_y": "수치 변수 Y 선택 📉",
        "select_cat_col1": "범주형 변수 1 선택 🧩",
        "select_cat_col2": "범주형 변수 2 선택 🧩",
        "select_cat_col": "범주형 열 선택 🧩",
        "select_text_col": "텍스트 열 선택 📝",
        "desc_stats_title": "데이터 요약 통계 📌",
        "normaltest_title": "정규성 검정 (D’Agostino–Pearson) 📏",
        "normaltest_not_enough": "정규성 검정을 수행할 만큼 충분한 데이터(최소 8개)가 없습니다 ⚠️.",
        "statistic": "통계량 📊",
        "pvalue": "p 값 📉",
        "alpha_note": "유의수준 α = 0.05 를 사용합니다 🎯.",
        "normal_interpret": "데이터가 정규 분포와 일치할 가능성이 높습니다 (귀무가설 기각 실패) ✅.",
        "not_normal_interpret": "데이터가 정규 분포에서 벗어날 가능성이 큽니다 (귀무가설 기각) ⚠️.",
        "hist_title": "히스토그램 📊",
        "box_title": "박스플롯 📦",
        "freq_table_title": "도수표 📋",
        "count": "개수 🔢",
        "percent": "비율 (%) 📈",
        "visual_hist_title": "선택한 수치형 열의 히스토그램 📊",
        "visual_box_title": "선택한 수치형 열의 박스플롯 📦",
        "scatter_title": "산점도 🔍",
        "scatter_x": "X축 ➡️",
        "scatter_y": "Y축 ⬆️",
        "bar_title": "막대 그래프 (상위 20개 범주) 📊",
        "corr_matrix_title": "피어슨 상관 행렬 🧮",
        "pearson_title": "피어슨 상관계수 📐",
        "spearman_title": "스피어만 상관계수 📐",
        "r_label": "상관계수 (r) 🔗",
        "strength": "강도 💪",
        "direction": "방향 ➡️",
        "p_label": "p 값 📉",
        "strength_very_weak": "매우 약함 💧",
        "strength_weak": "약함 🌱",
        "strength_moderate": "보통 ⚖️",
        "strength_strong": "강함 💪",
        "strength_very_strong": "매우 강함 🔥",
        "direction_positive": "양의 상관 📈",
        "direction_negative": "음의 상관 📉",
        "direction_none": "상관 없음 🚫",
        "chi_square_title": "카이제곱 독립성 검정 🧪",
        "chi2_label": "카이제곱 (χ²) 🧮",
        "df_label": "자유도 🎚️",
        "expected_title": "기대 도수 📊",
        "observed_title": "관측 도수 📊",
        "text_preview_title": "텍스트 토큰 예시 👀",
        "top_words_title": "가장 자주 등장한 단어 10개 🔝",
        "pdf_title": "PDF 보고서 내보내기 📄",
        "pdf_button": "PDF 보고서 생성 🖨️",
        "pdf_ready": "PDF 보고서가 준비되었습니다. 아래 버튼으로 다운로드하세요 ✅.",
        "pdf_download": "PDF 보고서 다운로드 📥",
        "pdf_filename": "survey_report_kr.pdf",
        "no_numeric": "이 데이터셋에는 수치형 열이 없습니다 ⚠️.",
        "no_categorical": "이 데이터셋에는 범주형 열이 없습니다 ⚠️.",
        "no_text": "이 데이터셋에는 텍스트 열이 없습니다 ⚠️.",
        "loading_pdf": "PDF 보고서를 생성하는 중입니다. 잠시만 기다려 주세요 ⏳.",
        "scatter_note": "산점도는 두 열 모두 값이 존재하는 행만 사용합니다 ✅.",
        "matrix_note": "상관 행렬은 모든 수치형 열에 대해 피어슨 방법으로 계산됩니다 📐.",
        "text_processing_note": "텍스트는 소문자로 변환되고, 구두점이 제거되며, 영어 불용어가 제거됩니다 🧹.",
        "app_footer": "Streamlit으로 제작된 설문 분석 도우미입니다 💡.",
        "team_members_title": "팀 구성원 👥",
        "team_members_box_title": "프로젝트 팀 👥",
        "team_member_1": "Regina Vinta Amanullah (004202400133) 🎓",
        "team_member_2": "Bill Christian Panjaitan (004202400058) 🎓",
        "team_member_3": "Putri Lasrida Malau (004202400132) 🎓",
        "team_member_4": "Elizabeth Kurniawan (004202400001) 🎓",
        "pdf_generated_on": "%Y-%m-%d %H:%M:%S 에 생성됨 🕒",
        "pdf_dataset_metadata": "데이터셋 메타데이터 ℹ️",
        "pdf_numeric_stats": "수치형 열 통계 🔢",
        "pdf_scatter_plots": "수치형 쌍에 대한 산점도 🔍",
        "pdf_cat_cols": "범주형 열 (상위 10개 범주) 🧩",
        "pdf_text_summary": "텍스트 분석 요약 (열별 상위 10개 단어) 💬",
        "pdf_column": "열 📁",
        "pdf_text_column": "텍스트 열 📝",
        "pdf_normaltest_stat_label": "정규성 통계량 📏",
        "pdf_p_value_label": "p 값 📉",
        "pdf_count": "개수 🔢",
        "pdf_mean": "평균 📊",
        "pdf_median": "중앙값 📊",
        "pdf_mode": "최빈값 📊",
        "pdf_min": "최솟값 🔽",
        "pdf_max": "최댓값 🔼",
        "pdf_std": "표준편차 📊",
        "pdf_normaltest_not_enough": "정규성 검정: 데이터가 부족합니다 (n < 8) ⚠️.",
        "no_valid_data": "선택한 열에 유효한 값이 없습니다 ⚠️.",
        "select_two_diff_numeric": "서로 다른 두 개의 수치형 열을 선택해 주세요 🙂.",
        "not_enough_corr": "상관 분석을 수행하기에 데이터가 부족합니다 ⚠️.",
        "not_enough_scatter": "산점도를 그리기에 충분한 데이터가 없습니다 ⚠️.",
        "select_two_diff_categorical": "서로 다른 두 개의 범주형 열을 선택해 주세요 🙂.",
        "not_enough_chi": "카이제곱 검정을 수행하기에 데이터가 부족합니다 ⚠️.",
        "quick_interp_title": "빠른 해석 포인트 💡",
        "quick_interp_hist_1": "히스토그램은 값이 각 구간에 얼마나 자주 나타나는지 보여 주어 분포의 전반적인 모양을 파악할 수 있습니다 📊.",
        "quick_interp_hist_2": "박스플롯은 선택한 수치형 열의 중앙값, 분산 정도, 이상치를 한눈에 요약해 줍니다 📦.",
        "quick_interp_scatter_1": "점들이 대체로 오른쪽 위로 증가하는 모양이면 두 변수 사이에 양의 관계가 있음을 의미합니다 📈.",
        "quick_interp_scatter_2": "점들이 오른쪽 아래로 줄어드는 모양이면 음의 관계를, 구름처럼 흩어져 있으면 선형 관계가 약하거나 거의 없음을 의미합니다 📉.",
        "quick_interp_corr_1": "상관계수가 +1 또는 -1에 가까울수록 두 변수 간의 선형 관계가 강하다는 뜻입니다 📐.",
        "quick_interp_corr_2": "상관계수가 0에 가까우면 선형 관계가 약하거나 거의 없다는 뜻입니다 ⚖️.",
    },
    "CN": {
        "title": "问卷分析仪表盘 📊",
        "subtitle": "上传问卷数据，一站式查看统计结果、可视化图表和多语言 PDF 报告 📈。",
        "dark_mode": "深色模式 🌙",
        "language": "语言 🌐",
        "upload_label": "上传 CSV 或 Excel 文件 📂",
        "no_file": "请先上传一个 CSV 或 Excel 文件以开始分析 🚀。",
        "invalid_file_type": "不支持的文件类型，请上传 CSV、XLS 或 XLSX 文件 ⚠️。",
        "preview_title": "数据预览 👀",
        "summary_title": "数据集概览 📂",
        "rows": "行数 🔢",
        "cols": "列数 🔢",
        "num_cols": "数值列 🔢",
        "cat_cols": "类别列 🧩",
        "text_cols": "文本列 📝",
        "tab_desc": "描述性统计 📌",
        "tab_visual": "可视化 📊",
        "tab_corr": "相关与检验 🔗",
        "tab_text": "文本分析 💬",
        "select_numeric_col": "请选择一个数值列 🎯",
        "select_numeric_col_x": "请选择数值变量 X 📈",
        "select_numeric_col_y": "请选择数值变量 Y 📉",
        "select_cat_col1": "请选择类别变量 1 🧩",
        "select_cat_col2": "请选择类别变量 2 🧩",
        "select_cat_col": "请选择一个类别列 🧩",
        "select_text_col": "请选择一个文本列 📝",
        "desc_stats_title": "数据的汇总统计量 📌",
        "normaltest_title": "正态性检验（D’Agostino–Pearson）📏",
        "normaltest_not_enough": "有效样本不足，无法进行正态性检验（至少需要 8 个样本）⚠️。",
        "statistic": "统计量 📊",
        "pvalue": "p 值 📉",
        "alpha_note": "使用显著性水平 α = 0.05 🎯。",
        "normal_interpret": "数据大致符合正态分布（无法拒绝原假设 H₀）✅。",
        "not_normal_interpret": "数据可能不符合正态分布（拒绝原假设 H₀）⚠️。",
        "hist_title": "直方图 📊",
        "box_title": "箱线图 📦",
        "freq_table_title": "频数表 📋",
        "count": "频数 🔢",
        "percent": "百分比 (%) 📈",
        "visual_hist_title": "选定数值列的直方图 📊",
        "visual_box_title": "选定数值列的箱线图 📦",
        "scatter_title": "散点图 🔍",
        "scatter_x": "X 轴 ➡️",
        "scatter_y": "Y 轴 ⬆️",
        "bar_title": "柱状图（前 20 个类别）📊",
        "corr_matrix_title": "皮尔逊相关矩阵 🧮",
        "pearson_title": "皮尔逊相关系数 📐",
        "spearman_title": "斯皮尔曼相关系数 📐",
        "r_label": "相关系数 (r) 🔗",
        "strength": "强度 💪",
        "direction": "方向 ➡️",
        "p_label": "p 值 📉",
        "strength_very_weak": "非常弱 💧",
        "strength_weak": "较弱 🌱",
        "strength_moderate": "中等 ⚖️",
        "strength_strong": "较强 💪",
        "strength_very_strong": "非常强 🔥",
        "direction_positive": "正相关 📈",
        "direction_negative": "负相关 📉",
        "direction_none": "无明显相关 🚫",
        "chi_square_title": "卡方独立性检验 🧪",
        "chi2_label": "卡方值 (χ²) 🧮",
        "df_label": "自由度 🎚️",
        "expected_title": "期望频数 📊",
        "observed_title": "观测频数 📊",
        "text_preview_title": "文本样本词汇 👀",
        "top_words_title": "出现频率最高的 10 个词 🔝",
        "pdf_title": "导出 PDF 报告 📄",
        "pdf_button": "生成 PDF 报告 🖨️",
        "pdf_ready": "PDF 报告已生成，可以通过下方按钮下载 ✅。",
        "pdf_download": "下载 PDF 报告 📥",
        "pdf_filename": "survey_report_cn.pdf",
        "no_numeric": "此数据集中未检测到数值列 ⚠️。",
        "no_categorical": "此数据集中未检测到类别列 ⚠️。",
        "no_text": "此数据集中未检测到文本列 ⚠️。",
        "loading_pdf": "正在生成 PDF 报告，请稍候 ⏳。",
        "scatter_note": "散点图仅使用在两个列中同时具有有效数值的行 ✅。",
        "matrix_note": "相关矩阵基于所有数值列，使用皮尔逊方法计算 📐。",
        "text_processing_note": "文本将被转换为小写，移除标点符号，并去除英文停用词 🧹。",
        "app_footer": "基于 Streamlit 构建的问卷分析助手 💡。",
        "team_members_title": "团队成员 👥",
        "team_members_box_title": "项目团队 👥",
        "team_member_1": "Regina Vinta Amanullah (004202400133) 🎓",
        "team_member_2": "Bill Christian Panjaitan (004202400058) 🎓",
        "team_member_3": "Putri Lasrida Malau (004202400132) 🎓",
        "team_member_4": "Elizabeth Kurniawan (004202400001) 🎓",
        "pdf_generated_on": "生成时间：%Y-%m-%d %H:%M:%S 🕒",
        "pdf_dataset_metadata": "数据集元信息 ℹ️",
        "pdf_numeric_stats": "数值列统计信息 🔢",
        "pdf_scatter_plots": "数值对的散点图 🔍",
        "pdf_cat_cols": "类别列（前 10 个类别）🧩",
        "pdf_text_summary": "文本分析摘要（每列前 10 个高频词）💬",
        "pdf_column": "列 📁",
        "pdf_text_column": "文本列 📝",
        "pdf_normaltest_stat_label": "正态性统计量 📏",
        "pdf_p_value_label": "p 值 📉",
        "pdf_count": "频数 🔢",
        "pdf_mean": "平均值 📊",
        "pdf_median": "中位数 📊",
        "pdf_mode": "众数 📊",
        "pdf_min": "最小值 🔽",
        "pdf_max": "最大值 🔼",
        "pdf_std": "标准差 📊",
        "pdf_normaltest_not_enough": "正态性检验：样本数量不足（n < 8）⚠️。",
        "no_valid_data": "所选列中没有有效的数值 ⚠️。",
        "select_two_diff_numeric": "请选择两个不同的数值列 🙂。",
        "not_enough_corr": "数据不足以计算可靠的相关系数 ⚠️。",
        "not_enough_scatter": "数据不足以绘制散点图 ⚠️。",
        "select_two_diff_categorical": "请选择两个不同的类别列 🙂。",
        "not_enough_chi": "数据不足以执行卡方检验 ⚠️。",
        "quick_interp_title": "快速解读要点 💡",
        "quick_interp_hist_1": "直方图展示数值在各个区间内出现的频率，可以直观地看出数据分布的整体形状 📊。",
        "quick_interp_hist_2": "箱线图可以同时概括中位数、数据离散程度以及是否存在异常值 📦。",
        "quick_interp_scatter_1": "点大致呈向右上方的趋势，说明两个变量之间存在正相关关系 📈。",
        "quick_interp_scatter_2": "点大致向右下方分布，说明存在负相关；如果点云分布杂乱，则线性相关关系较弱或几乎不存在 📉。",
        "quick_interp_corr_1": "相关系数接近 +1 或 -1 时，表示两个变量之间的线性关系非常强 📐。",
        "quick_interp_corr_2": "相关系数接近 0 时，说明变量之间几乎没有线性关系或关系很弱 ⚖️。",
    },
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_text(key: str) -> str:
    lang = st.session_state.get("language", "EN")
    if lang in TEXTS and key in TEXTS[lang]:
        return TEXTS[lang][key]
    if key in TEXTS["EN"]:
        return TEXTS["EN"][key]
    return key

def apply_theme():
    dark = st.session_state.get("dark_mode", False)
    if dark:
        sns.set_style("darkgrid")
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "axes.facecolor": "#111111",
                "figure.facecolor": "#111111",
                "axes.edgecolor": "#dddddd",
                "xtick.color": "#dddddd",
                "ytick.color": "#dddddd",
                "text.color": "#ffffff",
            }
        )
    else:
        sns.set_style("whitegrid")
        plt.style.use("default")
        plt.rcParams.update(
            {
                "axes.facecolor": "#ffffff",
                "figure.facecolor": "#ffffff",
                "axes.edgecolor": "#222222",
                "xtick.color": "#222222",
                "ytick.color": "#222222",
                "text.color": "#000000",
            }
        )

def load_data(file) -> Optional[pd.DataFrame]:
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    st.error(get_text("invalid_file_type"))
    return None

def preprocess_text_series(series: pd.Series):
    tokens_all = []
    for val in series.dropna():
        text = str(val).lower()
        text = text.translate(PUNCTUATION_TABLE)
        for tok in text.split():
            if tok and tok not in EN_STOPWORDS:
                tokens_all.append(tok)
    counter = Counter(tokens_all)
    return tokens_all, counter

def descriptive_stats(series: pd.Series):
    s = pd.Series(series).dropna()
    if s.empty:
        return None
    desc = {
        "sum": s.sum(),
        "mean": s.mean(),
        "median": s.median(),
        "mode": s.mode().iloc[0] if not s.mode().empty else np.nan,
        "min": s.min(),
        "max": s.max(),
        "std": s.std(ddof=1),
        "count": s.count(),
    }
    if len(s) >= 8:
        try:
            stat, p = normaltest(s)
        except Exception:
            stat, p = None, None
        desc["normaltest_stat"] = stat
        desc["normaltest_p"] = p
    else:
        desc["normaltest_stat"] = None
        desc["normaltest_p"] = None
    return desc

def frequency_tables(series: pd.Series):
    vc = series.value_counts(dropna=False)
    total = vc.sum()
    df_freq = pd.DataFrame(
        {
            get_text("count"): vc,
            get_text("percent"): (vc / total * 100.0).round(2),
        }
    )
    return df_freq

def visualize_data(df: pd.DataFrame, numeric_col: Optional[str] = None, cat_col: Optional[str] = None):
    apply_theme()
    if numeric_col is not None and numeric_col in df.columns:
        col_data = df[numeric_col].dropna()
        fig1, ax1 = plt.subplots()
        sns.histplot(col_data, kde=True, ax=ax1)
        ax1.set_title(f"{get_text('hist_title')} - {numeric_col}")
        st.pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        sns.boxplot(x=col_data, ax=ax2)
        ax2.set_title(f"{get_text('box_title')} - {numeric_col}")
        st.pyplot(fig2)
        plt.close(fig2)

    if cat_col is not None and cat_col in df.columns:
        freq_df = frequency_tables(df[cat_col])
        st.subheader(f"{get_text('freq_table_title')} - {cat_col}")
        st.dataframe(freq_df)

def _interpret_strength(r: float) -> str:
    ar = abs(r)
    if ar < 0.2:
        return get_text("strength_very_weak")
    if ar < 0.4:
        return get_text("strength_weak")
    if ar < 0.6:
        return get_text("strength_moderate")
    if ar < 0.8:
        return get_text("strength_strong")
    return get_text("strength_very_strong")

def _interpret_direction(r: float) -> str:
    if r > 0:
        return get_text("direction_positive")
    if r < 0:
        return get_text("direction_negative")
    return get_text("direction_none")

def correlation_analysis(df: pd.DataFrame, col_x: str, col_y: str):
    data = df[[col_x, col_y]].dropna()
    if data.empty:
        return None
    x = data[col_x]
    y = data[col_y]
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    result = {
        "pearson": {
            "r": pearson_r,
            "p": pearson_p,
            "strength": _interpret_strength(pearson_r),
            "direction": _interpret_direction(pearson_r),
        },
        "spearman": {
            "r": spearman_r,
            "p": spearman_p,
            "strength": _interpret_strength(spearman_r),
            "direction": _interpret_direction(spearman_r),
        },
    }
    return result

def chi_square_test(df: pd.DataFrame, col1: str, col2: str):
    ct = pd.crosstab(df[col1], df[col2])
    if ct.empty:
        return None
    chi2, p, dof, expected = chi2_contingency(ct)
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    return {
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "observed": ct,
        "expected": expected_df,
    }

def build_survey_report_pdf(
    df: pd.DataFrame, numeric_cols, cat_cols, text_cols, language: str
) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 30
    y = height - margin

    texts = TEXTS.get(language, TEXTS["EN"])

    def draw_line(text, font="Helvetica", size=9, new_page_if_needed=True):
        nonlocal y
        c.setFont(font, size)
        if new_page_if_needed and y < margin + 50:
            c.showPage()
            y = height - margin
            c.setFont(font, size)
        c.drawString(margin, y, text)
        y -= size + 3

    c.setTitle(texts["title"])
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, texts["title"])
    y -= 24
    c.setFont("Helvetica", 9)
    draw_line(time.strftime(texts["pdf_generated_on"]), new_page_if_needed=False)
    y -= 4

    draw_line("-" * 90)
    draw_line(texts["pdf_dataset_metadata"], "Helvetica-Bold", 11)
    draw_line(f"{texts['rows']}: {df.shape[0]}")
    draw_line(f"{texts['cols']}: {df.shape[1]}")
    draw_line(f"{texts['num_cols']}: {len(numeric_cols)}")
    draw_line(f"{texts['cat_cols']}: {len(cat_cols)}")
    draw_line(f"{texts['text_cols']}: {len(text_cols)}")

    if numeric_cols:
        draw_line("-" * 90)
        draw_line(texts["pdf_numeric_stats"], "Helvetica-Bold", 11)

        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            desc = descriptive_stats(s)
            draw_line(f"{texts['pdf_column']}: {col}", "Helvetica-Bold", 10)
            draw_line(
                f"  {texts['pdf_count']}: {desc['count']}  "
                f"{texts['pdf_mean']}: {desc['mean']:.4f}  "
                f"{texts['pdf_median']}: {desc['median']:.4f}"
            )
            draw_line(
                f"  {texts['pdf_mode']}: {desc['mode']:.4f}  "
                f"{texts['pdf_min']}: {desc['min']:.4f}  "
                f"{texts['pdf_max']}: {desc['max']:.4f}  "
                f"{texts['pdf_std']}: {desc['std']:.4f}"
            )
            if desc["normaltest_stat"] is not None:
                draw_line(
                    f"  {texts['pdf_normaltest_stat_label']}: {desc['normaltest_stat']:.4f}, "
                    f"{texts['pdf_p_value_label']}: {desc['normaltest_p']:.4g}"
                )
            else:
                draw_line(f"  {texts['pdf_normaltest_not_enough']}")

            apply_theme()
            fig_h, ax_h = plt.subplots()
            sns.histplot(s, kde=True, ax=ax_h)
            ax_h.set_title(f"{texts['hist_title']} - {col}")
            img_buffer = BytesIO()
            fig_h.savefig(img_buffer, format="png", bbox_inches="tight")
            plt.close(fig_h)
            img_buffer.seek(0)
            img = ImageReader(img_buffer)

            if y < margin + 180:
                c.showPage()
                y = height - margin

            c.drawImage(
                img,
                margin,
                y - 140,
                width=width - 2 * margin,
                height=140,
                preserveAspectRatio=True,
                mask="auto",
            )
            y -= 150

            fig_b, ax_b = plt.subplots()
            sns.boxplot(x=s, ax=ax_b)
            ax_b.set_title(f"{texts['box_title']} - {col}")
            img_buffer2 = BytesIO()
            fig_b.savefig(img_buffer2, format="png", bbox_inches="tight")
            plt.close(fig_b)
            img_buffer2.seek(0)
            img2 = ImageReader(img_buffer2)

            if y < margin + 160:
                c.showPage()
                y = height - margin
            c.drawImage(
                img2,
                margin,
                y - 120,
                width=width - 2 * margin,
                height=120,
                preserveAspectRatio=True,
                mask="auto",
            )
            y -= 130

    if len(numeric_cols) >= 2:
        draw_line("-" * 90)
        draw_line(texts["pdf_scatter_plots"], "Helvetica-Bold", 11)
        for col_x, col_y in itertools.combinations(numeric_cols, 2):
            pair_df = df[[col_x, col_y]].dropna()
            if pair_df.shape[0] < 3:
                continue
            apply_theme()
            fig_s, ax_s = plt.subplots()
            sns.scatterplot(data=pair_df, x=col_x, y=col_y, ax=ax_s)
            ax_s.set_title(f"{col_x} vs {col_y}")
            img_buf = BytesIO()
            fig_s.savefig(img_buf, format="png", bbox_inches="tight")
            plt.close(fig_s)
            img_buf.seek(0)
            img_s = ImageReader(img_buf)

            if y < margin + 180:
                c.showPage()
                y = height - margin
            c.drawImage(
                img_s,
                margin,
                y - 140,
                width=width - 2 * margin,
                height=140,
                preserveAspectRatio=True,
                mask="auto",
            )
            y -= 150

    if numeric_cols:
        draw_line("-" * 90)
        draw_line(texts["corr_matrix_title"], "Helvetica-Bold", 11)
        corr = df[numeric_cols].corr(method="pearson")
        cols_list = list(corr.columns)
        header = "      " + "  ".join([str(c)[:6].ljust(6) for c in cols_list])
        draw_line(header)
        for r in cols_list:
            row_vals = [f"{corr.loc[r, c]:.2f}" for c in cols_list]
            row_str = str(r)[:6].ljust(6) + "  " + "  ".join(v.ljust(6) for v in row_vals)
            draw_line(row_str)

    if cat_cols:
        draw_line("-" * 90)
        draw_line(texts["pdf_cat_cols"], "Helvetica-Bold", 11)
        for col in cat_cols:
            draw_line(f"{texts['pdf_column']}: {col}", "Helvetica-Bold", 10)
            vc = df[col].value_counts(dropna=False).head(10)
            total = vc.sum()
            for idx, val in vc.items():
                label = str(idx)
                perc = val / total * 100 if total > 0 else 0
                draw_line(f"  {label[:40]}: {val} ({perc:.1f}%)")

    if text_cols:
        draw_line("-" * 90)
        draw_line(texts["pdf_text_summary"], "Helvetica-Bold", 11)
        for col in text_cols:
            draw_line(f"{texts['pdf_text_column']}: {col}", "Helvetica-Bold", 10)
            _, counter = preprocess_text_series(df[col])
            for word, cnt in counter.most_common(10):
                draw_line(f"  {word}: {cnt}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
def main():
    if "language" not in st.session_state:
        st.session_state["language"] = "EN"
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False

    # Background video
    set_video_background("static/background.mp4")

    # Top bar
    top_left, top_right = st.columns([3, 2])
    with top_left:
        st.markdown(
            f"<h1 style='margin-bottom:0.2rem;'>{get_text('title')}</h1>",
            unsafe_allow_html=True,
        )
        st.caption(get_text("subtitle"))
    with top_right:
        col_mode, col_lang = st.columns(2)
        with col_mode:
            st.session_state["dark_mode"] = st.toggle(
                get_text("dark_mode"),
                value=st.session_state["dark_mode"],
            )
        with col_lang:
            lang_options = list(TEXTS.keys())
            current_lang = st.session_state.get("language", "EN")
            if current_lang not in lang_options:
                current_lang = "EN"
                st.session_state["language"] = "EN"
            st.session_state["language"] = st.selectbox(
                get_text("language"),
                options=lang_options,
                index=lang_options.index(current_lang),
            )

    dark = st.session_state.get("dark_mode", False)
    if dark:
        page_bg = "transparent"
        text_color = "#f5f5f5"
        card_bg = "rgba(20, 20, 20, 0.92)"
        border_color = "#444444"
    else:
        page_bg = "transparent"
        text_color = "#000000"
        card_bg = "rgba(255, 255, 255, 0.92)"
        border_color = "#cccccc"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {page_bg};
            color: {text_color};
        }}
        .card-box {{
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 12px;
            border: 1px solid {border_color};
            background-color: {card_bg};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    apply_theme()

    # Team box
    st.markdown(
        f"""
        <div class="card-box">
            <strong>{get_text("team_members_box_title")}</strong><br>
            {get_text("team_member_1")}<br>
            {get_text("team_member_2")}<br>
            {get_text("team_member_3")}<br>
            {get_text("team_member_4")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload box
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    up1, up2 = st.columns([2, 3])
    with up1:
        uploaded_file = st.file_uploader(
            get_text("upload_label"),
            type=["csv", "xls", "xlsx"],
            key="data_uploader",
        )
    with up2:
        st.info(get_text("alpha_note"))
    st.markdown("</div>", unsafe_allow_html=True)

    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)

    if df is None:
        st.info(get_text("no_file"))
        st.markdown(
            f"<p style='text-align:center;color:gray;margin-top:2rem;'>{get_text('app_footer')}</p>",
            unsafe_allow_html=True,
        )
        return

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    obj_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    cat_cols: List[str] = []
    text_cols: List[str] = []
    for col in obj_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 30:
            cat_cols.append(col)
        else:
            text_cols.append(col)

    # Compute x_total and y_total as sum of all columns starting with 'X' and 'Y'
    x_cols = [col for col in numeric_cols if col.startswith('X')]
    y_cols = [col for col in numeric_cols if col.startswith('Y')]
    x_total = df[x_cols].sum().sum() if x_cols else None
    y_total = df[y_cols].sum().sum() if y_cols else None

    # Compute normality test for X and Y columns
    x_normal_p = None
    if x_cols:
        x_values = df[x_cols].values.flatten()
        if len(x_values) >= 8:
            try:
                _, x_normal_p = normaltest(x_values)
            except:
                pass
    y_normal_p = None
    if y_cols:
        y_values = df[y_cols].values.flatten()
        if len(y_values) >= 8:
            try:
                _, y_normal_p = normaltest(y_values)
            except:
                pass

    # Preview box
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader(get_text("preview_title"))
    st.dataframe(df.head(1000))
    st.markdown("</div>", unsafe_allow_html=True)

    # Overview box
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader(get_text("summary_title"))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(get_text("rows"), df.shape[0])
        st.caption(get_text("rows_interp"))
    with col2:
        st.metric(get_text("cols"), df.shape[1])
        st.caption(get_text("cols_interp"))
    with col3:
        st.metric(get_text("num_cols"), len(numeric_cols))
        st.caption(get_text("num_cols_interp"))
    with col4:
        st.metric(get_text("cat_cols"), len(cat_cols))
        st.caption(get_text("cat_cols_interp"))

    if x_total is not None or y_total is not None:
        st.markdown("---")
        col_x, col_y = st.columns(2)
        with col_x:
            if x_total is not None:
                st.metric(get_text("x_total"), f"{x_total:.2f}")
                st.caption(get_text("x_total_interp"))
        with col_y:
            if y_total is not None:
                st.metric(get_text("y_total"), f"{y_total:.2f}")
                st.caption(get_text("y_total_interp"))
    st.markdown("</div>", unsafe_allow_html=True)

    # Tabs box
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    tab_desc, tab_visual, tab_corr, tab_text = st.tabs(
        [
            get_text("tab_desc"),
            get_text("tab_visual"),
            get_text("tab_corr"),
            get_text("tab_text"),
        ]
    )

    # Tab Deskriptif
    with tab_desc:
        if not numeric_cols and not cat_cols:
            st.warning(get_text("no_numeric") + " " + get_text("no_categorical"))
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {get_text('desc_stats_title')}")
                if x_total is not None:
                    st.metric(get_text("x_total"), f"{x_total:.2f}")
                if y_total is not None:
                    st.metric(get_text("y_total"), f"{y_total:.2f}")
                if x_total is None and y_total is None:
                    st.info("No columns starting with 'X' or 'Y' found.")
            with col2:
                st.markdown(
                    f"#### {get_text('hist_title')} & {get_text('box_title')}"
                )
                if numeric_cols:
                    num_col2 = st.selectbox(
                        get_text("select_numeric_col"),
                        numeric_cols,
                        key="dist_num_col",
                    )
                    visualize_data(df, numeric_col=num_col2, cat_col=None)
                if cat_cols:
                    st.markdown(f"#### {get_text('freq_table_title')}")
                    cat_col = st.selectbox(get_text("select_cat_col"), cat_cols)
                    freq_df = frequency_tables(df[cat_col])
                    st.dataframe(freq_df)
                else:
                    st.info(get_text("no_categorical"))

    # Tab Visual
    with tab_visual:
        st.markdown(f"### {get_text('tab_visual')}")
        st.markdown(
            f"#### {get_text('visual_hist_title')} / {get_text('visual_box_title')}"
        )
        if numeric_cols:
            v_num_col = st.selectbox(
                get_text("select_numeric_col"),
                numeric_cols,
                key="visual_num_col",
            )
            col_data = df[v_num_col].dropna()

            apply_theme()
            fig1, ax1 = plt.subplots()
            sns.histplot(col_data, kde=True, ax=ax1)
            ax1.set_title(f"{get_text('hist_title')} - {v_num_col}")
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots()
            sns.boxplot(x=col_data, ax=ax2)
            ax2.set_title(f"{get_text('box_title')} - {v_num_col}")
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown(f"**{get_text('quick_interp_title')}**")
            st.write(f"- {get_text('quick_interp_hist_1')}")
            st.write(f"- {get_text('quick_interp_hist_2')}")
        else:
            st.warning(get_text("no_numeric"))

        st.markdown(f"#### {get_text('scatter_title')}")
        if len(numeric_cols) >= 2:
            c3, c4 = st.columns(2)
            with c3:
                x_col = st.selectbox(
                    get_text("select_numeric_col_x"),
                    numeric_cols,
                )
            with c4:
                y_col = st.selectbox(
                    get_text("select_numeric_col_y"),
                    numeric_cols,
                    index=1 if len(numeric_cols) > 1 else 0,
                )

            data = df[[x_col, y_col]].dropna()
            if not data.empty:
                apply_theme()
                fig_sc, ax_sc = plt.subplots()
                sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax_sc)
                ax_sc.set_title(
                    f"{get_text('scatter_title')}: {x_col} vs {y_col}"
                )
                st.pyplot(fig_sc)
                plt.close(fig_sc)
                st.caption(get_text("scatter_note"))

                st.markdown(f"**{get_text('quick_interp_title')}**")
                st.write(f"- {get_text('quick_interp_scatter_1')}")
                st.write(f"- {get_text('quick_interp_scatter_2')}")
            else:
                st.info(get_text("not_enough_scatter"))
        else:
            st.info(get_text("no_numeric"))

        st.markdown(f"#### {get_text('bar_title')}")
        if cat_cols:
            b_cat_col = st.selectbox(
                get_text("select_cat_col"),
                cat_cols,
                key="bar_cat_col",
            )
            vc = df[b_cat_col].value_counts().head(20)
            bar_df = vc.reset_index()
            bar_df.columns = [b_cat_col, get_text("count")]

            apply_theme()
            fig_bar, ax_bar = plt.subplots()
            sns.barplot(data=bar_df, x=get_text("count"), y=b_cat_col, ax=ax_bar)
            ax_bar.set_title(f"{get_text('bar_title')} - {b_cat_col}")
            st.pyplot(fig_bar)
            plt.close(fig_bar)
        else:
            st.info(get_text("no_categorical"))

    # Tab Korelasi
    with tab_corr:
        st.markdown(f"### {get_text('tab_corr')}")
        st.markdown(
            f"#### {get_text('pearson_title')} & {get_text('spearman_title')}"
        )
        if len(numeric_cols) >= 2:
            c5, c6 = st.columns(2)
            with c5:
                corr_x = st.selectbox(
                    get_text("select_numeric_col_x"),
                    numeric_cols,
                    key="corr_x",
                )
            with c6:
                corr_y = st.selectbox(
                    get_text("select_numeric_col_y"),
                    numeric_cols,
                    index=1 if len(numeric_cols) > 1 else 0,
                    key="corr_y",
                )
            if corr_x == corr_y:
                st.warning(get_text("select_two_diff_numeric"))
            else:
                res = correlation_analysis(df, corr_x, corr_y)
                if res:
                    st.write(f"**{get_text('pearson_title')}**")
                    st.write(
                        {
                            get_text("r_label"): res["pearson"]["r"],
                            get_text("p_label"): res["pearson"]["p"],
                            get_text("strength"): res["pearson"]["strength"],
                            get_text("direction"): res["pearson"]["direction"],
                        }
                    )
                    st.write(f"**{get_text('spearman_title')}**")
                    st.write(
                        {
                            get_text("r_label"): res["spearman"]["r"],
                            get_text("p_label"): res["spearman"]["p"],
                            get_text("strength"): res["spearman"]["strength"],
                            get_text("direction"): res["spearman"]["direction"],
                        }
                    )

                    st.markdown(f"**{get_text('quick_interp_title')}**")
                    st.write(f"- {get_text('quick_interp_corr_1')}")
                    st.write(f"- {get_text('quick_interp_corr_2')}")
                else:
                    st.info(get_text("not_enough_corr"))
        else:
            st.info(get_text("no_numeric"))

        st.markdown(f"#### {get_text('chi_square_title')}")
        if len(cat_cols) >= 2:
            c7, c8 = st.columns(2)
            with c7:
                chi_c1 = st.selectbox(
                    get_text("select_cat_col1"),
                    cat_cols,
                    key="chi_c1",
                )
            with c8:
                chi_c2 = st.selectbox(
                    get_text("select_cat_col2"),
                    cat_cols,
                    index=1 if len(cat_cols) > 1 else 0,
                    key="chi_c2",
                )
            if chi_c1 == chi_c2:
                st.warning(get_text("select_two_diff_categorical"))
            else:
                chi_res = chi_square_test(df, chi_c1, chi_c2)
                if chi_res:
                    st.write(
                        {
                            get_text("chi2_label"): chi_res["chi2"],
                            get_text("p_label"): chi_res["p"],
                            get_text("df_label"): chi_res["dof"],
                        }
                    )
                    st.markdown(f"**{get_text('observed_title')}**")
                    st.dataframe(chi_res["observed"])
                    st.markdown(f"**{get_text('expected_title')}**")
                    st.dataframe(chi_res["expected"])
                else:
                    st.info(get_text("not_enough_chi"))
        else:
            st.info(get_text("no_categorical"))

        st.markdown(f"#### {get_text('corr_matrix_title')}")
        if numeric_cols:
            corr_mat = df[numeric_cols].corr(method="pearson")
            st.dataframe(corr_mat.style.background_gradient(cmap="coolwarm"))
            st.caption(get_text("matrix_note"))
        else:
            st.info(get_text("no_numeric"))

    # Tab Teks
    with tab_text:
        st.markdown(f"### {get_text('tab_text')}")
        st.caption(get_text("text_processing_note"))

        if text_cols:
            t_col = st.selectbox(get_text("select_text_col"), text_cols)
            tokens_all, counter = preprocess_text_series(df[t_col])
            st.markdown(f"#### {get_text('text_preview_title')}")
            st.write(tokens_all[:50])

            st.markdown(f"#### {get_text('top_words_title')}")
            top_words = counter.most_common(10)
            top_df = pd.DataFrame(top_words, columns=["word", "count"])
            st.dataframe(top_df)
        else:
            st.info(get_text("no_text"))

    st.markdown("</div>", unsafe_allow_html=True)

    # PDF export box
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown(f"### {get_text('pdf_title')}")
    lang = st.session_state.get("language", "EN")

    if st.button(get_text("pdf_button")):
        with st.spinner(get_text("loading_pdf")):
            pdf_buffer = build_survey_report_pdf(
                df, numeric_cols, cat_cols, text_cols, lang
            )
        st.success(get_text("pdf_ready"))
        st.download_button(
            label=get_text("pdf_download"),
            data=pdf_buffer,
            file_name=TEXTS.get(lang, TEXTS["EN"]).get("pdf_filename", "report.pdf"),
            mime="application/pdf",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='text-align:center;color:gray;margin-top:1rem;'>{get_text('app_footer')}</p>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
