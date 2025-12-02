import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, chi2_contingency, normaltest
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter
import time
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import itertools

# ------------------------------------------------------------
# Multi-language texts
# ------------------------------------------------------------
TEXTS = {
    "EN": {
        "title": "Survey Data Analysis",
        "subtitle": "Upload your survey dataset to explore statistics, visualizations, and exportable reports.",
        "dark_mode": "Dark mode",
        "language": "Language",
        "upload_label": "Upload CSV or Excel file",
        "no_file": "Please upload a CSV or Excel file to begin.",
        "preview_title": "Data Preview",
        "summary_title": "Dataset Summary",
        "rows": "Rows",
        "cols": "Columns",
        "num_cols": "Numeric columns",
        "cat_cols": "Categorical columns",
        "text_cols": "Text columns",
        "tab_desc": "Descriptive Stats",
        "tab_visual": "Visualizations",
        "tab_corr": "Correlations & Tests",
        "tab_text": "Text Processing",
        "select_numeric_col": "Select a numeric column",
        "select_numeric_col_x": "Select numeric column X",
        "select_numeric_col_y": "Select numeric column Y",
        "select_cat_col1": "Select categorical column 1",
        "select_cat_col2": "Select categorical column 2",
        "select_cat_col": "Select a categorical column",
        "select_text_col": "Select a text column",
        "desc_stats_title": "Descriptive statistics",
        "normaltest_title": "Normality test (D’Agostino-Pearson)",
        "normaltest_not_enough": "Not enough data (need at least 8 valid observations).",
        "statistic": "Statistic",
        "pvalue": "p-value",
        "alpha_note": "Using α = 0.05",
        "normal_interpret": "Data likely follows a normal distribution (fail to reject H₀).",
        "not_normal_interpret": "Data does not appear normally distributed (reject H₀).",
        "hist_title": "Histogram",
        "box_title": "Boxplot",
        "freq_table_title": "Frequency table",
        "count": "Count",
        "percent": "Percent",
        "visual_hist_title": "Histogram for selected numeric column",
        "visual_box_title": "Boxplot for selected numeric column",
        "scatter_title": "Scatter plot",
        "scatter_x": "X axis",
        "scatter_y": "Y axis",
        "bar_title": "Bar chart (Top 20 categories)",
        "corr_matrix_title": "Pearson correlation matrix",
        "pearson_title": "Pearson correlation",
        "spearman_title": "Spearman correlation",
        "r_label": "Correlation (r)",
        "strength": "Strength",
        "direction": "Direction",
        "p_label": "p-value",
        "strength_very_weak": "Very weak",
        "strength_weak": "Weak",
        "strength_moderate": "Moderate",
        "strength_strong": "Strong",
        "strength_very_strong": "Very strong",
        "direction_positive": "Positive",
        "direction_negative": "Negative",
        "direction_none": "None",
        "chi_square_title": "Chi-square test of independence",
        "chi2_label": "Chi-square (χ²)",
        "df_label": "Degrees of freedom",
        "expected_title": "Expected frequencies",
        "observed_title": "Observed frequencies",
        "text_preview_title": "Example tokens",
        "top_words_title": "Top 10 words",
        "pdf_title": "Export PDF Report",
        "pdf_button": "Generate PDF report",
        "pdf_ready": "PDF generated. Use the button below to download.",
        "pdf_download": "Download PDF report",
        "pdf_filename": "survey_report_en.pdf",
        "no_numeric": "No numeric columns detected.",
        "no_categorical": "No categorical columns detected.",
        "no_text": "No text columns detected.",
        "loading_pdf": "Building PDF report, please wait...",
        "scatter_note": "Scatter plots use rows where both selected columns are non-missing.",
        "matrix_note": "Correlation matrix computed using Pearson method for all numeric columns.",
        "text_processing_note": "Text is converted to lowercase, punctuation removed, tokenized by space, and English stopwords removed.",
        "app_footer": "Built with Streamlit · Analysis helper"
    },
    "ID": {
        "title": "Analisis Data Survei",
        "subtitle": "Unggah data survei Anda untuk melihat statistik, visualisasi, dan laporan yang dapat diekspor.",
        "dark_mode": "Mode gelap",
        "language": "Bahasa",
        "upload_label": "Unggah file CSV atau Excel",
        "no_file": "Silakan unggah file CSV atau Excel terlebih dahulu.",
        "preview_title": "Pratinjau Data",
        "summary_title": "Ringkasan Dataset",
        "rows": "Jumlah baris",
        "cols": "Jumlah kolom",
        "num_cols": "Kolom numerik",
        "cat_cols": "Kolom kategorikal",
        "text_cols": "Kolom teks",
        "tab_desc": "Statistik Deskriptif",
        "tab_visual": "Visualisasi",
        "tab_corr": "Korelasi & Uji",
        "tab_text": "Pemrosesan Teks",
        "select_numeric_col": "Pilih kolom numerik",
        "select_numeric_col_x": "Pilih kolom numerik X",
        "select_numeric_col_y": "Pilih kolom numerik Y",
        "select_cat_col1": "Pilih kolom kategorikal 1",
        "select_cat_col2": "Pilih kolom kategorikal 2",
        "select_cat_col": "Pilih kolom kategorikal",
        "select_text_col": "Pilih kolom teks",
        "desc_stats_title": "Statistik deskriptif",
        "normaltest_title": "Uji normalitas (D’Agostino-Pearson)",
        "normaltest_not_enough": "Data tidak cukup (minimal 8 observasi valid).",
        "statistic": "Statistik",
        "pvalue": "p-value",
        "alpha_note": "Menggunakan α = 0,05",
        "normal_interpret": "Data kemungkinan berdistribusi normal (gagal menolak H₀).",
        "not_normal_interpret": "Data tampaknya tidak berdistribusi normal (menolak H₀).",
        "hist_title": "Histogram",
        "box_title": "Boxplot",
        "freq_table_title": "Tabel frekuensi",
        "count": "Frekuensi",
        "percent": "Persen",
        "visual_hist_title": "Histogram untuk kolom numerik terpilih",
        "visual_box_title": "Boxplot untuk kolom numerik terpilih",
        "scatter_title": "Scatter plot",
        "scatter_x": "Sumbu X",
        "scatter_y": "Sumbu Y",
        "bar_title": "Diagram batang (Top 20 kategori)",
        "corr_matrix_title": "Matriks korelasi Pearson",
        "pearson_title": "Korelasi Pearson",
        "spearman_title": "Korelasi Spearman",
        "r_label": "Korelasi (r)",
        "strength": "Kekuatan",
        "direction": "Arah",
        "p_label": "p-value",
        "strength_very_weak": "Sangat lemah",
        "strength_weak": "Lemah",
        "strength_moderate": "Sedang",
        "strength_strong": "Kuat",
        "strength_very_strong": "Sangat kuat",
        "direction_positive": "Positif",
        "direction_negative": "Negatif",
        "direction_none": "Tidak ada",
        "chi_square_title": "Uji Chi-square keterkaitan",
        "chi2_label": "Chi-square (χ²)",
        "df_label": "Derajat bebas",
        "expected_title": "Frekuensi harapan",
        "observed_title": "Frekuensi teramati",
        "text_preview_title": "Contoh token",
        "top_words_title": "10 kata teratas",
        "pdf_title": "Ekspor Laporan PDF",
        "pdf_button": "Buat laporan PDF",
        "pdf_ready": "PDF berhasil dibuat. Gunakan tombol di bawah untuk mengunduh.",
        "pdf_download": "Unduh laporan PDF",
        "pdf_filename": "laporan_survei_id.pdf",
        "no_numeric": "Tidak ada kolom numerik yang terdeteksi.",
        "no_categorical": "Tidak ada kolom kategorikal yang terdeteksi.",
        "no_text": "Tidak ada kolom teks yang terdeteksi.",
        "loading_pdf": "Sedang membuat laporan PDF, harap tunggu...",
        "scatter_note": "Scatter plot menggunakan baris yang memiliki data lengkap di kedua kolom.",
        "matrix_note": "Matriks korelasi dihitung dengan metode Pearson untuk semua kolom numerik.",
        "text_processing_note": "Teks diubah ke huruf kecil, tanda baca dihapus, dipisah berdasarkan spasi, dan stopwords bahasa Inggris dihapus.",
        "app_footer": "Dibangun dengan Streamlit · Asisten analisis"
    },
    "JP": {
        "title": "アンケートデータ分析",
        "subtitle": "アンケートデータをアップロードして、統計・可視化・PDFレポートを確認できます。",
        "dark_mode": "ダークモード",
        "language": "言語",
        "upload_label": "CSV または Excel ファイルをアップロード",
        "no_file": "最初に CSV または Excel ファイルをアップロードしてください。",
        "preview_title": "データプレビュー",
        "summary_title": "データセット概要",
        "rows": "行数",
        "cols": "列数",
        "num_cols": "数値列",
        "cat_cols": "カテゴリ列",
        "text_cols": "テキスト列",
        "tab_desc": "記述統計",
        "tab_visual": "可視化",
        "tab_corr": "相関と検定",
        "tab_text": "テキスト処理",
        "select_numeric_col": "数値列を選択",
        "select_numeric_col_x": "数値列 X を選択",
        "select_numeric_col_y": "数値列 Y を選択",
        "select_cat_col1": "カテゴリ列 1 を選択",
        "select_cat_col2": "カテゴリ列 2 を選択",
        "select_cat_col": "カテゴリ列を選択",
        "select_text_col": "テキスト列を選択",
        "desc_stats_title": "記述統計量",
        "normaltest_title": "正規性検定（D’Agostino-Pearson）",
        "normaltest_not_enough": "データ数が不足しています（8 以上の有効データが必要）。",
        "statistic": "統計量",
        "pvalue": "p値",
        "alpha_note": "有意水準 α = 0.05",
        "normal_interpret": "データは正規分布に従う可能性が高い（帰無仮説を棄却しない）。",
        "not_normal_interpret": "データは正規分布に従わない可能性が高い（帰無仮説を棄却）。",
        "hist_title": "ヒストグラム",
        "box_title": "箱ひげ図",
        "freq_table_title": "度数表",
        "count": "度数",
        "percent": "割合",
        "visual_hist_title": "選択した数値列のヒストグラム",
        "visual_box_title": "選択した数値列の箱ひげ図",
        "scatter_title": "散布図",
        "scatter_x": "X 軸",
        "scatter_y": "Y 軸",
        "bar_title": "棒グラフ（上位 20 カテゴリ）",
        "corr_matrix_title": "Pearson 相関行列",
        "pearson_title": "Pearson 相関",
        "spearman_title": "Spearman 相関",
        "r_label": "相関係数 (r)",
        "strength": "強さ",
        "direction": "方向",
        "p_label": "p値",
        "strength_very_weak": "ごく弱い",
        "strength_weak": "弱い",
        "strength_moderate": "中程度",
        "strength_strong": "強い",
        "strength_very_strong": "非常に強い",
        "direction_positive": "正の相関",
        "direction_negative": "負の相関",
        "direction_none": "相関なし",
        "chi_square_title": "カイ二乗独立性検定",
        "chi2_label": "カイ二乗値 (χ²)",
        "df_label": "自由度",
        "expected_title": "期待度数",
        "observed_title": "観測度数",
        "text_preview_title": "トークン例",
        "top_words_title": "上位 10 語",
        "pdf_title": "PDF レポート出力",
        "pdf_button": "PDF レポートを作成",
        "pdf_ready": "PDF が生成されました。下のボタンからダウンロードできます。",
        "pdf_download": "PDF レポートをダウンロード",
        "pdf_filename": "survey_report_jp.pdf",
        "no_numeric": "数値列が検出されませんでした。",
        "no_categorical": "カテゴリ列が検出されませんでした。",
        "no_text": "テキスト列が検出されませんでした。",
        "loading_pdf": "PDF レポートを作成中です。しばらくお待ちください…",
        "scatter_note": "散布図は、両方の列に欠損のない行のみを使用します。",
        "matrix_note": "相関行列は、すべての数値列に対して Pearson 法で計算されます。",
        "text_processing_note": "テキストは小文字化し、句読点を削除し、スペースで分割して、英語のストップワードを除去します。",
        "app_footer": "Streamlit で構築 · 分析アシスタント"
    },
    "KR": {
        "title": "설문 데이터 분석",
        "subtitle": "설문 데이터를 업로드하여 통계, 시각화 및 PDF 보고서를 확인하세요.",
        "dark_mode": "다크 모드",
        "language": "언어",
        "upload_label": "CSV 또는 Excel 파일 업로드",
        "no_file": "먼저 CSV 또는 Excel 파일을 업로드하세요.",
        "preview_title": "데이터 미리보기",
        "summary_title": "데이터셋 요약",
        "rows": "행 수",
        "cols": "열 수",
        "num_cols": "수치형 열",
        "cat_cols": "범주형 열",
        "text_cols": "텍스트 열",
        "tab_desc": "기술 통계",
        "tab_visual": "시각화",
        "tab_corr": "상관 및 검정",
        "tab_text": "텍스트 처리",
        "select_numeric_col": "수치형 열 선택",
        "select_numeric_col_x": "수치형 열 X 선택",
        "select_numeric_col_y": "수치형 열 Y 선택",
        "select_cat_col1": "범주형 열 1 선택",
        "select_cat_col2": "범주형 열 2 선택",
        "select_cat_col": "범주형 열 선택",
        "select_text_col": "텍스트 열 선택",
        "desc_stats_title": "기술 통계량",
        "normaltest_title": "정규성 검정 (D’Agostino-Pearson)",
        "normaltest_not_enough": "데이터가 부족합니다 (유효 데이터 8개 이상 필요).",
        "statistic": "검정 통계량",
        "pvalue": "p-값",
        "alpha_note": "유의수준 α = 0.05",
        "normal_interpret": "데이터가 정규분포를 따른다고 볼 수 있습니다 (귀무가설 기각 불가).",
        "not_normal_interpret": "데이터가 정규분포를 따르지 않는다고 볼 수 있습니다 (귀무가설 기각).",
        "hist_title": "히스토그램",
        "box_title": "박스플롯",
        "freq_table_title": "빈도표",
        "count": "빈도",
        "percent": "비율",
        "visual_hist_title": "선택한 수치형 열의 히스토그램",
        "visual_box_title": "선택한 수치형 열의 박스플롯",
        "scatter_title": "산점도",
        "scatter_x": "X축",
        "scatter_y": "Y축",
        "bar_title": "막대 그래프 (상위 20개 범주)",
        "corr_matrix_title": "피어슨 상관 행렬",
        "pearson_title": "피어슨 상관",
        "spearman_title": "스피어만 상관",
        "r_label": "상관계수 (r)",
        "strength": "강도",
        "direction": "방향",
        "p_label": "p-값",
        "strength_very_weak": "매우 약함",
        "strength_weak": "약함",
        "strength_moderate": "보통",
        "strength_strong": "강함",
        "strength_very_strong": "매우 강함",
        "direction_positive": "양의 상관",
        "direction_negative": "음의 상관",
        "direction_none": "상관 없음",
        "chi_square_title": "카이제곱 독립성 검정",
        "chi2_label": "카이제곱 (χ²)",
        "df_label": "자유도",
        "expected_title": "기대 빈도",
        "observed_title": "관측 빈도",
        "text_preview_title": "토큰 예시",
        "top_words_title": "상위 10개 단어",
        "pdf_title": "PDF 보고서 내보내기",
        "pdf_button": "PDF 보고서 생성",
        "pdf_ready": "PDF가 생성되었습니다. 아래 버튼으로 다운로드하세요.",
        "pdf_download": "PDF 보고서 다운로드",
        "pdf_filename": "survey_report_kr.pdf",
        "no_numeric": "수치형 열이 없습니다.",
        "no_categorical": "범주형 열이 없습니다.",
        "no_text": "텍스트 열이 없습니다.",
        "loading_pdf": "PDF 보고서를 생성하는 중입니다. 잠시만 기다려 주세요...",
        "scatter_note": "산점도는 두 열 모두 결측치가 없는 행만 사용합니다.",
        "matrix_note": "상관 행렬은 모든 수치형 열에 대해 피어슨 방법으로 계산됩니다.",
        "text_processing_note": "텍스트는 소문자로 변환되고, 구두점 제거 후 공백 기준 토큰화, 영어 불용어 제거를 수행합니다.",
        "app_footer": "Streamlit으로 제작 · 분석 도우미"
    },
    "CN": {
        "title": "问卷数据分析",
        "subtitle": "上传问卷数据，查看统计结果、可视化图表，并导出 PDF 报告。",
        "dark_mode": "深色模式",
        "language": "语言",
        "upload_label": "上传 CSV 或 Excel 文件",
        "no_file": "请先上传 CSV 或 Excel 文件。",
        "preview_title": "数据预览",
        "summary_title": "数据集概览",
        "rows": "行数",
        "cols": "列数",
        "num_cols": "数值列",
        "cat_cols": "分类列",
        "text_cols": "文本列",
        "tab_desc": "描述性统计",
        "tab_visual": "可视化",
        "tab_corr": "相关与检验",
        "tab_text": "文本处理",
        "select_numeric_col": "选择数值列",
        "select_numeric_col_x": "选择数值列 X",
        "select_numeric_col_y": "选择数值列 Y",
        "select_cat_col1": "选择分类列 1",
        "select_cat_col2": "选择分类列 2",
        "select_cat_col": "选择分类列",
        "select_text_col": "选择文本列",
        "desc_stats_title": "描述性统计量",
        "normaltest_title": "正态性检验（D’Agostino-Pearson）",
        "normaltest_not_enough": "数据不足（至少需要 8 个有效观测）。",
        "statistic": "统计量",
        "pvalue": "p 值",
        "alpha_note": "显著性水平 α = 0.05",
        "normal_interpret": "数据可能服从正态分布（不能拒绝原假设）。",
        "not_normal_interpret": "数据可能不服从正态分布（拒绝原假设）。",
        "hist_title": "直方图",
        "box_title": "箱线图",
        "freq_table_title": "频数表",
        "count": "频数",
        "percent": "百分比",
        "visual_hist_title": "所选数值列的直方图",
        "visual_box_title": "所选数值列的箱线图",
        "scatter_title": "散点图",
        "scatter_x": "X 轴",
        "scatter_y": "Y 轴",
        "bar_title": "柱状图（前 20 类别）",
        "corr_matrix_title": "Pearson 相关矩阵",
        "pearson_title": "Pearson 相关",
        "spearman_title": "Spearman 相关",
        "r_label": "相关系数 (r)",
        "strength": "强度",
        "direction": "方向",
        "p_label": "p 值",
        "strength_very_weak": "极弱",
        "strength_weak": "较弱",
        "strength_moderate": "中等",
        "strength_strong": "较强",
        "strength_very_strong": "极强",
        "direction_positive": "正相关",
        "direction_negative": "负相关",
        "direction_none": "无相关",
        "chi_square_title": "卡方独立性检验",
        "chi2_label": "卡方值 (χ²)",
        "df_label": "自由度",
        "expected_title": "期望频数",
        "observed_title": "观测频数",
        "text_preview_title": "示例分词",
        "top_words_title": "前 10 高频词",
        "pdf_title": "导出 PDF 报告",
        "pdf_button": "生成 PDF 报告",
        "pdf_ready": "PDF 已生成，可通过下方按钮下载。",
        "pdf_download": "下载 PDF 报告",
        "pdf_filename": "survey_report_cn.pdf",
        "no_numeric": "未检测到数值列。",
        "no_categorical": "未检测到分类列。",
        "no_text": "未检测到文本列。",
        "loading_pdf": "正在生成 PDF 报告，请稍候…",
        "scatter_note": "散点图仅使用两列都非缺失的行。",
        "matrix_note": "相关矩阵使用所有数值列，采用 Pearson 方法计算。",
        "text_processing_note": "文本将转为小写，去除标点，按空格分词，并去除英文停用词。",
        "app_footer": "基于 Streamlit · 数据分析助手"
    },
}


def get_text(key: str) -> str:
    lang = st.session_state.get("language", "EN")
    if lang in TEXTS and key in TEXTS[lang]:
        return TEXTS[lang][key]
    if key in TEXTS["EN"]:
        return TEXTS["EN"][key]
    return key


# ------------------------------------------------------------
# NLTK stopwords initialization
# ------------------------------------------------------------
try:
    STOPWORDS_EN = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS_EN = set(stopwords.words("english"))


# ------------------------------------------------------------
# Theme helper
# ------------------------------------------------------------
def apply_theme():
    if st.session_state.get("dark_mode", False):
        sns.set_style("darkgrid")
        plt.style.use("dark_background")
    else:
        sns.set_style("whitegrid")
        plt.style.use("default")


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_data(file) -> pd.DataFrame:
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload CSV, XLS, or XLSX.")
        return None


# ------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------
def preprocess_text_series(series: pd.Series):
    tokens_all = []
    translator = str.maketrans("", "", string.punctuation)
    for val in series.dropna():
        text = str(val).lower()
        text = text.translate(translator)
        for tok in text.split():
            if tok and tok not in STOPWORDS_EN:
                tokens_all.append(tok)
    counter = Counter(tokens_all)
    return tokens_all, counter


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------
def descriptive_stats(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    desc = {
        "mean": s.mean(),
        "median": s.median(),
        "mode": s.mode().iloc[0] if not s.mode().empty else np.nan,
        "min": s.min(),
        "max": s.max(),
        "std": s.std(ddof=1),
        "count": s.count(),
    }
    if len(s) >= 8:
        stat, p = normaltest(s)
        desc["normaltest_stat"] = stat
        desc["normaltest_p"] = p
    else:
        desc["normaltest_stat"] = None
        desc["normaltest_p"] = None
    return desc


# ------------------------------------------------------------
# Frequency tables
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Visualization helper
# ------------------------------------------------------------
def visualize_data(df: pd.DataFrame, numeric_col: str = None, cat_col: str = None):
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


# ------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------
def _interpret_strength(r: float) -> str:
    ar = abs(r)
    if ar < 0.2:
        return get_text("strength_very_weak")
    elif ar < 0.4:
        return get_text("strength_weak")
    elif ar < 0.6:
        return get_text("strength_moderate")
    elif ar < 0.8:
        return get_text("strength_strong")
    else:
        return get_text("strength_very_strong")


def _interpret_direction(r: float) -> str:
    if r > 0:
        return get_text("direction_positive")
    elif r < 0:
        return get_text("direction_negative")
    else:
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


# ------------------------------------------------------------
# Chi-square test
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# PDF report builder
# ------------------------------------------------------------
def build_survey_report_pdf(
    df: pd.DataFrame, numeric_cols, cat_cols, text_cols, language: str
) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    def draw_line(text, font="Helvetica", size=10, new_page_if_needed=True):
        nonlocal y
        c.setFont(font, size)
        if new_page_if_needed and y < margin + 60:
            c.showPage()
            y = height - margin
            c.setFont(font, size)
        c.drawString(margin, y, text)
        y -= size + 4

    # Title
    c.setTitle(TEXTS.get(language, TEXTS["EN"])["title"])
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, TEXTS.get(language, TEXTS["EN"])["title"])
    y -= 24
    c.setFont("Helvetica", 10)
    draw_line(time.strftime("Generated on %Y-%m-%d %H:%M:%S"), new_page_if_needed=False)
    y -= 6

    # Metadata
    draw_line("-" * 80)
    draw_line("Dataset metadata", "Helvetica-Bold", 12)
    draw_line(f"Rows: {df.shape[0]}")
    draw_line(f"Columns: {df.shape[1]}")
    draw_line(f"Numeric columns: {len(numeric_cols)}")
    draw_line(f"Categorical columns: {len(cat_cols)}")
    draw_line(f"Text columns: {len(text_cols)}")

    # Numeric columns section
    if numeric_cols:
        draw_line("-" * 80)
        draw_line("Numeric columns statistics", "Helvetica-Bold", 12)

        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            desc = descriptive_stats(s)
            draw_line(f"Column: {col}", "Helvetica-Bold", 11)
            draw_line(
                f"  Count: {desc['count']}  Mean: {desc['mean']:.4f}  Median: {desc['median']:.4f}"
            )
            draw_line(
                f"  Mode: {desc['mode']:.4f}  Min: {desc['min']:.4f}  Max: {desc['max']:.4f}  Std: {desc['std']:.4f}"
            )
            if desc["normaltest_stat"] is not None:
                draw_line(
                    f"  Normaltest stat: {desc['normaltest_stat']:.4f}, p-value: {desc['normaltest_p']:.4g}"
                )
            else:
                draw_line("  Normaltest: not enough data (n < 8)")

            # Histogram
            apply_theme()
            fig_h, ax_h = plt.subplots()
            sns.histplot(s, kde=True, ax=ax_h)
            ax_h.set_title(f"Histogram - {col}")
            img_buffer = BytesIO()
            fig_h.savefig(img_buffer, format="png", bbox_inches="tight")
            plt.close(fig_h)
            img_buffer.seek(0)
            img = ImageReader(img_buffer)

            if y < margin + 220:
                c.showPage()
                y = height - margin

            c.drawImage(img, margin, y - 180, width=width - 2 * margin, height=180, preserveAspectRatio=True, mask="auto")
            y -= 190

            # Boxplot
            fig_b, ax_b = plt.subplots()
            sns.boxplot(x=s, ax=ax_b)
            ax_b.set_title(f"Boxplot - {col}")
            img_buffer2 = BytesIO()
            fig_b.savefig(img_buffer2, format="png", bbox_inches="tight")
            plt.close(fig_b)
            img_buffer2.seek(0)
            img2 = ImageReader(img_buffer2)

            if y < margin + 220:
                c.showPage()
                y = height - margin
            c.drawImage(img2, margin, y - 150, width=width - 2 * margin, height=150, preserveAspectRatio=True, mask="auto")
            y -= 160

    # Scatter plots for numeric pairs
    if len(numeric_cols) >= 2:
        draw_line("-" * 80)
        draw_line("Scatter plots (numeric pairs)", "Helvetica-Bold", 12)
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

            if y < margin + 220:
                c.showPage()
                y = height - margin
            c.drawImage(img_s, margin, y - 180, width=width - 2 * margin, height=180, preserveAspectRatio=True, mask="auto")
            y -= 190

    # Correlation matrix
    if numeric_cols:
        draw_line("-" * 80)
        draw_line("Pearson correlation matrix", "Helvetica-Bold", 12)
        corr = df[numeric_cols].corr(method="pearson")
        cols_list = list(corr.columns)
        # Header
        header = "      " + "  ".join([str(c)[:6].ljust(6) for c in cols_list])
        draw_line(header)
        for r in cols_list:
            row_vals = [f"{corr.loc[r, c]:.2f}" for c in cols_list]
            row_str = str(r)[:6].ljust(6) + "  " + "  ".join(v.ljust(6) for v in row_vals)
            draw_line(row_str)

    # Categorical frequency tables
    if cat_cols:
        draw_line("-" * 80)
        draw_line("Categorical columns (Top 10 categories)", "Helvetica-Bold", 12)
        for col in cat_cols:
            draw_line(f"Column: {col}", "Helvetica-Bold", 11)
            vc = df[col].value_counts(dropna=False).head(10)
            total = vc.sum()
            for idx, val in vc.items():
                label = str(idx)
                perc = val / total * 100 if total > 0 else 0
                draw_line(f"  {label[:40]}: {val} ({perc:.1f}%)")

    # Text processing summary
    if text_cols:
        draw_line("-" * 80)
        draw_line("Text processing summary (Top 10 words per column)", "Helvetica-Bold", 12)
        for col in text_cols:
            draw_line(f"Text column: {col}", "Helvetica-Bold", 11)
            _, counter = preprocess_text_series(df[col])
            for word, cnt in counter.most_common(10):
                draw_line(f"  {word}: {cnt}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ------------------------------------------------------------
# Main Streamlit app
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Survey Data Analysis", layout="wide")

    if "language" not in st.session_state:
        st.session_state["language"] = "EN"
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False

    with st.sidebar:
        st.markdown("### Settings")
        st.session_state["dark_mode"] = st.toggle(get_text("dark_mode"), value=st.session_state["dark_mode"])
        st.session_state["language"] = st.radio(
            get_text("language"),
            options=["EN", "ID", "JP", "KR", "CN"],
            horizontal=False,
            index=["EN", "ID", "JP", "KR", "CN"].index(st.session_state["language"]),
        )

    apply_theme()

    st.title(get_text("title"))
    st.caption(get_text("subtitle"))

    uploaded_file = st.file_uploader(get_text("upload_label"), type=["csv", "xls", "xlsx"])

    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)

    if df is None:
        st.info(get_text("no_file"))
        st.markdown(f"<p style='text-align:center;color:gray;'>{get_text('app_footer')}</p>", unsafe_allow_html=True)
        return

    # Detect column types
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    # Heuristic: object/category/bool as categorical/text
    obj_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    cat_cols = []
    text_cols = []
    for col in obj_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 30:
            cat_cols.append(col)
        else:
            text_cols.append(col)

    # Preview & summary
    st.subheader(get_text("preview_title"))
    st.dataframe(df.head(1000))

    st.subheader(get_text("summary_title"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(get_text("rows"), df.shape[0])
    with c2:
        st.metric(get_text("cols"), df.shape[1])
    with c3:
        st.metric(get_text("num_cols"), len(numeric_cols))
    with c4:
        st.metric(get_text("cat_cols"), len(cat_cols))

    if text_cols:
        st.caption(f"{get_text('text_cols')}: {', '.join(text_cols)}")

    # Tabs
    tab_desc, tab_visual, tab_corr, tab_text = st.tabs(
        [get_text("tab_desc"), get_text("tab_visual"), get_text("tab_corr"), get_text("tab_text")]
    )

    # --------------------------------------------------------
    # Descriptive Stats Tab
    # --------------------------------------------------------
    with tab_desc:
        if not numeric_cols and not cat_cols:
            st.warning(get_text("no_numeric") + " " + get_text("no_categorical"))
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {get_text('desc_stats_title')}")
                if numeric_cols:
                    num_col = st.selectbox(get_text("select_numeric_col"), numeric_cols, key="desc_num_col")
                    s = df[num_col]
                    desc = descriptive_stats(s)
                    if desc:
                        st.write(
                            {
                                "mean": desc["mean"],
                                "median": desc["median"],
                                "mode": desc["mode"],
                                "min": desc["min"],
                                "max": desc["max"],
                                "std": desc["std"],
                                "count": desc["count"],
                            }
                        )

                        st.markdown(f"**{get_text('normaltest_title')}**")
                        if desc["normaltest_stat"] is None:
                            st.info(get_text("normaltest_not_enough"))
                        else:
                            st.write(
                                {
                                    get_text("statistic"): desc["normaltest_stat"],
                                    get_text("pvalue"): desc["normaltest_p"],
                                    "α": 0.05,
                                }
                            )
                            if desc["normaltest_p"] < 0.05:
                                st.warning(get_text("not_normal_interpret"))
                            else:
                                st.success(get_text("normal_interpret"))
                    else:
                        st.info("No valid data in the selected column.")
                else:
                    st.warning(get_text("no_numeric"))

            with col2:
                st.markdown(f"#### {get_text('hist_title')} & {get_text('box_title')}")
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

    # --------------------------------------------------------
    # Visualizations Tab
    # --------------------------------------------------------
    with tab_visual:
        st.markdown(f"### {get_text('tab_visual')}")

        # Histogram & boxplot
        st.markdown(f"#### {get_text('visual_hist_title')} / {get_text('visual_box_title')}")
        if numeric_cols:
            v_num_col = st.selectbox(get_text("select_numeric_col"), numeric_cols, key="visual_num_col")
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
        else:
            st.warning(get_text("no_numeric"))

        # Scatter plot
        st.markdown(f"#### {get_text('scatter_title')}")
        if len(numeric_cols) >= 2:
            c3, c4 = st.columns(2)
            with c3:
                x_col = st.selectbox(get_text("select_numeric_col_x"), numeric_cols)
            with c4:
                y_col = st.selectbox(get_text("select_numeric_col_y"), numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

            data = df[[x_col, y_col]].dropna()
            if not data.empty:
                apply_theme()
                fig_sc, ax_sc = plt.subplots()
                sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax_sc)
                ax_sc.set_title(f"{get_text('scatter_title')}: {x_col} vs {y_col}")
                st.pyplot(fig_sc)
                plt.close(fig_sc)
                st.caption(get_text("scatter_note"))
            else:
                st.info("Not enough complete data for scatter plot.")
        else:
            st.info(get_text("no_numeric"))

        # Bar chart for categorical
        st.markdown(f"#### {get_text('bar_title')}")
        if cat_cols:
            b_cat_col = st.selectbox(get_text("select_cat_col"), cat_cols, key="bar_cat_col")
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

    # --------------------------------------------------------
    # Correlations & Tests Tab
    # --------------------------------------------------------
    with tab_corr:
        st.markdown(f"### {get_text('tab_corr')}")

        # Pearson & Spearman
        st.markdown(f"#### {get_text('pearson_title')} & {get_text('spearman_title')}")
        if len(numeric_cols) >= 2:
            c5, c6 = st.columns(2)
            with c5:
                corr_x = st.selectbox(get_text("select_numeric_col_x"), numeric_cols, key="corr_x")
            with c6:
                corr_y = st.selectbox(get_text("select_numeric_col_y"), numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="corr_y")
            if corr_x == corr_y:
                st.warning("Select two different numeric columns.")
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
                else:
                    st.info("Not enough data for correlation analysis.")
        else:
            st.info(get_text("no_numeric"))

        # Chi-square
        st.markdown(f"#### {get_text('chi_square_title')}")
        if len(cat_cols) >= 2:
            c7, c8 = st.columns(2)
            with c7:
                chi_c1 = st.selectbox(get_text("select_cat_col1"), cat_cols, key="chi_c1")
            with c8:
                chi_c2 = st.selectbox(get_text("select_cat_col2"), cat_cols, index=1 if len(cat_cols) > 1 else 0, key="chi_c2")
            if chi_c1 == chi_c2:
                st.warning("Select two different categorical columns.")
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
                    st.info("Not enough data for Chi-square test.")
        else:
            st.info(get_text("no_categorical"))

        # Correlation matrix
        st.markdown(f"#### {get_text('corr_matrix_title')}")
        if numeric_cols:
            corr_mat = df[numeric_cols].corr(method="pearson")
            st.dataframe(corr_mat.style.background_gradient(cmap="coolwarm"))
            st.caption(get_text("matrix_note"))
        else:
            st.info(get_text("no_numeric"))

    # --------------------------------------------------------
    # Text Processing Tab
    # --------------------------------------------------------
    with tab_text:
        st.markdown(f"### {get_text("tab_text")}")
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

    # --------------------------------------------------------
    # PDF Export
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### {get_text('pdf_title')}")

    lang = st.session_state.get("language", "EN")

    if st.button(get_text("pdf_button")):
        with st.spinner(get_text("loading_pdf")):
            pdf_buffer = build_survey_report_pdf(df, numeric_cols, cat_cols, text_cols, lang)
        st.success(get_text("pdf_ready"))
        st.download_button(
            label=get_text("pdf_download"),
            data=pdf_buffer,
            file_name=TEXTS.get(lang, TEXTS["EN"])["pdf_filename"],
            mime="application/pdf",
        )

    st.markdown(f"<p style='text-align:center;color:gray;'>{get_text('app_footer')}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
