import os
import streamlit as st
import pandas as pd
import numpy as np
import re, io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Netflix NLP Dashboard", layout="wide")

st.title(" Netflix NLP Dashboard")


DEFAULT_PATH = "netflix_titles.csv"

@st.cache_data(show_spinner=False)
def load_data(path: str):
    df = pd.read_csv(path)
    expected = ["type","title","release_year","listed_in","description","country","date_added","rating","duration","cast","director"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    df["description"] = df["description"].astype(str)
    df = df.replace({"description": {"nan": np.nan}})
    df = df.dropna(subset=["description"])
    if "release_year" in df.columns:
        with np.errstate(invalid="ignore"):
            df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    return df

dataset_path = DEFAULT_PATH if os.path.exists(DEFAULT_PATH) else None
if dataset_path:
    df = load_data(dataset_path)
    st.caption(f"Loaded dataset: **{dataset_path}**  \nRows: {len(df):,}")
else:
    st.warning("Local Netflix dataset not found. Upload your `netflix_titles.csv` below.")
    uploaded = st.file_uploader("Upload netflix_titles.csv", type=["csv"])
    if uploaded:
        df = load_data(uploaded)
        st.caption(f"Loaded from upload. Rows: {len(df):,}")
    else:
        st.stop()

st.sidebar.header("Filters")
types = sorted([t for t in df["type"].dropna().unique().tolist() if isinstance(t, str)])
type_sel = st.sidebar.multiselect("Type", options=types, default=types)
year_min, year_max = int(np.nanmin(df["release_year"])), int(np.nanmax(df["release_year"]))
yr_range = st.sidebar.slider("Release year range", min_value=year_min, max_value=year_max, value=(max(year_min, year_max-30), year_max), step=1)

def split_genres(s):
    if not isinstance(s, str):
        return []
    return [x.strip() for x in s.split(",")]

all_genres = sorted(set(g for lst in df["listed_in"].dropna().apply(split_genres).tolist() for g in lst))
genre_sel = st.sidebar.multiselect("Genres (listed_in)", options=all_genres)

def apply_filters(df):
    m = pd.Series(True, index=df.index)
    if type_sel:
        m &= df["type"].isin(type_sel)
    if yr_range:
        m &= df["release_year"].between(yr_range[0], yr_range[1], inclusive="both")
    if genre_sel:
        m &= df["listed_in"].apply(lambda s: any(g in (s or "") for g in genre_sel))
    return df[m]

df_f = apply_filters(df)
st.subheader("Dataset preview")
st.dataframe(df_f[["type","title","release_year","listed_in","description"]].head(20), use_container_width=True)

st.header("1) Text Cleaning")
st.caption("Simple regex-based cleanup; TF-IDF will also remove English stopwords.")
clean_settings = {
    "lowercase": st.checkbox("Lowercase", value=True),
    "remove_urls": st.checkbox("Remove URLs", value=True),
    "remove_html": st.checkbox("Remove HTML tags", value=True),
    "remove_non_letters": st.checkbox("Keep only letters & spaces", value=True),
    "remove_digits": st.checkbox("Remove digits", value=False),
    "min_token_len": st.number_input("Min token length", min_value=1, max_value=10, value=2, step=1)
}

url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
html_re = re.compile(r"<.*?>", re.IGNORECASE)
non_alpha_re = re.compile(r"[^a-zA-Z\s]")
digits_re = re.compile(r"\d+")
multi_space_re = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = str(s)
    if clean_settings["lowercase"]:
        s = s.lower()
    if clean_settings["remove_urls"]:
        s = url_re.sub(" ", s)
    if clean_settings["remove_html"]:
        s = html_re.sub(" ", s)
    if clean_settings["remove_digits"]:
        s = digits_re.sub(" ", s)
    if clean_settings["remove_non_letters"]:
        s = non_alpha_re.sub(" ", s)
    s = multi_space_re.sub(" ", s).strip()
    if clean_settings["min_token_len"] > 1:
        tokens = [t for t in s.split() if len(t) >= clean_settings["min_token_len"]]
        s = " ".join(tokens)
    return s

@st.cache_data(show_spinner=False)
def add_clean_text(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out["_text_clean"] = out["description"].apply(clean_text)
    out = out[out["_text_clean"].str.len() > 0]
    return out

df_c = add_clean_text(df_f)
st.write("Cleaned rows:", len(df_c))

st.header("2) Word Cloud & Token Frequency")
col_wc, col_hist = st.columns(2)

with col_wc:
    st.subheader("Word Cloud")
    text_blob = " ".join(df_c["_text_clean"].tolist())
    if len(text_blob) > 0:
        wc = WordCloud(width=900, height=500, background_color="white", stopwords=None).generate(text_blob)
        fig = plt.figure(figsize=(8,4.5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No text after cleaning.")

with col_hist:
    st.subheader("Top Terms (by TF-IDF)")
    max_feats = st.slider("Max features for TF-IDF (display)", min_value=1000, max_value=20000, value=5000, step=500)
    n_top = st.slider("Show top N terms", min_value=10, max_value=40, value=20, step=5)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_feats, ngram_range=(1,2), min_df=2)
    X = vectorizer.fit_transform(df_c["_text_clean"])
    means = np.asarray(X.mean(axis=0)).ravel()
    idx = np.argsort(means)[-n_top:][::-1]
    vocab = np.array(vectorizer.get_feature_names_out())
    top_terms = vocab[idx]
    top_vals = means[idx]
    fig2 = plt.figure(figsize=(8,4.5))
    plt.barh(top_terms[::-1], top_vals[::-1])
    plt.title("Top terms by mean TF-IDF")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

st.header("3) Unsupervised Clustering (K-Means)")
k = st.slider("Number of clusters (k)", min_value=3, max_value=12, value=6, step=1)

@st.cache_data(show_spinner=False)
def compute_tfidf(df_in: pd.DataFrame, max_features=20000):
    vec = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(df_in["_text_clean"])
    return vec, X

def run_kmeans(X, k: int, random_state: int = 42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return labels, centers


vec, X_all = compute_tfidf(df_c)
labels, centers = run_kmeans(X_all, k)

df_clusters = df_c.copy()
df_clusters["_cluster"] = labels
feature_names = np.array(vec.get_feature_names_out())

def top_terms_for_cluster(c_idx, top_n=12):
    order = np.argsort(centers[c_idx])[-top_n:][::-1]
    return feature_names[order]

st.subheader("Cluster keywords")
cols = st.columns(min(k, 6))
for i in range(k):
    block = ", ".join(top_terms_for_cluster(i, top_n=10))
    with cols[i % len(cols)]:
        st.markdown(f"**Cluster {i}**  \n{block}")

st.subheader("Titles by cluster")
cluster_pick = st.selectbox("Select a cluster to inspect", options=sorted(df_clusters["_cluster"].unique().tolist()))
subset = df_clusters[df_clusters["_cluster"] == cluster_pick][["title","type","release_year","listed_in","description"]].head(30)
st.dataframe(subset, use_container_width=True)

st.header("4) Similar Titles (Cosine similarity on TF-IDF)")
query = st.text_input("Enter a short description or keywords (e.g., 'serial killer investigation in small town')", "")
n_sim = st.slider("How many similar titles?", 5, 30, 10, 1)

if query:
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, X_all).ravel()
    top_idx = np.argsort(sims)[-n_sim:][::-1]
    sim_df = df_c.iloc[top_idx][["title","type","release_year","listed_in","description"]].copy()
    sim_df["similarity"] = sims[top_idx]
    st.dataframe(sim_df, use_container_width=True)
