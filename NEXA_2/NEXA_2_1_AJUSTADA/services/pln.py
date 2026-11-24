"""PLN: TF-IDF para temas e modelo transformer opcional para sentimento."""
from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from services.sentimentos import normalize

STOPWORDS = set("a o as os um uma uns umas de da do das dos e em no na nos nas para por com que como se ao aos à às é são foi ser ter tem mais menos muito muitos muitas seu sua seus suas eu você ele ela nós eles elas isso este esta esse essa meu minha sobre ainda já não sim também quando onde qual quais quem porque pois ou mas até entre após antes cada todos todas todo toda pela pelo pelas pelos durante sendo sem sua sua".split())

def keywords(texts, limit=15, min_df=2):
    """Termos relevantes por TF-IDF (não implica identificação de causa)."""
    series = pd.Series(texts).dropna().astype(str)
    series = series[series.str.strip().ne("")]
    if len(series) < 2:
        return pd.DataFrame(columns=["Termo", "Relevância TF-IDF"])
    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=sorted({normalize(w) for w in STOPWORDS}),
                                 ngram_range=(1, 2), min_df=min(min_df, len(series)), max_features=1500,
                                 token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{3,}\b")
    try:
        matrix = vectorizer.fit_transform(series.tolist())
    except ValueError:
        return pd.DataFrame(columns=["Termo", "Relevância TF-IDF"])
    scores = matrix.mean(axis=0).A1
    result = pd.DataFrame({"Termo": vectorizer.get_feature_names_out(), "Relevância TF-IDF": scores})
    return result.sort_values("Relevância TF-IDF", ascending=False).head(limit).reset_index(drop=True)

def classify_transformer(texts, batch_size=16):
    """Modelo pré-treinado multilíngue; requer dependências opcionais e download inicial."""
    from transformers import pipeline
    import streamlit as st
    @st.cache_resource(show_spinner="Carregando modelo de sentimentos (primeira vez pode demorar)...")
    def get_pipeline():
        return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                        tokenizer="lxyuan/distilbert-base-multilingual-cased-sentiments-student", device=-1)
    model = get_pipeline()
    labels = {"positive": "Positivo", "negative": "Negativo", "neutral": "Neutro"}
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [str(x)[:1200] if pd.notna(x) else "" for x in texts[i:i+batch_size]]
        preds = model(batch, truncation=True, max_length=512, batch_size=batch_size)
        results.extend([{"Sentimento": labels.get(p["label"].lower(), p["label"]), "Confiança": round(float(p["score"]), 3)} for p in preds])
    return pd.DataFrame(results)
