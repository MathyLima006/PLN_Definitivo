"""Acesso de leitura ao MongoDB; nenhuma credencial é armazenada no repositório."""
from __future__ import annotations
import os
from pathlib import Path
import certifi
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATABASE = os.getenv("MONGODB_DATABASE", "Nexa_V2")
COLLECTIONS = {"clientes": "clientes", "pessoas": "colaboradores", "operacoes": "operacoes"}
DATE_COLUMNS = ("data", "data_abertura", "data_resolucao")
NUMERIC_COLUMNS = ("avaliacao", "tempo_resposta_h", "nota_clima", "tempo_empresa_meses", "tempo_resolucao_h", "impacto_clientes")
REQUIRED = {"clientes": {"texto", "data", "avaliacao"}, "pessoas": {"texto", "data", "nota_clima"}, "operacoes": {"data_abertura", "status", "prioridade"}}

def get_uri():
    try:
        uri = st.secrets.get("MONGODB_URI")
        if uri:
            return str(uri)
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return os.getenv("MONGODB_URI", "")

@st.cache_data(ttl=300, show_spinner="Consultando MongoDB Atlas...")
def load_data(_uri: str, database: str = DATABASE):
    if not _uri:
        raise ValueError("MONGODB_URI ausente: configure .env ou Streamlit secrets.")
    frames = {}
    with MongoClient(_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000, connectTimeoutMS=20000, socketTimeoutMS=20000) as client:
        client.admin.command("ping")
        db = client[database]
        for alias, collection in COLLECTIONS.items():
            docs = list(db[collection].find({}, {"_id": 0}))
            df = pd.DataFrame(docs)
            if df.empty:
                raise ValueError(f"Coleção '{collection}' está vazia ou indisponível.")
            missing = REQUIRED[alias] - set(df.columns)
            if missing:
                raise ValueError(f"Coleção '{collection}' sem colunas necessárias: {sorted(missing)}")
            for col in DATE_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
            for col in NUMERIC_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            frames[alias] = df
    return frames

def refresh():
    load_data.clear()
