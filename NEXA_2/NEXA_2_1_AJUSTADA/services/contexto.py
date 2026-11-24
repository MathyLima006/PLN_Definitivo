"""Inicialização compartilhada das páginas Streamlit."""
import streamlit as st
from services.mongodb import get_uri, load_data, refresh
from services.sentimentos import enrich

def get_data():
    try:
        frames = load_data(get_uri())
    except Exception as exc:
        st.error("Não foi possível consultar o MongoDB Atlas.")
        st.code(str(exc))
        st.info("Verifique o .env, o acesso de rede e as coleções de Nexa_V2.")
        st.stop()
    return enrich(frames["clientes"]), enrich(frames["pessoas"]), frames["operacoes"].copy()

def sidebar():
    st.sidebar.markdown("## ◈ NEXA")
    st.sidebar.caption("CORPORATE INTELLIGENCE · V2")
    st.sidebar.divider()
    if st.sidebar.button("↻ Atualizar dados do Atlas", use_container_width=True):
        refresh()
        st.rerun()
    st.sidebar.caption("MongoDB Atlas · Nexa_V2")
    st.sidebar.caption("Dados sintéticos · Projeto acadêmico")
    st.sidebar.caption("Cache de dados: 5 minutos")

def footer():
    st.divider()
    st.caption("NEXA 2.0 · Dados sintéticos · Indicadores exploratórios · MongoDB Atlas + Python + Streamlit")
