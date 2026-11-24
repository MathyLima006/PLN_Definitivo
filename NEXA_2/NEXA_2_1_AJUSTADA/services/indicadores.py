"""Métricas puras e testáveis, sem dependência de Streamlit."""
from __future__ import annotations
import pandas as pd
from services.sentimentos import normalize

def taxa_negativa(df):
    return float(df["sentimento"].eq("Negativo").mean()*100) if len(df) and "sentimento" in df else 0.0

def nao_concluidas(df):
    return df["status"].fillna("").map(normalize).str.strip().isin({"aberto", "pendente", "em andamento", "em aberto"}) if "status" in df else pd.Series(False,index=df.index)

def alta_prioridade(df):
    return df["prioridade"].fillna("").map(normalize).str.strip().isin({"alta", "critica", "urgente"}) if "prioridade" in df else pd.Series(False,index=df.index)

def resumo(clientes, pessoas, operacoes):
    return {"registros": len(clientes)+len(pessoas)+len(operacoes),
            "clientes": len(clientes), "colaboradores": len(pessoas), "ocorrencias": len(operacoes),
            "negativos_clientes_pct": taxa_negativa(clientes),
            "negativos_colaboradores_pct": taxa_negativa(pessoas),
            "pendentes": int(nao_concluidas(operacoes).sum()),
            "criticas_pendentes": int((nao_concluidas(operacoes)&alta_prioridade(operacoes)).sum())}
