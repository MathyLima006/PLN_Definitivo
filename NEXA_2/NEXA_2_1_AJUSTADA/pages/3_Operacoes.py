from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from components.estilos import apply_style
from components.ui import *
from services.contexto import get_data, sidebar, footer
from services.sentimentos import normalize

st.set_page_config(page_title="NEXA | Inteligência Corporativa", page_icon="◈", layout="wide")
apply_style()
sidebar()
customers, people, operations = get_data()

page_header("Eficiência Operacional", "Priorize ocorrências por gravidade, tempo de resolução e impacto nos clientes.")
with st.expander("Filtros operacionais", expanded=True):
    a, b, c, d = st.columns(4)
    with a:
        filtered = date_filter(operations, "data_abertura", "ops_date")
    with b:
        filtered = multiselect_filter(filtered, "departamento", "Departamento", "ops_dept")
    with c:
        filtered = multiselect_filter(filtered, "prioridade", "Prioridade", "ops_priority")
    with d:
        filtered = multiselect_filter(filtered, "status", "Status", "ops_status")
a, b, c, d = st.columns(4)
a.metric("Ocorrências", fmt_int(len(filtered)))
b.metric("Não concluídas", fmt_int(unresolved(filtered).sum()))
c.metric("Prioridade alta/crítica", fmt_int(high_priority(filtered).sum()))
d.metric("Tempo médio de resolução (h)", fmt_dec(filtered["tempo_resolucao_h"].mean()) if "tempo_resolucao_h" in filtered else "—")
left, right = st.columns(2)
with left:
    count_by(filtered, "tipo_ocorrencia", "Ocorrências por tipo")
with right:
    if {"tipo_ocorrencia", "impacto_clientes"}.issubset(filtered.columns) and not filtered.empty:
        tmp = filtered.groupby("tipo_ocorrencia", dropna=False)["impacto_clientes"].sum().sort_values()
        fig = px.bar(x=tmp.values, y=tmp.index, orientation="h", title="Impacto agregado nos clientes por tipo",
                     labels={"x": "Soma dos clientes impactados", "y": ""}, color_discrete_sequence=[RED])
        chart(fig, max(340, 70 + 42 * len(tmp)))
    else:
        st.info("Sem informações de impacto disponíveis.")
left, right = st.columns(2)
with left:
    if {"departamento", "tempo_resolucao_h"}.issubset(filtered.columns) and not filtered.empty:
        tmp = filtered.dropna(subset=["tempo_resolucao_h"]).groupby("departamento")["tempo_resolucao_h"].agg(["mean", "count"]).reset_index().sort_values("mean")
        if not tmp.empty:
            fig = px.bar(tmp, x="mean", y="departamento", orientation="h", hover_data=["count"],
                         title="Tempo médio de resolução por departamento", labels={"mean": "Horas", "departamento": ""},
                         color_discrete_sequence=[TEAL])
            chart(fig)
        else:
            st.info("Sem tempos de resolução válidos.")
with right:
    if {"prioridade", "status"}.issubset(filtered.columns) and not filtered.empty:
        cross = pd.crosstab(filtered["prioridade"].fillna("Não informada"), filtered["status"].fillna("Não informado"))
        fig = px.imshow(cross, text_auto=True, aspect="auto", title="Prioridade × status",
                        color_continuous_scale="Blues", labels=dict(x="Status", y="Prioridade", color="Registros"))
        chart(fig)
monthly(filtered, "data_abertura", "Evolução mensal das ocorrências")
with st.expander("Como interpretar os indicadores operacionais?", expanded=False):
    st.write("Ocorrências por tipo representa frequência; impacto agregado soma os clientes afetados informados em cada ocorrência. Os mesmos clientes podem aparecer em ocorrências diferentes. Quantidades parecidas podem refletir a distribuição do dataset sintético.")
st.caption("Impacto agregado representa a soma dos valores registrados; pode haver sobreposição de clientes entre ocorrências. Tempos médios consideram apenas registros com valores válidos.")
detail_table(filtered, ["data_abertura", "data_resolucao", "departamento", "tipo_ocorrencia", "prioridade", "status", "tempo_resolucao_h", "impacto_clientes"], "Ocorrências detalhadas")


footer()
