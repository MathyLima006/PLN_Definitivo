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

page_header("Visão Geral", "Indicadores consolidados para identificar riscos, tendências e oportunidades de melhoria.")
total = len(customers) + len(people) + len(operations)
negatives = int((customers["sentimento"] == "Negativo").sum() + (people["sentimento"] == "Negativo").sum())
pending = int(unresolved(operations).sum())
critical_pending = int((unresolved(operations) & high_priority(operations)).sum())
a, b, c, d = st.columns(4)
a.metric("Registros analisados", fmt_int(total))
b.metric("Feedbacks negativos", f"{fmt_dec(percent(negatives, len(customers) + len(people)))}%")
c.metric("Ocorrências em aberto", fmt_int(pending))
d.metric("Alta prioridade em aberto", fmt_int(critical_pending))
st.caption("Percentual negativo calculado sobre feedbacks de clientes e colaboradores; classificação textual heurística.")
left, right = st.columns(2)
with left:
    monthly(customers, "data", "Evolução dos feedbacks de clientes")
with right:
    monthly(operations, "data_abertura", "Evolução das ocorrências")
st.subheader("Pontos de atenção")
if pending:
    insight("Ocorrências não concluídas", f"Existem {fmt_int(pending)} ocorrências com status aberto, pendente ou em andamento. Verifique prioridade e impacto.")
if critical_pending:
    insight("Prioridades elevadas", f"Há {fmt_int(critical_pending)} ocorrências não concluídas com prioridade alta, crítica ou urgente.")
if "avaliacao" in customers:
    insight("Experiência do cliente", f"Avaliação média registrada: {fmt_dec(customers['avaliacao'].mean(), 2)}. Consulte a página Clientes para comparar canais e produtos.")
if "nota_clima" in people:
    insight("Clima organizacional", f"Nota média de clima: {fmt_dec(people['nota_clima'].mean(), 2)}. Consulte a página Pessoas para identificar diferenças entre departamentos.")
st.caption("Alertas são descritivos; não representam previsões ou causalidade.")


footer()
