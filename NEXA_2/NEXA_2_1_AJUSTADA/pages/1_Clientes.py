
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.estilos import apply_style
from components.ui import *
from services.contexto import get_data, sidebar, footer
from services.pln import keywords
from services.sentimentos_llm import analisar_sentimentos

st.set_page_config(
    page_title="NEXA | Clientes",
    page_icon="◈",
    layout="wide"
)

apply_style()
sidebar()

customers, people, operations = get_data()

page_header(
    "Experiência do Cliente",
    "Explore avaliações, canais de atendimento, produtos e padrões nos feedbacks."
)

with st.expander("Filtros de clientes", expanded=True):
    a, b, c = st.columns(3)
    with a:
        filtered = date_filter(customers, "data", "cli_date")
    with b:
        filtered = multiselect_filter(
            filtered, "canal", "Canal", "cli_channel"
        )
    with c:
        filtered = multiselect_filter(
            filtered, "produto_servico", "Produto ou serviço", "cli_product"
        )

a, b, c, d = st.columns(4)

a.metric("Feedbacks", fmt_int(len(filtered)))
b.metric(
    "Avaliação média",
    fmt_dec(filtered["avaliacao"].mean(), 2)
    if "avaliacao" in filtered else "—"
)
c.metric(
    "Sentimentos negativos",
    f"{fmt_dec(negative_share(filtered))}%"
)
d.metric(
    "Resposta média (h)",
    fmt_dec(filtered["tempo_resposta_h"].mean(), 1)
    if "tempo_resposta_h" in filtered else "—"
)

# Gráficos originais de médias
left, right = st.columns(2)

with left:
    rating_by(
        filtered,
        "canal",
        "avaliacao",
        "Avaliação média por canal"
    )

with right:
    rating_by(
        filtered,
        "produto_servico",
        "avaliacao",
        "Avaliação média por produto/serviço"
    )

left, right = st.columns(2)

with left:
    monthly(
        filtered,
        "data",
        "Evolução de feedbacks por sentimento",
        "sentimento",
        key="clientes"
    )

with right:
    if {"tempo_resposta_h", "avaliacao"}.issubset(filtered.columns):
        sample = filtered.dropna(
            subset=["tempo_resposta_h", "avaliacao"]
        )

        if not sample.empty:
            fig = px.scatter(
                sample,
                x="tempo_resposta_h",
                y="avaliacao",
                color="sentimento",
                opacity=0.55,
                title="Tempo de resposta × avaliação",
                labels={
                    "tempo_resposta_h": "Tempo de resposta (horas)",
                    "avaliacao": "Avaliação"
                },
                color_discrete_map={
                    "Positivo": TEAL,
                    "Neutro": BLUE,
                    "Negativo": RED
                }
            )

            fig.update_yaxes(range=[0.5, 5.5], dtick=1)
            chart(fig)

            st.caption(
                "A avaliação vai de 1 a 5. O mesmo tempo de resposta "
                "pode receber notas diferentes, pois a espera não "
                "explica sozinha a satisfação."
            )
        else:
            st.info("Sem valores numéricos válidos para comparação.")

detail_table(
    filtered,
    [
        "data", "canal", "produto_servico", "avaliacao",
        "tempo_resposta_h", "sentimento", "texto"
    ],
    "Feedbacks detalhados"
)

st.subheader("PLN — temas recorrentes nos feedbacks")
st.caption(
    "Extração de termos com TF-IDF. Não representa causalidade "
    "ou diagnóstico automático."
)

if "texto" in filtered and not filtered.empty:
    term = keywords(filtered["texto"], limit=15)

    if len(term):
        fig = px.bar(
            term.sort_values("Relevância TF-IDF"),
            x="Relevância TF-IDF",
            y="Termo",
            orientation="h",
            title="Termos de maior relevância"
        )
        st.plotly_chart(
            theme(fig, 430),
            use_container_width=True
        )
    else:
        st.info("Textos insuficientes para calcular termos relevantes.")

    
    with st.expander(
        "Análise de sentimentos com IA (LLM)",
        expanded=False
    ):
        st.caption(
            "Classificação de feedbacks com Groq. "
            "Os textos selecionados são enviados à API "
            "para identificar sentimentos e justificativas."
        )

        quantidade = st.slider(
            "Quantidade de feedbacks para analisar",
            min_value=1,
            max_value=min(30, len(filtered)),
            value=min(10, len(filtered)),
            key="clientes_llm_qtd"
        )

        if st.button(
            "Analisar sentimentos com IA",
            key="clientes_llm_botao"
        ):
            try:
                textos = (
                    filtered["texto"]
                    .fillna("")
                    .astype(str)
                    .head(quantidade)
                    .tolist()
                )

                with st.spinner(
                    "A IA está classificando os feedbacks..."
                ):
                    resultado = analisar_sentimentos(textos)

                st.session_state["clientes_llm_resultado"] = resultado

            except Exception as erro:
                st.error(f"Erro na análise: {erro}")

        if "clientes_llm_resultado" in st.session_state:

            resultado = st.session_state[
                "clientes_llm_resultado"
            ]

            st.markdown("#### Distribuição dos sentimentos")

            contagem = (
                resultado["Sentimento"]
                .value_counts()
                .reset_index()
            )

            contagem.columns = [
                "Sentimento", "Quantidade"
            ]

            fig = px.bar(
                contagem,
                x="Sentimento",
                y="Quantidade",
                color="Sentimento",
                text="Quantidade",
                color_discrete_map={
                    "Positivo": "#2DD4BF",
                    "Neutro": "#60A5FA",
                    "Negativo": "#FB7185",
                    "Indeterminado": "#9CA3AF"
                },
                title="Classificação realizada pela LLM"
            )

            st.plotly_chart(
                theme(fig, 350),
                use_container_width=True
            )

            st.markdown("#### Feedbacks analisados")

            st.dataframe(
                resultado,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Classificações geradas por IA sobre uma "
                "amostra de dados sintéticos. "
                "Os resultados podem conter erros."
            )
