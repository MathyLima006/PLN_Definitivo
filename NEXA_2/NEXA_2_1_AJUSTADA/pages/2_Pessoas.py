
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
    page_title="NEXA | Pessoas",
    page_icon="◈",
    layout="wide"
)

apply_style()
sidebar()

customers, people, operations = get_data()

page_header(
    "Inteligência de Pessoas",
    "Acompanhe clima organizacional, temas recorrentes "
    "e diferenças entre departamentos."
)

with st.expander("Filtros de colaboradores", expanded=True):
    a, b, c = st.columns(3)

    with a:
        filtered = date_filter(people, "data", "ppl_date")
    with b:
        filtered = multiselect_filter(
            filtered, "departamento", "Departamento", "ppl_dept"
        )
    with c:
        filtered = multiselect_filter(
            filtered, "categoria_informada", "Categoria", "ppl_category"
        )

a, b, c, d = st.columns(4)

a.metric("Feedbacks", fmt_int(len(filtered)))
b.metric(
    "Nota média de clima",
    fmt_dec(filtered["nota_clima"].mean(), 2)
    if "nota_clima" in filtered else "—"
)
c.metric(
    "Sentimentos negativos",
    f"{fmt_dec(negative_share(filtered))}%"
)
d.metric(
    "Tempo médio de empresa (meses)",
    fmt_dec(filtered["tempo_empresa_meses"].mean())
    if "tempo_empresa_meses" in filtered else "—"
)

left, right = st.columns(2)

with left:
    rating_by(
        filtered,
        "departamento",
        "nota_clima",
        "Clima médio por departamento"
    )

with right:
    count_by(
        filtered,
        "categoria_informada",
        "Categorias mencionadas"
    )

left, right = st.columns(2)

with left:
    monthly(
        filtered,
        "data",
        "Evolução de feedbacks por sentimento",
        "sentimento",
        key="pessoas"
    )

with right:
    if {"tempo_empresa_meses", "nota_clima"}.issubset(filtered.columns):
        tmp = filtered.dropna(
            subset=["tempo_empresa_meses", "nota_clima"]
        ).copy()

        if not tmp.empty:
            tmp["Faixa de tempo"] = pd.cut(
                tmp["tempo_empresa_meses"],
                bins=[-1, 6, 12, 24, 60, float("inf")],
                labels=[
                    "Até 6 meses",
                    "7–12 meses",
                    "1–2 anos",
                    "2–5 anos",
                    "Mais de 5 anos"
                ]
            )

            summary = (
                tmp.groupby("Faixa de tempo", observed=True)["nota_clima"]
                .agg(["mean", "count"])
                .reset_index()
            )

            fig = px.bar(
                summary,
                x="Faixa de tempo",
                y="mean",
                text="count",
                title="Clima médio por tempo de empresa",
                labels={
                    "mean": "Nota média",
                    "Faixa de tempo": "Tempo de empresa"
                },
                color_discrete_sequence=[TEAL]
            )

            fig.update_traces(
                hovertemplate=(
                    "Faixa: %{x}<br>"
                    "Nota média: %{y:.2f}<br>"
                    "Respostas: %{text}<extra></extra>"
                )
            )

            chart(fig)

            st.caption(
                "Os rótulos indicam a quantidade de respostas "
                "em cada faixa de tempo."
            )
        else:
            st.info("Sem dados válidos de tempo de empresa.")

detail_table(
    filtered,
    [
        "data", "departamento", "categoria_informada",
        "nota_clima", "tempo_empresa_meses", "sentimento", "texto"
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
            "Classificação de feedbacks dos colaboradores "
            "com Groq. Os textos selecionados são enviados "
            "à API para análise."
        )

        quantidade = st.slider(
            "Quantidade de feedbacks para analisar",
            min_value=1,
            max_value=min(30, len(filtered)),
            value=min(10, len(filtered)),
            key="pessoas_llm_qtd"
        )

        if st.button(
            "Analisar sentimentos com IA",
            key="pessoas_llm_botao"
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

                st.session_state["pessoas_llm_resultado"] = resultado

            except Exception as erro:
                st.error(f"Erro na análise: {erro}")

        if "pessoas_llm_resultado" in st.session_state:

            resultado = st.session_state[
                "pessoas_llm_resultado"
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
