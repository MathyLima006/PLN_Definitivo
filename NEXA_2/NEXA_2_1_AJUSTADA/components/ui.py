
"""Gráficos e componentes reutilizáveis da NEXA."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.sentimentos import normalize

PANEL = "#121D30"
BLUE = "#60A5FA"
TEAL = "#2DD4BF"
AMBER = "#FBBF24"
RED = "#FB7185"
MUTED = "#9CAFCB"
COLORS = [BLUE, TEAL, "#A78BFA", AMBER, RED, "#38BDF8", "#F472B6"]


def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def fmt_dec(value, digits=1):
    if value is None or pd.isna(value):
        return "—"
    return (
        f"{value:,.{digits}f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def percent(part, total):
    return 100 * part / total if total else 0.0


def theme(fig, height=370):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color="#D9E7FA", size=12),
        height=height,
        margin=dict(l=18, r=18, t=58, b=28),
        legend=dict(orientation="h", y=-0.22, x=0)
    )
    fig.update_xaxes(gridcolor="#28364B", zeroline=False)
    fig.update_yaxes(gridcolor="#28364B", zeroline=False)
    return fig


def chart(fig, height=370):
    st.plotly_chart(
        theme(fig, height),
        use_container_width=True
    )


def page_header(title, description):
    st.markdown(
        f'<div class="nexa-hero"><h1>{title}</h1>'
        f'<p>{description}</p></div>',
        unsafe_allow_html=True
    )


def insight(title, body):
    st.markdown(
        f'<div class="insight"><strong>{title}</strong>'
        f'<div class="small-muted">{body}</div></div>',
        unsafe_allow_html=True
    )


def count_by(df, col, title, top=12):
    if col not in df or df.empty:
        st.info("Sem registros suficientes para esta visualização.")
        return

    counts = (
        df[col]
        .fillna("Não informado")
        .astype(str)
        .value_counts()
        .head(top)
        .sort_values()
    )

    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        text=counts.values,
        title=title,
        labels={"x": "Registros", "y": ""},
        color_discrete_sequence=[BLUE]
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )

    chart(fig, max(340, 65 + 42 * len(counts)))


def monthly(df, date_col, title, subset_col=None, key=None):
    if date_col not in df or df.empty:
        st.info("Não há datas disponíveis.")
        return

    tmp = df.dropna(subset=[date_col]).copy()

    tmp["Mês"] = (
        pd.to_datetime(tmp[date_col], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    tmp = tmp.dropna(subset=["Mês"])

    if tmp.empty:
        st.info("Não há datas válidas.")
        return

    if subset_col and subset_col in tmp.columns:
        options = [
            x for x in ["Negativo", "Neutro", "Positivo"]
            if x in tmp[subset_col].dropna().unique()
        ]

        options += sorted(
            set(tmp[subset_col].dropna().astype(str))
            - set(options)
        )

        default = (
            ["Negativo"]
            if "Negativo" in options
            else options[:1]
        )

        chosen = st.multiselect(
            "Sentimentos exibidos",
            options,
            default=default,
            key=f"{key or date_col}_sentiments"
        )

        mode = st.radio(
            "Visualização",
            ["Percentual mensal", "Quantidade"],
            horizontal=True,
            key=f"{key or date_col}_mode"
        )

        if not chosen:
            st.info("Selecione ao menos um sentimento.")
            return

        series = (
            tmp.groupby(["Mês", subset_col], dropna=False)
            .size()
            .reset_index(name="Registros")
        )

        if mode == "Percentual mensal":
            totals = tmp.groupby("Mês").size().rename("Total")
            series = series.join(totals, on="Mês")
            series["Percentual"] = (
                series["Registros"] / series["Total"] * 100
            )
            y = "Percentual"
        else:
            y = "Registros"

        series = series[
            series[subset_col].isin(chosen)
        ]

        fig = px.line(
            series,
            x="Mês",
            y=y,
            color=subset_col,
            markers=True,
            title=title,
            color_discrete_map={
                "Negativo": RED,
                "Neutro": BLUE,
                "Positivo": TEAL
            },
            hover_data={"Registros": True}
        )

        if mode == "Percentual mensal":
            limite = max(
                10,
                min(100, series[y].max() * 1.2)
            )
            fig.update_yaxes(
                ticksuffix="%",
                range=[0, limite]
            )

    else:
        series = (
            tmp.groupby("Mês")
            .size()
            .reset_index(name="Registros")
        )

        fig = px.line(
            series,
            x="Mês",
            y="Registros",
            markers=True,
            title=title,
            color_discrete_sequence=[BLUE]
        )

    chart(fig)


def rating_by(df, group, score, title, min_n=1):
    if df.empty or group not in df or score not in df:
        st.info("Sem dados suficientes para comparar avaliações.")
        return

    tmp = df[[group, score]].copy()
    tmp[score] = pd.to_numeric(tmp[score], errors="coerce")
    tmp = tmp.dropna(subset=[score])

    summary = (
        tmp.groupby(group, dropna=False)[score]
        .agg(["mean", "count"])
        .reset_index()
    )

    summary = (
        summary[summary["count"] >= min_n]
        .sort_values("mean")
    )

    if summary.empty:
        st.info("Nenhum grupo atingiu o mínimo de respostas.")
        return

    fig = px.bar(
        summary,
        x="mean",
        y=group,
        orientation="h",
        color="mean",
        color_continuous_scale=[RED, AMBER, TEAL],
        title=title,
        hover_data={"count": True, "mean": ":.2f"},
        labels={
            "mean": "Nota média",
            group: ""
        }
    )

    fig.update_layout(coloraxis_showscale=False)

    chart(
        fig,
        max(340, 70 + 42 * len(summary))
    )


def low_rating_by(df, group, score, title, cutoff=2):
    if df.empty or group not in df or score not in df:
        st.info("Sem avaliações disponíveis.")
        return

    tmp = df[[group, score]].copy()
    tmp[score] = pd.to_numeric(tmp[score], errors="coerce")
    tmp = tmp.dropna(subset=[group, score])

    if tmp.empty:
        st.info("Sem avaliações válidas.")
        return

    summary = (
        tmp.groupby(group)[score]
        .agg(
            total="count",
            baixas=lambda x: int((x <= cutoff).sum())
        )
        .reset_index()
    )

    summary["Percentual"] = (
        100 * summary["baixas"] / summary["total"]
    )

    summary = summary.sort_values("Percentual")

    fig = px.bar(
        summary,
        x="Percentual",
        y=group,
        orientation="h",
        title=title,
        hover_data={
            "total": True,
            "baixas": True,
            "Percentual": ":.1f"
        },
        labels={
            "Percentual": f"Avaliações até {cutoff} (%)",
            group: "",
            "total": "Respostas",
            "baixas": "Notas baixas"
        },
        color_discrete_sequence=[RED]
    )

    fig.update_xaxes(range=[0, 100], ticksuffix="%")

    chart(fig, max(340, 70 + 43 * len(summary)))


def rating_distribution(df, group, score, title):
    if df.empty or group not in df or score not in df:
        st.info("Sem avaliações disponíveis.")
        return

    tmp = df[[group, score]].copy()
    tmp[score] = pd.to_numeric(tmp[score], errors="coerce")
    tmp = tmp.dropna(subset=[group, score])

    if tmp.empty:
        st.info("Sem avaliações válidas.")
        return

    tmp["Nota"] = tmp[score].round().astype(int).astype(str)

    counts = (
        tmp.groupby([group, "Nota"])
        .size()
        .rename("Registros")
        .reset_index()
    )

    counts["Percentual"] = (
        counts["Registros"]
        / counts.groupby(group)["Registros"].transform("sum")
        * 100
    )

    fig = px.bar(
        counts,
        x="Percentual",
        y=group,
        color="Nota",
        orientation="h",
        barmode="stack",
        title=title,
        category_orders={
            "Nota": sorted(counts["Nota"].unique(), key=int)
        },
        color_discrete_sequence=[
            RED, AMBER, "#A78BFA", BLUE, TEAL
        ],
        labels={
            "Percentual": "Distribuição das avaliações (%)",
            group: ""
        },
        hover_data={
            "Registros": True,
            "Percentual": ":.1f"
        }
    )

    fig.update_xaxes(range=[0, 100], ticksuffix="%")

    chart(fig, max(370, 95 + 44 * tmp[group].nunique()))


def table(df, cols=None):
    selected = [
        col for col in (cols or df.columns.tolist())
        if col in df.columns
    ]

    st.dataframe(
        df[selected],
        use_container_width=True,
        hide_index=True
    )


def detail_table(
    df,
    cols,
    label="Consultar registros detalhados"
):
    with st.expander(
        f"📋 {label} ({fmt_int(len(df))} registros)",
        expanded=False
    ):
        table(df, cols)

        if not df.empty:
            selected = [
                col for col in cols
                if col in df.columns
            ]

            st.download_button(
                "Baixar registros filtrados (CSV)",
                data=df[selected].to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="nexa_registros_filtrados.csv",
                mime="text/csv"
            )


def date_filter(df, column, key):
    if column not in df or df[column].dropna().empty:
        return df

    valid = df[column].dropna()
    low = valid.min().date()
    high = valid.max().date()

    if low == high:
        return df

    selected = st.date_input(
        "Período",
        value=(low, high),
        min_value=low,
        max_value=high,
        key=key
    )

    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        start = pd.Timestamp(selected[0])
        end = (
            pd.Timestamp(selected[1])
            + pd.Timedelta(days=1)
        )

        return df[
            (df[column] >= start)
            & (df[column] < end)
        ].copy()

    return df


def multiselect_filter(df, column, label, key):
    if column not in df:
        return df

    options = sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selection = st.multiselect(
        label,
        options,
        key=key
    )

    if selection:
        return df[
            df[column].astype(str).isin(selection)
        ].copy()

    return df


def negative_share(df):
    sentiments = df.get(
        "sentimento",
        pd.Series(dtype=str)
    )

    return percent(
        int((sentiments == "Negativo").sum()),
        len(df)
    )


def unresolved(df):
    if "status" not in df:
        return pd.Series(False, index=df.index)

    return (
        df["status"]
        .map(normalize)
        .str.strip()
        .isin({
            "aberto",
            "pendente",
            "em andamento",
            "em aberto"
        })
    )


def high_priority(df):
    if "prioridade" not in df:
        return pd.Series(False, index=df.index)

    return (
        df["prioridade"]
        .map(normalize)
        .str.strip()
        .isin({
            "alta",
            "critica",
            "urgente"
        })
    )
