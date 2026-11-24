from __future__ import annotations

import json
import logging
import threading

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.estilos import apply_style

from components.ui import (
    page_header,
    fmt_int,
    fmt_dec,
    negative_share,
    unresolved,
    high_priority,
    table,
    insight,
)

from services.contexto import (
    get_data,
    sidebar,
    footer,
)

from services.indicadores import resumo

from services.assistente import (
    explain,
    api_key,
    resposta_local,
    validar_mensagem,
    normalizar,
    LimiteAPI,
)

from services.graficos import (
    gerar_grafico_inteligente,
    solicitou_grafico,
)

from services.feedbacks import (
    investigar_feedbacks,
    solicitou_investigacao,
    interpretar_investigacao,
    COLUNAS_FILTRO,
)


logger = logging.getLogger("NEXA")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

LIMITE_HISTORICO = 20
LIMITE_PERGUNTA = 2000

# Complementa as colunas autorizadas para investigação.
COLUNAS_FILTRO.update({
    "categoria_informada",
    "tempo_empresa_meses",
    "tempo_resposta_h",
    "tempo_resolucao_h",
    "impacto_clientes",
})

st.set_page_config(
    page_title="NEXA | Inteligência Corporativa",
    page_icon="◈",
    layout="wide",
)

apply_style()

sidebar()

clientes, pessoas, operacoes = get_data()

page_header(
    "Inteligência NEXA",
    "Transforme dados corporativos em decisões."
)

st.info(
    "Converse com os dados da empresa, investigue "
    "comentários e ocorrências, solicite análises "
    "personalizadas e gere gráficos utilizando "
    "linguagem natural. Os dados são sintéticos."
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def numero_medio(df, coluna):

    if coluna not in df.columns or df.empty:
        return None

    valores = pd.to_numeric(
        df[coluna],
        errors="coerce"
    )

    media = valores.mean()

    return (
        None
        if pd.isna(media)
        else float(media)
    )


def resumo_grupos(
    df,
    grupos,
    coluna,
    nome_media="nota_media"
):

    if not set(
        grupos + [coluna]
    ).issubset(df.columns):
        return []

    temp = df[
        grupos + [coluna]
    ].copy()

    temp[coluna] = pd.to_numeric(
        temp[coluna],
        errors="coerce"
    )

    temp = temp.dropna(subset=[coluna])

    if temp.empty:
        return []

    resultado = (
        temp.groupby(
            grupos,
            dropna=False
        )[coluna]
        .agg(["mean", "count"])
        .reset_index()
    )

    resultado["mean"] = (
        resultado["mean"].round(2)
    )

    resultado = resultado.rename(
        columns={
            "mean": nome_media,
            "count": "respostas"
        }
    )

    return resultado.where(
        pd.notna(resultado),
        None
    ).to_dict(orient="records")


def contagem_grupos(df, grupos):

    if not set(grupos).issubset(df.columns):
        return []

    resultado = (
        df.groupby(
            grupos,
            dropna=False
        )
        .size()
        .reset_index(name="quantidade")
    )

    return resultado.where(
        pd.notna(resultado),
        None
    ).to_dict(orient="records")


# ============================================================
# INDICADORES PARA A IA
# ============================================================

def criar_contexto():

    indicadores = dict(
        resumo(
            clientes,
            pessoas,
            operacoes
        )
    )

    indicadores.update({

        "avaliacao_media_clientes": numero_medio(
            clientes,
            "avaliacao"
        ),

        "nota_media_clima": numero_medio(
            pessoas,
            "nota_clima"
        ),

        "tempo_medio_resposta_h": numero_medio(
            clientes,
            "tempo_resposta_h"
        ),

        "tempo_medio_resolucao_h": numero_medio(
            operacoes,
            "tempo_resolucao_h"
        ),

        "quantidade_clientes": len(clientes),

        "quantidade_colaboradores": len(pessoas),

        "quantidade_ocorrencias": len(operacoes),

    })

    # ========================================================
    # CLIENTES
    # ========================================================

    indicadores["avaliacao_por_canal"] = resumo_grupos(
        clientes,
        ["canal"],
        "avaliacao"
    )

    indicadores["avaliacao_por_produto"] = resumo_grupos(
        clientes,
        ["produto_servico"],
        "avaliacao"
    )

    indicadores["avaliacoes_canal_sentimento"] = resumo_grupos(
        clientes,
        ["canal", "sentimento"],
        "avaliacao"
    )

    indicadores["avaliacoes_produto_sentimento"] = resumo_grupos(
        clientes,
        ["produto_servico", "sentimento"],
        "avaliacao"
    )

    indicadores["feedbacks_por_canal_sentimento"] = contagem_grupos(
        clientes,
        ["canal", "sentimento"]
    )

    indicadores["feedbacks_por_produto_sentimento"] = contagem_grupos(
        clientes,
        ["produto_servico", "sentimento"]
    )

    indicadores["avaliacao_por_regiao"] = resumo_grupos(
        clientes,
        ["regiao"],
        "avaliacao"
    )

    indicadores["resposta_por_canal"] = resumo_grupos(
        clientes,
        ["canal"],
        "tempo_resposta_h",
        "tempo_medio_h"
    )

    # ========================================================
    # PESSOAS
    # ========================================================

    indicadores["clima_por_departamento"] = resumo_grupos(
        pessoas,
        ["departamento"],
        "nota_clima"
    )

    indicadores["clima_departamento_sentimento"] = resumo_grupos(
        pessoas,
        ["departamento", "sentimento"],
        "nota_clima"
    )

    indicadores["feedbacks_por_categoria"] = contagem_grupos(
        pessoas,
        ["categoria_informada"]
    )

    indicadores["feedbacks_categoria_departamento"] = contagem_grupos(
        pessoas,
        ["departamento", "categoria_informada"]
    )

    indicadores["tempo_empresa_por_departamento"] = resumo_grupos(
        pessoas,
        ["departamento"],
        "tempo_empresa_meses",
        "tempo_medio_meses"
    )

    # ========================================================
    # OPERAÇÕES
    # ========================================================

    indicadores["ocorrencias_por_tipo"] = contagem_grupos(
        operacoes,
        ["tipo_ocorrencia"]
    )

    indicadores["ocorrencias_por_departamento"] = contagem_grupos(
        operacoes,
        ["departamento"]
    )

    indicadores["ocorrencias_por_status"] = contagem_grupos(
        operacoes,
        ["status"]
    )

    indicadores["ocorrencias_por_prioridade"] = contagem_grupos(
        operacoes,
        ["prioridade"]
    )

    indicadores["prioridades_por_departamento"] = contagem_grupos(
        operacoes,
        ["departamento", "prioridade"]
    )

    indicadores["status_por_departamento"] = contagem_grupos(
        operacoes,
        ["departamento", "status"]
    )

    indicadores["prioridade_por_tipo"] = contagem_grupos(
        operacoes,
        ["tipo_ocorrencia", "prioridade"]
    )

    indicadores["status_por_prioridade"] = contagem_grupos(
        operacoes,
        ["status", "prioridade"]
    )

    indicadores["resolucao_por_departamento"] = resumo_grupos(
        operacoes,
        ["departamento"],
        "tempo_resolucao_h",
        "tempo_medio_h"
    )

    indicadores["resolucao_por_prioridade"] = resumo_grupos(
        operacoes,
        ["prioridade"],
        "tempo_resolucao_h",
        "tempo_medio_h"
    )

    if {
        "tipo_ocorrencia",
        "impacto_clientes"
    }.issubset(operacoes.columns):

        temp = operacoes[
            [
                "tipo_ocorrencia",
                "impacto_clientes"
            ]
        ].copy()

        temp["impacto_clientes"] = pd.to_numeric(
            temp["impacto_clientes"],
            errors="coerce"
        )

        impacto = (
            temp.groupby(
                "tipo_ocorrencia"
            )["impacto_clientes"]
            .sum(min_count=1)
            .sort_values(ascending=False)
        )

        indicadores["impacto_por_tipo"] = {

            str(tipo): (
                float(valor)
                if pd.notna(valor)
                else None
            )

            for tipo, valor in impacto.items()

        }

    return indicadores


# ============================================================
# TABELAS DOS RELATÓRIOS
# ============================================================

def tabela_resumo(
    df,
    grupo,
    coluna,
    nome_grupo
):

    if not {
        grupo,
        coluna
    }.issubset(df.columns):

        st.info(
            "Dados insuficientes para esta comparação."
        )

        return

    temp = df[[grupo, coluna]].copy()

    temp[coluna] = pd.to_numeric(
        temp[coluna],
        errors="coerce"
    )

    resultado = (
        temp.groupby(grupo)[coluna]
        .agg(["mean", "count"])
        .sort_values("mean")
        .reset_index()
        .rename(columns={
            grupo: nome_grupo,
            "mean": "Nota média",
            "count": "Respostas"
        })
    )

    resultado["Nota média"] = (
        resultado["Nota média"].round(2)
    )

    table(resultado)


# ============================================================
# RELATÓRIOS RÁPIDOS
# ============================================================

def gerar_relatorio(tipo):

    st.subheader(tipo)

    # ========================================================
    # CLIENTES
    # ========================================================

    if tipo == "Satisfação de clientes":

        a, b, c = st.columns(3)

        a.metric(
            "Feedbacks",
            fmt_int(len(clientes))
        )

        media = numero_medio(
            clientes,
            "avaliacao"
        )

        b.metric(
            "Avaliação média",
            f"{fmt_dec(media, 2)} / 5"
            if media is not None
            else "—"
        )

        c.metric(
            "Sentimentos negativos",
            f"{fmt_dec(negative_share(clientes))}%"
        )

        st.markdown(
            "#### Comparação por canal"
        )

        tabela_resumo(
            clientes,
            "canal",
            "avaliacao",
            "Canal"
        )

        insight(
            "Interpretação",
            "Compare as avaliações médias considerando "
            "a quantidade de respostas de cada canal."
        )

    # ========================================================
    # PESSOAS
    # ========================================================

    elif tipo == "Clima dos colaboradores":

        a, b, c = st.columns(3)

        a.metric(
            "Feedbacks",
            fmt_int(len(pessoas))
        )

        media = numero_medio(
            pessoas,
            "nota_clima"
        )

        b.metric(
            "Nota média de clima",
            fmt_dec(media, 2)
            if media is not None
            else "—"
        )

        c.metric(
            "Sentimentos negativos",
            f"{fmt_dec(negative_share(pessoas))}%"
        )

        st.markdown(
            "#### Comparação por departamento"
        )

        tabela_resumo(
            pessoas,
            "departamento",
            "nota_clima",
            "Departamento"
        )

        insight(
            "Interpretação",
            "Diferenças pequenas entre médias não indicam, "
            "isoladamente, problemas organizacionais."
        )

    # ========================================================
    # OPERAÇÕES
    # ========================================================

    elif tipo == "Ocorrências operacionais":

        pendentes = unresolved(operacoes)

        prioridades = high_priority(operacoes)

        a, b, c = st.columns(3)

        a.metric(
            "Ocorrências",
            fmt_int(len(operacoes))
        )

        b.metric(
            "Não concluídas",
            fmt_int(int(pendentes.sum()))
        )

        c.metric(
            "Alta prioridade pendente",
            fmt_int(
                int(
                    (pendentes & prioridades).sum()
                )
            )
        )

        if "tipo_ocorrencia" in operacoes.columns:

            st.markdown(
                "#### Ocorrências por tipo"
            )

            ranking = (
                operacoes["tipo_ocorrencia"]
                .value_counts()
                .rename_axis("Tipo")
                .reset_index(name="Registros")
            )

            table(ranking)

        insight(
            "Interpretação",
            "Considere o volume, a prioridade, "
            "a situação e o impacto das ocorrências."
        )

    # ========================================================
    # RESUMO EXECUTIVO
    # ========================================================

    else:

        indicadores = resumo(
            clientes,
            pessoas,
            operacoes
        )

        a, b, c = st.columns(3)

        a.metric(
            "Registros totais",
            fmt_int(indicadores["registros"])
        )

        b.metric(
            "Feedbacks de clientes",
            fmt_int(len(clientes))
        )

        c.metric(
            "Feedbacks de colaboradores",
            fmt_int(len(pessoas))
        )

        st.metric(
            "Ocorrências não concluídas",
            fmt_int(indicadores["pendentes"])
        )

        insight(
            "Resumo executivo",
            "A NEXA reúne indicadores de clientes, "
            "colaboradores e operações para auxiliar "
            "a tomada de decisões."
        )

    st.caption(
        "Relatório produzido diretamente pelo Python."
    )


# ============================================================
# CONTROLE DA SESSÃO
# ============================================================

if "historico_nexa" not in st.session_state:

    st.session_state.historico_nexa = []


if "ultimo_grafico_nexa" not in st.session_state:

    st.session_state.ultimo_grafico_nexa = None


if "ultima_investigacao_nexa" not in st.session_state:

    st.session_state.ultima_investigacao_nexa = None


if "ultimo_tipo_analise_nexa" not in st.session_state:

    st.session_state.ultimo_tipo_analise_nexa = None


if "nexa_lock_processamento" not in st.session_state:

    st.session_state.nexa_lock_processamento = (
        threading.Lock()
    )


# ============================================================
# GERENCIAMENTO DO HISTÓRICO
# ============================================================

def salvar_mensagem(mensagem):

    st.session_state.historico_nexa.append(
        mensagem
    )

    # Preserva o último gráfico.

    if mensagem.get("contexto_grafico"):

        st.session_state.ultimo_grafico_nexa = {

            "role": "assistant",

            "content": mensagem.get(
                "content", ""
            ),

            "plano_grafico": mensagem.get(
                "plano_grafico"
            ),

            "contexto_grafico": mensagem.get(
                "contexto_grafico"
            ),

        }

        st.session_state.ultimo_tipo_analise_nexa = (
            "grafico"
        )

    # Preserva a última investigação.

    elif mensagem.get("contexto_investigacao"):

        st.session_state.ultima_investigacao_nexa = {

            "role": "assistant",

            "content": mensagem.get(
                "content", ""
            ),

            "plano_investigacao": mensagem.get(
                "plano_investigacao"
            ),

            "contexto_investigacao": mensagem.get(
                "contexto_investigacao"
            ),

        }

        st.session_state.ultimo_tipo_analise_nexa = (
            "investigacao"
        )

    elif mensagem.get("tipo_analise") == "indicadores":

        st.session_state.ultimo_tipo_analise_nexa = (
            "indicadores"
        )

    st.session_state.historico_nexa = (
        st.session_state.historico_nexa[
            -LIMITE_HISTORICO:
        ]
    )


def obter_historico():

    historico = list(
        st.session_state.historico_nexa
    )

    ultimo_grafico = (
        st.session_state.ultimo_grafico_nexa
    )

    ultima_investigacao = (
        st.session_state.ultima_investigacao_nexa
    )

    possui_grafico = any(
        mensagem.get("contexto_grafico")
        for mensagem in historico
    )

    possui_investigacao = any(
        mensagem.get("contexto_investigacao")
        for mensagem in historico
    )

    if ultimo_grafico and not possui_grafico:

        historico.insert(
            0,
            ultimo_grafico
        )

    if ultima_investigacao and not possui_investigacao:

        historico.insert(
            0,
            ultima_investigacao
        )

    return historico


# ============================================================
# RECUPERAÇÃO DO ÚLTIMO GRÁFICO
# ============================================================

def recuperar_contexto_grafico(historico):

    for mensagem in reversed(historico or []):

        contexto = mensagem.get(
            "contexto_grafico"
        )

        if contexto:
            return contexto

    ultimo = st.session_state.get(
        "ultimo_grafico_nexa"
    )

    if ultimo:

        return ultimo.get(
            "contexto_grafico"
        )

    return None


# ============================================================
# RECUPERAÇÃO DA ÚLTIMA INVESTIGAÇÃO
# ============================================================

def recuperar_investigacao(historico):

    for mensagem in reversed(historico or []):

        if mensagem.get("contexto_investigacao"):

            return mensagem

    return st.session_state.get(
        "ultima_investigacao_nexa"
    )


# ============================================================
# IDENTIFICAÇÃO DE REFERÊNCIAS
# ============================================================

def refere_ao_grafico(pergunta):

    texto = normalizar(pergunta)

    referencias = [

        "ultimo grafico",
        "grafico anterior",
        "esse grafico",
        "este grafico",
        "desse grafico",
        "deste grafico",
        "nesse grafico",
        "neste grafico",
        "desse ultimo grafico",
        "essa visualizacao",
        "esta visualizacao",

    ]

    return any(
        termo in texto
        for termo in referencias
    )


def refere_a_investigacao(pergunta):

    texto = normalizar(pergunta)

    referencias = [

        "ultima investigacao",
        "investigacao anterior",
        "essa investigacao",
        "esta investigacao",
        "dessa investigacao",
        "esses comentarios",
        "desses comentarios",
        "esses relatos",
        "essas reclamacoes",
        "essas evidencias",
        "dessas evidencias",

    ]

    return any(
        termo in texto
        for termo in referencias
    )


def solicitou_continuacao(pergunta):

    texto = normalizar(pergunta)

    referencias = [

        "agora",
        "faca o mesmo",
        "mesma analise",
        "mesma investigacao",
        "e no",
        "e na",
        "e para",
        "considere apenas",
        "considere somente",
        "somente",
        "apenas",
        "em vez de",
        "ao inves de",

    ]

    return any(
        texto.startswith(termo)
        or termo in texto
        for termo in referencias
    )


def solicitou_nova_consulta(pergunta):

    texto = normalizar(pergunta)

    termos = [

        "agora",
        "faca o mesmo",
        "mesma analise",
        "mesma investigacao",
        "considere",
        "somente",
        "apenas",
        "compare",
        "filtre",
        "em vez de",
        "ao inves de",
        "e no",
        "e na",
        "e para",
        "outro canal",
        "outro departamento",
        "outro produto",

    ]

    if any(termo in texto for termo in termos):
        return True

    # Identifica categorias existentes nos dados.
    # Isso permite reconhecer novos filtros mencionados
    # pelo usuário em perguntas complementares.

    bases = [clientes, pessoas, operacoes]

    colunas = [

        "canal",
        "produto_servico",
        "departamento",
        "categoria_informada",
        "tipo_ocorrencia",
        "sentimento",
        "prioridade",
        "status",
        "regiao",

    ]

    for df in bases:

        for coluna in colunas:

            if coluna not in df.columns:
                continue

            valores = (
                df[coluna]
                .dropna()
                .astype(str)
                .unique()
            )

            for valor in valores:

                valor_normalizado = normalizar(valor)

                if (
                    len(valor_normalizado) >= 3
                    and valor_normalizado in texto
                ):
                    return True

    return False


def refere_a_resultados(pergunta):

    texto = normalizar(pergunta)

    referencias = [

        "esses dados",
        "esses resultados",
        "desse resultado",
        "desses resultados",
        "com base nisso",
        "com base nesses",
        "com base nessas",
        "ultima analise",
        "analise anterior",
        "o que podemos concluir",
        "o que voce recomenda",
        "quais acoes",
        "como resolver",
        "como melhorar",

    ]

    return any(
        termo in texto
        for termo in referencias
    )


# ============================================================
# PREPARAÇÃO DA MEMÓRIA DA INVESTIGAÇÃO
# ============================================================

def preparar_pergunta_investigacao(
    pergunta,
    investigacao_anterior
):

    if not investigacao_anterior:

        return pergunta

    plano = investigacao_anterior.get(
        "plano_investigacao"
    )

    if not plano:

        return pergunta

    plano_serializado = json.dumps(
        plano,
        ensure_ascii=False,
        default=str,
        separators=(",", ":")
    )

    return (

        f"Pergunta complementar: {pergunta}\n\n"

        "Plano da investigação anterior:\n"

        f"{plano_serializado}\n\n"

        "Preserve o objetivo e os filtros anteriores "
        "que continuarem relevantes. Altere somente "
        "os elementos solicitados pelo usuário. "
        "Não preserve filtros incompatíveis com "
        "uma nova base ou uma nova intenção."

    )


# ============================================================
# ESTRUTURA PADRÃO DAS MENSAGENS
# ============================================================

def criar_mensagem(conteudo=""):

    return {

        "role": "assistant",

        "content": conteudo,

        "tipo_analise": None,

        "grafico": None,

        "plano_grafico": None,

        "contexto_grafico": None,

        "plano_investigacao": None,

        "contexto_investigacao": None,

    }


# ============================================================
# EXECUÇÃO DAS INVESTIGAÇÕES
# ============================================================

def executar_investigacao(
    pergunta,
    historico,
    investigacao_anterior=None
):

    pergunta_consulta = preparar_pergunta_investigacao(
        pergunta,
        investigacao_anterior
    )

    resultado = investigar_feedbacks(

        pergunta=pergunta_consulta,

        clientes=clientes,

        pessoas=pessoas,

        operacoes=operacoes,

        historico=historico,

    )

    # O planejador pode decidir que a pergunta
    # não necessita de uma investigação.

    if resultado is None:

        return None

    mensagem = criar_mensagem(
        resultado.get(
            "resposta",
            "Não foi possível concluir a investigação."
        )
    )

    mensagem["tipo_analise"] = "investigacao"

    mensagem["plano_investigacao"] = (
        resultado.get("plano")
    )

    mensagem["contexto_investigacao"] = (
        resultado.get("contexto")
    )

    return mensagem


def interpretar_investigacao_anterior(
    pergunta,
    investigacao,
    historico
):

    contexto = investigacao.get(
        "contexto_investigacao"
    )

    if not contexto:

        return criar_mensagem(
            "Ainda não existe uma investigação "
            "disponível para interpretar."
        )

    resposta = interpretar_investigacao(

        pergunta=pergunta,

        contexto=contexto,

        historico=historico,

    )

    mensagem = criar_mensagem(resposta)

    mensagem["tipo_analise"] = "investigacao"

    mensagem["contexto_investigacao"] = contexto

    mensagem["plano_investigacao"] = (
        investigacao.get("plano_investigacao")
    )

    return mensagem


# ============================================================
# PROCESSAMENTO DAS PERGUNTAS
# ============================================================

def analisar_pergunta(
    pergunta,
    historico
):

    mensagem = criar_mensagem()

    # ========================================================
    # 1. RESPOSTAS LOCAIS
    # ========================================================

    local = resposta_local(pergunta)

    if local is not None:

        mensagem["content"] = local

        print(
            "[NEXA] Resposta local — 0 tokens da API."
        )

        return mensagem

    # ========================================================
    # 2. RECUPERAÇÃO DO CONTEXTO ANTERIOR
    # ========================================================

    investigacao_anterior = recuperar_investigacao(
        historico
    )

    ultimo_tipo = st.session_state.get(
        "ultimo_tipo_analise_nexa"
    )

    contexto_grafico_anterior = (
        recuperar_contexto_grafico(historico)
    )

    referencia_investigacao = (
        refere_a_investigacao(pergunta)
    )

    referencia_grafico = (
        refere_ao_grafico(pergunta)
    )

    continuacao = solicitou_continuacao(
        pergunta
    )

    referencia_resultados = refere_a_resultados(
        pergunta
    )

    texto = normalizar(pergunta)

    pedido_visual = any(
        termo in texto
        for termo in [
            "grafico",
            "grafco",
            "visualizacao",
            "histograma",
            "boxplot",
            "dispersao",
        ]
    )

    # Evita que uma continuação de investigação
    # seja confundida com a alteração de um
    # gráfico criado anteriormente.

    historico_graficos = historico

    if (
        ultimo_tipo != "grafico"
        and not pedido_visual
    ):

        historico_graficos = [

            {
                chave: valor

                for chave, valor in item.items()

                if chave not in (
                    "plano_grafico",
                    "contexto_grafico",
                    "grafico",
                )
            }

            for item in historico

        ]

    # ========================================================
    # 3. CRIAÇÃO OU ALTERAÇÃO DE GRÁFICOS
    # ========================================================

    if solicitou_grafico(
        pergunta,
        historico_graficos
    ):

        resultado = gerar_grafico_inteligente(

            pergunta=pergunta,

            clientes=clientes,

            pessoas=pessoas,

            operacoes=operacoes,

            historico=historico_graficos

        )

        if resultado.get("mensagem"):

            mensagem["content"] = (
                resultado["mensagem"]
            )

            return mensagem

        figura = resultado.get("figura")

        if figura is None:

            mensagem["content"] = (

                "Não foi possível gerar uma visualização "
                "com os dados disponíveis."

            )

            return mensagem

        contexto_grafico = resultado.get(
            "contexto"
        )

        plano = resultado.get("plano")

        mensagem["tipo_analise"] = "grafico"

        mensagem["grafico"] = json.loads(
            figura.to_json()
        )

        mensagem["plano_grafico"] = plano

        mensagem["contexto_grafico"] = (
            contexto_grafico
        )

        try:

            resposta = explain(

                question=pergunta,

                indicators={},

                history=historico,

                contexto_grafico=contexto_grafico

            )

            texto_resposta = normalizar(
                resposta
            )

            falhas = [

                "temporariamente sobrecarregada",

                "temporariamente indisponivel",

                "limite de requisicoes",

            ]

            if any(
                termo in texto_resposta
                for termo in falhas
            ):

                mensagem["content"] = (

                    "O gráfico foi calculado com sucesso. "
                    "A interpretação automática está "
                    "temporariamente indisponível."

                )

            else:

                mensagem["content"] = resposta

        except Exception:

            logger.exception(
                "Erro na interpretação do gráfico."
            )

            mensagem["content"] = (

                "O gráfico foi calculado com sucesso. "
                "A interpretação automática está "
                "temporariamente indisponível."

            )

        return mensagem

    # ========================================================
    # 4. INTERPRETAÇÃO DE GRÁFICOS ANTERIORES
    # ========================================================

    if referencia_grafico or (

        ultimo_tipo == "grafico"

        and referencia_resultados

        and not referencia_investigacao

        and not continuacao

    ):

        if contexto_grafico_anterior:

            mensagem["content"] = explain(

                question=pergunta,

                indicators={},

                history=historico,

                contexto_grafico=contexto_grafico_anterior

            )

            mensagem["tipo_analise"] = "grafico"

        else:

            mensagem["content"] = (

                "Ainda não existe um gráfico nesta conversa "
                "para que eu possa interpretar. "
                "Solicite uma visualização primeiro."

            )

        return mensagem

    # ========================================================
    # 5. CONTINUAÇÃO DE INVESTIGAÇÕES
    # ========================================================

    continuar_investigacao = (

        referencia_investigacao

        or (

            ultimo_tipo == "investigacao"

            and (
                continuacao
                or referencia_resultados
            )

        )

    )

    if continuar_investigacao:

        if investigacao_anterior:

            # Se o usuário solicitou novos filtros,
            # executa uma nova consulta aos registros.

            if solicitou_nova_consulta(pergunta):

                resultado = executar_investigacao(

                    pergunta,

                    historico,

                    investigacao_anterior

                )

                if resultado is not None:

                    return resultado

                return criar_mensagem(
                    "Não consegui identificar os novos "
                    "filtros da investigação. Informe "
                    "qual canal, produto ou departamento "
                    "deseja analisar."
                )

            # Se deseja apenas interpretar os resultados,
            # reutiliza as evidências já calculadas.

            return interpretar_investigacao_anterior(

                pergunta,

                investigacao_anterior,

                historico

            )

        elif referencia_investigacao:

            return criar_mensagem(

                "Ainda não existe uma investigação "
                "nesta conversa. Solicite uma análise "
                "dos comentários ou ocorrências primeiro."

            )

    # ========================================================
    # 6. NOVAS INVESTIGAÇÕES DOS REGISTROS
    # ========================================================

    if solicitou_investigacao(pergunta):

        resultado = executar_investigacao(

            pergunta,

            historico

        )

        if resultado is not None:

            return resultado

    # ========================================================
    # 7. PERGUNTAS SOBRE INDICADORES
    # ========================================================

    indicadores = criar_contexto()

    # Evita que o assistente recupere um gráfico antigo
    # indevidamente ao interpretar indicadores gerais.

    historico_indicadores = [

        {
            chave: valor

            for chave, valor in item.items()

            if chave not in (
                "contexto_grafico",
                "plano_grafico",
                "grafico",
            )
        }

        for item in historico

    ]

    mensagem["content"] = explain(

        question=pergunta,

        indicators=indicadores,

        history=historico_indicadores

    )

    mensagem["tipo_analise"] = "indicadores"

    return mensagem

# ============================================================
# PROTEÇÃO DA MEMÓRIA E DO CONTEXTO CORPORATIVO
# ============================================================

from services.assistente import (
    verificar_memoria,
    solicitou_contexto_interno,
    formatar_resposta_nexa,
)


_analisar_pergunta_original = analisar_pergunta


def analisar_pergunta(
    pergunta,
    historico
):

    # Impede solicitações de configurações internas.

    if solicitou_contexto_interno(pergunta):

        return criar_mensagem(

            "Não posso fornecer configurações internas "
            "ou credenciais da plataforma. Posso ajudar "
            "com análises dos dados corporativos."

        )

    # Verifica o histórico antes de consultar
    # indicadores, gráficos ou registros.

    resposta_memoria = verificar_memoria(
        pergunta,
        historico
    )

    if resposta_memoria is not None:

        print(
            "[NEXA] Verificação local de memória "
            "— 0 tokens da API."
        )

        return criar_mensagem(
            resposta_memoria
        )

    # Preserva o processamento original.

    mensagem = _analisar_pergunta_original(
        pergunta,
        historico
    )

    # Padroniza todas as respostas exibidas.

    if mensagem.get("content"):

        mensagem["content"] = (
            formatar_resposta_nexa(
                mensagem["content"]
            )
        )

    return mensagem

# ============================================================
# PROCESSAMENTO SEGURO
# ============================================================

def processar_com_seguranca(
    pergunta,
    historico
):

    try:

        return analisar_pergunta(
            pergunta,
            historico
        )

    except LimiteAPI:

        logger.warning(
            "Limite temporário da API Groq atingido."
        )

        return criar_mensagem(

            "A inteligência artificial está recebendo "
            "muitas solicitações no momento. "
            "Aguarde alguns instantes e tente novamente."

        )

    except ValueError as erro:

        logger.warning(

            "Erro de validação na Inteligência NEXA: %s",

            erro

        )

        return criar_mensagem(

            "Não foi possível concluir essa análise. "
            "Tente reformular sua pergunta ou solicitar "
            "uma comparação diferente."

        )

    except RuntimeError:

        logger.exception(
            "Erro de comunicação na Inteligência NEXA."
        )

        return criar_mensagem(

            "A inteligência artificial está "
            "temporariamente indisponível. "
            "Tente novamente em alguns instantes."

        )

    except Exception:

        logger.exception(
            "Erro inesperado na Inteligência NEXA."
        )

        return criar_mensagem(

            "Não foi possível concluir a análise. "
            "Tente novamente."

        )


# ============================================================
# EXIBIÇÃO DAS EVIDÊNCIAS DA INVESTIGAÇÃO
# ============================================================

def exibir_investigacao(contexto):

    if not contexto:

        return

    nomes_bases = {

        "clientes": "Clientes",

        "pessoas": "Colaboradores",

        "operacoes": "Operações",

    }

    nomes_colunas = {

        "canal": "Canal",

        "produto_servico": "Produto ou serviço",

        "sentimento": "Sentimento",

        "departamento": "Departamento",

        "categoria_informada": "Categoria",

        "tipo_ocorrencia": "Tipo de ocorrência",

        "prioridade": "Prioridade",

        "status": "Status",

        "regiao": "Região",

        "avaliacao": "Avaliação",

        "nota_clima": "Nota de clima",

        "tempo_resposta_h": "Tempo de resposta (h)",

        "tempo_resolucao_h": "Tempo de resolução (h)",

        "tempo_empresa_meses": "Tempo de empresa (meses)",

        "impacto_clientes": "Impacto nos clientes",

    }

    operadores = {

        "igual": "=",

        "diferente": "≠",

        "em": "entre os valores",

        "maior": ">",

        "menor": "<",

        "maior_igual": "≥",

        "menor_igual": "≤",

    }

    quantidade = contexto.get(
        "registros_analisados", 0
    )

    base = nomes_bases.get(
        contexto.get("base"),
        "Dados corporativos"
    )

    st.caption(

        f"Investigação realizada na base de {base} "
        f"com {fmt_int(quantidade)} registros filtrados."

    )

    with st.expander(
        "Consultar evidências da investigação",
        expanded=False
    ):

        a, b = st.columns(2)

        a.metric(
            "Registros investigados",
            fmt_int(quantidade)
        )

        b.metric(
            "Registros na base",
            fmt_int(
                contexto.get("total_base", 0)
            )
        )

        if "comentarios_validos" in contexto:

            st.metric(

                "Comentários válidos",

                fmt_int(
                    contexto["comentarios_validos"]
                )

            )

        filtros = contexto.get(
            "filtros_aplicados", []
        )

        st.markdown("#### Filtros aplicados")

        if filtros:

            for filtro in filtros:

                coluna = nomes_colunas.get(

                    filtro.get("coluna"),

                    filtro.get("coluna", "")

                )

                operador = operadores.get(

                    filtro.get("operador"),

                    filtro.get("operador", "")

                )

                valor = filtro.get("valor")

                if isinstance(valor, list):

                    valor = ", ".join(
                        str(item)
                        for item in valor
                    )

                st.text(
                    f"{coluna} {operador} {valor}"
                )

        else:

            st.caption(
                "Nenhum filtro específico foi aplicado."
            )

        # ====================================================
        # PADRÕES TEXTUAIS
        # ====================================================

        padroes = contexto.get(
            "padroes_textuais", {}
        )

        expressoes = padroes.get(
            "expressoes_recorrentes", []
        )

        palavras = padroes.get(
            "palavras_recorrentes", []
        )

        if expressoes:

            st.markdown(
                "#### Expressões recorrentes"
            )

            df_expressoes = pd.DataFrame(
                expressoes
            ).rename(columns={

                "termo": "Expressão",

                "comentarios": "Comentários",

                "percentual": "Presença (%)",

            })

            table(df_expressoes)

        if palavras:

            st.markdown(
                "#### Palavras recorrentes"
            )

            df_palavras = pd.DataFrame(
                palavras
            ).rename(columns={

                "termo": "Palavra",

                "comentarios": "Comentários",

                "percentual": "Presença (%)",

            })

            table(df_palavras)

        if palavras or expressoes:

            st.caption(

                "Os percentuais representam a presença "
                "de cada termo nos comentários analisados. "
                "Um comentário pode conter vários termos. "
                "Esses valores não representam diretamente "
                "percentuais de reclamações por assunto."

            )

        # ====================================================
        # EXEMPLOS DE COMENTÁRIOS
        # ====================================================

        evidencias = contexto.get(
            "evidencias_textuais", []
        )

        if evidencias:

            st.markdown(
                "#### Exemplos de comentários"
            )

            for indice, evidencia in enumerate(
                evidencias,
                start=1
            ):

                comentario = evidencia.get(
                    "comentario", ""
                )

                frequencia = evidencia.get(
                    "frequencia_texto_identico", 0
                )

                st.text(
                    f"{indice}. {comentario}"
                )

                st.caption(

                    "Frequência do texto apresentado "
                    f"após tratamento: {fmt_int(frequencia)}"

                )

            st.caption(

                "Os comentários apresentados são uma "
                "seleção das evidências disponíveis, "
                "não a totalidade dos registros."

            )

        # ====================================================
        # INDICADORES FILTRADOS
        # ====================================================

        indicadores = contexto.get(
            "indicadores_filtrados", {}
        )

        if indicadores:

            st.markdown(
                "#### Indicadores dos registros filtrados"
            )

            for coluna, valores in indicadores.items():

                nome = nomes_colunas.get(
                    coluna,
                    coluna.replace("_", " ").capitalize()
                )

                st.markdown(f"**{nome}**")

                if isinstance(valores, list):

                    df_indicador = pd.DataFrame(
                        valores
                    ).rename(columns={

                        "categoria": "Categoria",

                        "registros": "Registros",

                    })

                    table(df_indicador)

                elif isinstance(valores, dict):

                    a, b, c = st.columns(3)

                    a.metric(
                        "Média",
                        fmt_dec(
                            valores.get("media", 0), 2
                        )
                    )

                    b.metric(
                        "Mínimo",
                        fmt_dec(
                            valores.get("minimo", 0), 2
                        )
                    )

                    c.metric(
                        "Máximo",
                        fmt_dec(
                            valores.get("maximo", 0), 2
                        )
                    )

        if contexto.get("indicadores_parciais"):

            st.caption(

                "Parte dos indicadores foi omitida "
                "para controlar o tamanho do contexto "
                "enviado à inteligência artificial."

            )

        limitacao = contexto.get(
            "limitacao_textual"
        )

        if limitacao:

            st.caption(limitacao)


# ============================================================
# EXIBIÇÃO DO CONTEÚDO
# ============================================================

def exibir_conteudo(
    mensagem,
    indice
):

    if mensagem.get("content"):

        st.markdown(
            mensagem["content"]
        )

    if mensagem.get("grafico"):

        figura = go.Figure(
            mensagem["grafico"]
        )

        st.plotly_chart(

            figura,

            use_container_width=True,

            key=f"grafico_nexa_{indice}",

            config={
                "displaylogo": False
            }

        )

        contexto = mensagem.get(
            "contexto_grafico"
        ) or {}

        quantidade = contexto.get(
            "registros_filtrados"
        )

        if quantidade is not None:

            st.caption(

                "Gráfico calculado pelo Python "
                f"com {fmt_int(quantidade)} registros "
                "após a aplicação dos filtros."

            )

        else:

            st.caption(

                "Gráfico calculado diretamente "
                "com os dados da plataforma."

            )

    if mensagem.get("contexto_investigacao"):

        exibir_investigacao(
            mensagem["contexto_investigacao"]
        )


def exibir_mensagem(
    mensagem,
    indice
):

    with st.chat_message(
        mensagem["role"]
    ):

        exibir_conteudo(
            mensagem,
            indice
        )


# ============================================================
# ASSISTENTE NEXA
# ============================================================

st.subheader(
    "Assistente NEXA"
)

st.caption(

    "Investigue registros, analise comentários, "
    "gere gráficos personalizados e obtenha "
    "recomendações utilizando linguagem natural."

)

chave_disponivel = bool(
    api_key()
)

if not chave_disponivel:

    st.warning(

        "A inteligência artificial não está configurada. "
        "Verifique GROQ_API_KEY. "
        "Os relatórios rápidos continuam disponíveis."

    )


# ============================================================
# EXEMPLOS DE PERGUNTAS
# ============================================================

with st.expander(

    "O que posso perguntar?",

    expanded=False

):

    st.markdown(
        """
        **Investigação dos registros**

        - Qual é a maior reclamação dos clientes no WhatsApp?
        - O que os clientes mais elogiam nos comentários?
        - Quais problemas aparecem nos feedbacks negativos?
        - Quais são os principais motivos de insatisfação
          dos colaboradores?
        - Quais ocorrências críticas continuam pendentes?
        - Investigue os comentários negativos do canal E-mail.

        **Análises inteligentes**

        - Qual departamento precisa de mais atenção?
        - Quais são os principais problemas da empresa?
        - Quais ações você recomenda?

        **Gráficos personalizados**

        - Gere um gráfico dos feedbacks do canal E-mail.
        - Compare as avaliações dos produtos considerando
          somente o canal E-mail. Gere um gráfico.
        - Gere um gráfico de dispersão entre o tempo
          de resposta e a avaliação dos clientes.
        - Mostre um boxplot das notas de clima
          por departamento.
        - Gere um histograma do tempo de resolução.
        - Escolha o melhor gráfico para apresentar
          os problemas discutidos anteriormente.

        **Perguntas complementares**

        - Agora faça a mesma investigação para WhatsApp.
        - Quais ações você recomenda com base nesses comentários?
        - O que podemos concluir desse último gráfico?
        - Transforme esse gráfico em um gráfico de barras.
        - Explique melhor os resultados da última investigação.
        """
    )


# ============================================================
# HISTÓRICO DA CONVERSA
# ============================================================

st.session_state.historico_nexa = (

    st.session_state.historico_nexa[
        -LIMITE_HISTORICO:
    ]

)

for indice, mensagem in enumerate(

    st.session_state.historico_nexa

):

    exibir_mensagem(
        mensagem,
        indice
    )


# ============================================================
# ENTRADA DE PERGUNTAS
# ============================================================

pergunta = st.chat_input(

    "Pergunte ou solicite uma análise personalizada...",

    disabled=not chave_disponivel,

    max_chars=LIMITE_PERGUNTA,

    key="pergunta_livre_nexa"

)


# ============================================================
# CONTROLE DE SOLICITAÇÕES
# ============================================================

if pergunta:

    pergunta = pergunta.strip()

    lock = st.session_state.nexa_lock_processamento

    if not lock.acquire(blocking=False):

        st.warning(

            "A NEXA ainda está processando "
            "uma solicitação. Aguarde a conclusão."

        )

    else:

        try:

            erro_validacao = validar_mensagem(
                pergunta
            )

            if erro_validacao:

                st.warning(
                    erro_validacao
                )

            else:

                historico_anterior = obter_historico()

                salvar_mensagem({

                    "role": "user",

                    "content": pergunta

                })

                with st.chat_message("user"):

                    st.markdown(
                        pergunta
                    )

                with st.chat_message("assistant"):

                    with st.spinner(

                        "Interpretando a solicitação "
                        "e analisando os dados..."

                    ):

                        mensagem = processar_com_seguranca(

                            pergunta,

                            historico_anterior

                        )

                    salvar_mensagem(
                        mensagem
                    )

                    exibir_conteudo(

                        mensagem,

                        "resposta_atual"

                    )

        finally:

            lock.release()


# ============================================================
# LIMPAR CONVERSA
# ============================================================

if st.session_state.historico_nexa:

    if st.button(

        "Limpar conversa",

        key="limpar_historico_nexa",

        disabled=(

            st.session_state
            .nexa_lock_processamento
            .locked()

        )

    ):

        st.session_state.historico_nexa = []

        st.session_state.ultimo_grafico_nexa = None

        st.session_state.ultima_investigacao_nexa = None

        st.session_state.ultimo_tipo_analise_nexa = None

        st.rerun()


# ============================================================
# ANÁLISES RÁPIDAS
# ============================================================

st.divider()

st.subheader(
    "Análises rápidas"
)

st.caption(

    "Consulte relatórios predefinidos sem utilizar "
    "a inteligência artificial generativa."

)

opcoes = [

    "Resumo executivo",

    "Satisfação de clientes",

    "Clima dos colaboradores",

    "Ocorrências operacionais",

]

tipo_relatorio = st.selectbox(

    "Escolha uma análise",

    opcoes,

    key="relatorio_nexa"

)

if st.button(

    "Gerar relatório",

    type="primary",

    use_container_width=True,

    key="botao_relatorio_nexa"

):

    st.session_state.relatorio_atual_nexa = (
        tipo_relatorio
    )

if "relatorio_atual_nexa" in st.session_state:

    st.divider()

    gerar_relatorio(
        st.session_state.relatorio_atual_nexa
    )

footer()