import pandas as pd
from services.indicadores import resumo, taxa_negativa, nao_concluidas, alta_prioridade
from services.sentimentos import classify_text
from services.pln import keywords

def test_resumo():
    cli = pd.DataFrame({"sentimento": ["Negativo", "Positivo"]})
    pes = pd.DataFrame({"sentimento": ["Neutro"]})
    ops = pd.DataFrame({"status": ["Aberto", "Concluído", "Em andamento"], "prioridade": ["Crítica", "Alta", "Baixa"]})
    result = resumo(cli, pes, ops)
    assert result["registros"] == 6
    assert result["pendentes"] == 2
    assert result["criticas_pendentes"] == 1
    assert result["negativos_clientes_pct"] == 50.0

def test_texto():
    assert classify_text("Atendimento excelente") == "Positivo"
    assert classify_text("Serviço péssimo") == "Negativo"
    assert classify_text("Relatório mensal") == "Neutro"

def test_keywords():
    result = keywords(["atendimento lento e demorado", "atendimento muito lento", "demora no atendimento"])
    assert not result.empty
