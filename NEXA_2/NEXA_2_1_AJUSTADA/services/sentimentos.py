"""Análise textual heurística, transparente e reprodutível."""
import re
import unicodedata
import pandas as pd

def normalize(value: object) -> str:
    s = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(c for c in s if not unicodedata.combining(c))


POSITIVE = {"otimo", "excelente", "bom", "boa", "rapido", "eficiente", "satisfeito", "satisfeita", "recomendo", "agradavel", "resolveu", "resolvido", "organizado", "melhorou", "positivo", "qualidade", "atencioso", "facil", "parabens"}
NEGATIVE = {"ruim", "pessimo", "demora", "demorado", "problema", "erro", "falha", "reclamacao", "insatisfeito", "insatisfeita", "atraso", "dificil", "lento", "negativo", "defeito", "indisponivel", "inadequado", "pior", "frustrante"}


def classify_text(text: object) -> str:
    """Heurística de palavras inteiras, não é modelo treinado de PLN."""
    words = set(re.findall(r"[a-z]+", normalize(text)))
    p = len(words & POSITIVE)
    n = len(words & NEGATIVE)
    return "Positivo" if p > n else "Negativo" if n > p else "Neutro"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "texto" in result:
        result["sentimento"] = result["texto"].fillna("").map(classify_text)
    return result


