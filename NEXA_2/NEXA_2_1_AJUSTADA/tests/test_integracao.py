"""Teste opcional da integração, executado apenas com credenciais disponíveis."""
import os
import pytest
from services.mongodb import load_data

@pytest.mark.skipif(not os.getenv("MONGODB_URI"), reason="Requer MONGODB_URI e acesso à rede")
def test_mongodb():
    frames = load_data.__wrapped__(os.environ["MONGODB_URI"])
    assert set(frames) == {"clientes", "pessoas", "operacoes"}
    assert all(len(df) > 0 for df in frames.values())
