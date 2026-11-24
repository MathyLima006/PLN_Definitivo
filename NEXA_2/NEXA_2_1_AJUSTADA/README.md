# NEXA 2.0 — Corporate Intelligence

Projeto acadêmico de inteligência corporativa com **dados sintéticos**. Dashboards de Clientes, Pessoas, Operações e Inteligência NEXA.

## Como iniciar no Windows (VS Code)

1. Extraia este ZIP para uma **nova pasta**; não substitua a versão antiga antes de validar.
2. No terminal, dentro da pasta, execute `python -m pip install -r requirements.txt`.
3. Crie `.env` na raiz usando `.env.example` como referência e coloque sua URI real **sem compartilhar nem publicar**.
4. Execute `python -m streamlit run PLN.py`.
5. Use a barra lateral para navegar pelas páginas.

## Fonte de dados

Banco MongoDB Atlas `Nexa_V2`, coleções `clientes`, `colaboradores`, `operacoes`. O aplicativo consulta as coleções via PyMongo e converte para Pandas; não depende dos CSVs originais. Cache de 5 minutos; botão de atualização manual. Os valores 4.837 / 1.769 / 2.943 são esperados apenas para a importação original.

## PLN e IA

- **Sentimento padrão:** regras lexicais explicáveis; não é modelo treinado.
- **Temas:** extração estatística de termos TF-IDF com scikit-learn (PLN real, sem treinamento supervisionado).
- **Modelo de sentimento opcional:** instale `python -m pip install -r requirements-ml.txt`; abra Clientes ou Pessoas e execute a análise em amostra. O primeiro uso pode baixar pesos grandes e exigir memória; revise os resultados.
- **LLM opcional:** configure `GROQ_API_KEY` no `.env` para explicações textuais. O LLM recebe somente agregados (contagens e taxas), não feedbacks individuais. Necessita conexão com a internet, conta/chave válidas e pode estar sujeito a limites e custos do provedor. Sem chave, a análise local continua funcionando.

## Testes

`python -m pip install pytest` e `python -m pytest -q` (teste de MongoDB só roda com `MONGODB_URI` no ambiente).

## Publicação

Na Streamlit Community Cloud, defina o arquivo principal `PLN.py` e cadastre `MONGODB_URI` em **Secrets**. Nunca faça commit de `.env` ou `secrets.toml`. Libere no Atlas somente os IPs necessários, de acordo com sua política de segurança.

## Limitações

Não há autenticação de usuários, nem CRUD na interface. Indicadores são descritivos, não demonstram causalidade. O modelo multilíngue é opcional e não foi validado para o domínio específico. O código foi testado estaticamente; é necessário validar a conexão e os gráficos no seu ambiente.
