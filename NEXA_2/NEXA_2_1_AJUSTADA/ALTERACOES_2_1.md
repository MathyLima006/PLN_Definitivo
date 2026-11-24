# NEXA 2.1 — melhorias visuais e analíticas

Esta versão **não modifica o MongoDB** nem os dados existentes. Usa o mesmo `.env` e banco `Nexa_V2`.

- Clientes: substituída a média por canal pela taxa de notas 1–2; substituída a média por produto pela distribuição percentual de notas.
- Clientes e Pessoas: gráficos de sentimentos com filtro de séries (inicialmente Negativo) e alternância entre percentual mensal e quantidade.
- Pessoas: substituídas as médias por departamento e tempo de empresa por taxas de clima desfavorável (nota até 5). As contagens permanecem acessíveis.
- Clientes: gráfico tempo de resposta × avaliação com escala de 1 a 5 e texto interpretativo.
- Clientes, Pessoas e Operações: tabelas recolhidas inicialmente, com download CSV dos dados filtrados.
- Operações: nota metodológica para distinguir frequência de impacto.

## Instalação
1. Extraia a pasta, copie o `.env` funcional da versão anterior para a raiz ao lado de `PLN.py`.
2. `python -m pip install -r requirements.txt`
3. `python -m streamlit run PLN.py`

O Groq continua opcional. As médias semelhantes do banco não foram artificialmente alteradas.
