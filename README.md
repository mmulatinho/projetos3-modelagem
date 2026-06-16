# Prisma Key — Aplicação ESG

Aplicação Streamlit que integra o banco de dados ESG da disciplina, cobrindo:

- **CRUD** em 4 tabelas (Empresa, Unidade, Métrica, Registro) — inserção / alteração / exclusão.
- **Consultas SQL** das Etapas 03 e 04 (8 no total, incluindo anti-join, full outer e subconsultas).
- **Views** `vw_resumo_metricas_empresas` e `vw_detalhes_auditoria_completa` acessíveis via interface.
- **Funções / Procedimentos / Triggers** da Etapa 05 — incluindo procedimento com cursor e trigger de log.
- **Dashboard** estatístico com KPIs (média, mediana, moda, variância, desvio padrão, %) e **8 gráficos** dinâmicos (barras, pizza, linha, box, histograma, radar, empilhado, heatmap) com filtros por empresa, pilar ESG e status.

## Estrutura

```
app/
├── app.py                # Aplicação Streamlit
├── db.py                 # Camada de acesso ao SQLite
├── requirements.txt
├── prismakey.db          # gerado no primeiro run
└── sql/
    ├── create_db.sql     # schema + índices + views + triggers
    ├── insert_data.sql   # carga inicial
    └── queries.sql       # 8 consultas das Etapas 03 e 04
```

## Como executar

```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Na primeira execução o banco SQLite é criado automaticamente (`prismakey.db`).
Use o botão **Reset banco** na barra lateral para recarregar os dados.

## Mapeamento dos entregáveis

| Etapa | Entregável                                  | Onde está                                   |
|-------|---------------------------------------------|---------------------------------------------|
| 03    | CRUD ≥ 2 tabelas                            | Aba **CRUD** (4 tabelas)                    |
| 03    | 4 consultas (≥1 join)                       | Aba **Consultas** Q1–Q4                     |
| 03    | Gráficos da estatística                     | Aba **Dashboard**                           |
| 04    | 2 índices                                   | `sql/create_db.sql` (idx_registro_*)        |
| 04    | 4 consultas avançadas                       | Aba **Consultas** Q5–Q8                     |
| 04    | 2 views                                     | Aba **Views**                               |
| 05    | 2 funções (uma com condicional)             | Aba **Funções & Procedimentos**             |
| 05    | 2 procedimentos (um UPDATE, um com CURSOR)  | Aba **Funções & Procedimentos**             |
| 05    | 2 triggers (um com log)                     | `sql/create_db.sql` + aba **Log de Triggers** |
| 06    | CRUD ≥ 4 tabelas                            | Aba **CRUD**                                |
| 06    | Integração com funções/procs/triggers       | Abas correspondentes                        |
| 06    | Dashboard com ≥ 6 gráficos                  | Aba **Dashboard** (8 gráficos)              |

## Observação sobre o banco

O SQL original das funções/procedimentos/triggers foi escrito em **MySQL**.
Para portabilidade e demonstração no Streamlit usamos **SQLite** com:
- `triggers` reescritos em sintaxe SQLite (mesma semântica).
- `funções` e `procedimentos` MySQL traduzidos para chamadas Python em `db.py` (que executam o mesmo SQL e a mesma lógica de IF/CURSOR).

Os scripts MySQL originais permanecem nas pastas das entregas anteriores.
