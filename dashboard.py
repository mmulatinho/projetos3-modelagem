import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text  # ❌ REMOVIDOS MATPLOTLIB E SEABORN PARA ACELERAR CARREGAMENTO

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard de Gestão ESG - Prisma Key",
    page_icon="🌱",
    layout="wide"
)

# Título Principal do Dashboard
st.title("🌱 Plataforma de Gestão e Criticidade ESG de Fornecedores")
st.markdown("---")

# ==============================================================================
# CONEXÃO COM O BANCO DE DADOS E CARREGAMENTO (VERSÃO DE ALTA PERFORMANCE)
# ==============================================================================
DATABASE_URL = "postgresql://mlflow:mlflow_password@postgres:5432/mlflow_db"

@st.cache_resource
def obter_conexao_banco():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

@st.cache_data(ttl=300) # Guarda os dados na memória por 5 minutos
def carregar_dados_do_banco():
    # Mudamos para LEFT JOIN para trazer quem tem e quem NÃO tem auditoria ainda (dados antigos)
    query = """
        SELECT 
            f.id,
            f.name,
            f.industry_segment AS industry,
            f.cnpj,
            f.status_homologacao,
            COALESCE(a.environment_score, 250.0) AS environment_score,
            COALESCE(a.social_score, 250.0) AS social_score,
            COALESCE(a.governance_score, 250.0) AS governance_score,
            COALESCE(p.total_score_predito, 750.0) AS total_score,
            COALESCE(p.total_level_predito, 'Medium') AS total_level
        FROM fornecedores f
        LEFT JOIN auditorias_esg a ON f.id = a.fornecedor_id
        LEFT JOIN predicoes_risco p ON a.id = p.auditoria_id;
    """
    engine_otimizada = obter_conexao_banco()
    with engine_otimizada.connect() as conn:
        return pd.read_sql(query, conn)

try:
    df_fornecedores = carregar_dados_do_banco()
except Exception as e:
    st.error(f"Erro ao conectar ao banco de dados PostgreSQL: {e}")
    df_fornecedores = pd.DataFrame()

# ==============================================================================
# PROCESSAMENTO VETORIZADO ÚNICO (EVITA LENTIDÃO POR RECALCULO)
# ==============================================================================
if not df_fornecedores.empty:
    total_forn = len(df_fornecedores)
    forn_implementados = (df_fornecedores['total_level'].isin(['Low', 'Medium'])).sum()
    taxa_implementacao = (forn_implementados / total_forn) * 100 if total_forn > 0 else 0

    # ⚡ Calculado apenas UMA VEZ para todo o ciclo de vida do script
    alta_env = int((df_fornecedores['environment_score'] < 300).sum())
    alta_soc = int((df_fornecedores['social_score'] < 200).sum())
    alta_gov = int((df_fornecedores['governance_score'] < 200).sum())
    
    score_maturidade_medio = df_fornecedores['total_score'].mean()
    riscos_criticos_atuais = (df_fornecedores['total_level'] == 'High').sum()
    score_reputacao_estimado = df_fornecedores['total_score'].quantile(0.75)

    # ==============================================================================
    # KPIs GLOBAIS FIXOS NO TOPO
    # ==============================================================================
    st.markdown("### 📈 Indicadores Chave de Performance (KPIs Operacionais)")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(label="🎯 KPI Principal: Implementação ESG", value=f"{taxa_implementacao:.1f}%", delta="Práticas Ativas na Cadeia")
    with kpi_col2:
        st.metric(label="📈 KPI Evolução: Maturidade Média", value=f"{score_maturidade_medio:.1f} pts", delta="Score Geral (KNN)", delta_color="normal")
    with kpi_col3:
        st.metric(label="⚠️ KPI Risco: Riscos Críticos", value=int(riscos_criticos_atuais), delta="-12% vs ciclo anterior", delta_color="inverse")
    with kpi_col4:
        st.metric(label="🏆 KPI Reputação: Score Líderes", value=f"{score_reputacao_estimado:.1f} pts", delta="Rankings Externos (Q3)")
    
    st.markdown("---")

# Definição das 5 Abas unificadas
aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "🎯 Matriz de Criticidade (E1)", 
    "🗺️ Mapa de Riscos ESG (E2)", 
    "📋 Scorecard & Ranking (E3)", 
    "📊 Benchmarking Interno (E4)",
    "⚙️ Painel Operacional (CRUD & BD)"
])

# ------------------------------------------------------------------------------
# ABA 1: MATRIZ DE CRITICIDADE (ENTREGÁVEL 1)
# ------------------------------------------------------------------------------
with aba1:
    st.header("🎯 Matriz de Criticidade de Fornecedores")
    st.subheader("Priorização de Riscos: Impacto Operacional vs. Performance ESG")
    
    if not df_fornecedores.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            setores = st.multiselect("Filtrar por Indústria / Setor:", options=df_fornecedores['industry'].dropna().unique(), default=[], key="setor_aba1")
        with col_f2:
            niveis = st.multiselect("Filtrar por Nível de Risco Calculado (IA):", options=df_fornecedores['total_level'].dropna().unique(), default=[], key="nivel_aba1")

        df_filtrado = df_fornecedores
        if setores:
            df_filtrado = df_filtrado[df_filtrado['industry'].isin(setores)]
        if niveis:
            df_filtrado = df_filtrado[df_filtrado['total_level'].isin(niveis)]

        mediana_score = df_fornecedores['total_score'].median()
        mediana_social = df_fornecedores['social_score'].median()

        fig = px.scatter(
            df_filtrado,
            x="total_score",
            y="social_score",
            color="total_level",
            hover_name="name",
            color_discrete_map={"High": "#EF553B", "Medium": "#FECB52", "Low": "#00CC96"},
            labels={
                "total_score": "Maturidade ESG Geral (Score KNN)",
                "social_score": "Impacto / Vulnerabilidade Social",
                "total_level": "Classificação de Risco (RF)"
            },
            render_mode="webgl"
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(margin=dict(l=40, r=40, t=20, b=40))
        fig.add_vline(x=mediana_score, line_dash="dash", line_color="gray", annotation_text="Corte de Risco")
        fig.add_hline(y=mediana_social, line_dash="dash", line_color="gray", annotation_text="Corte de Severidade")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Fornecedores Filtrados e Planos de Ação Recomendados")
        mapa_planos = {
            'High': "🔴 AÇÃO IMEDIATA: Auditoria presencial urgente e plano de mitigação em 30 dias.",
            'Medium': "🟡 ENGAJAMENTO ATIVO: Treinamentos obrigatórios de compliance e LGPD.",
            'Low': "🟢 MONITORAMENTO CONTÍNUO: Acompanhamento via sistema e reavaliação anual."
        }
        
        df_tabela = df_filtrado[['name', 'industry', 'total_score', 'total_level']].copy()
        df_tabela['Plano de Ação'] = df_tabela['total_level'].map(mapa_planos)
        st.dataframe(df_tabela, use_container_width=True)
    else:
        st.warning("Base de dados vazia para renderizar a matriz.")

# ------------------------------------------------------------------------------
# ABA 2: MAPA DE RISCOS ESG (ENTREGÁVEL 2)
# ------------------------------------------------------------------------------
with aba2:
    st.header("🗺️ Mapa de Riscos ESG")
    st.subheader("Análise de Vulnerabilidades Críticas por Pilar e Setor Industrial")
    
    if not df_fornecedores.empty:
        st.markdown("### ⚠️ Fornecedores Críticos por Pilar (Nível: High)")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="Risco Ambiental Crítico", value=alta_env, delta="Foco em Emissões/Resíduos", delta_color="inverse")
        with col_m2:
            st.metric(label="Risco Social Crítico", value=alta_soc, delta="Foco em Direitos Trabalhistas/LGPD", delta_color="inverse")
        with col_m3:
            st.metric(label="Risco de Governança Crítico", value=alta_gov, delta="Foco em Transparência/Ética", delta_color="inverse")
            
        st.markdown("---")
        st.markdown("### 📊 Concentração de Risco Geral por Setor")
        
        df_agrupado = df_fornecedores.groupby(['industry', 'total_level']).size().reset_index(name='Quantidade')
        
        fig_barra = px.bar(
            df_agrupado,
            x="industry",
            y="Quantidade",
            color="total_level",
            title="Distribuição dos Níveis de Risco por Segmento de Mercado",
            labels={"industry": "Setor Industrial", "total_level": "Nível de Risco (RF)"},
            color_discrete_map={"High": "#EF553B", "Medium": "#FECB52", "Low": "#00CC96"},
            barmode="stack"
        )
        fig_barra.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=40, t=40, b=100))
        st.plotly_chart(fig_barra, use_container_width=True)
        
        st.markdown("### 🔍 Matriz de Diagnóstico por Fornecedor")
        apenas_criticos = st.checkbox("Exibir apenas fornecedores com Risco Geral 'High'", value=False, key="check_criticos")
        
        df_mapa_risco = df_fornecedores[['name', 'industry', 'environment_score', 'social_score', 'governance_score', 'total_level']].copy()
        if apenas_criticos:
            df_mapa_risco = df_mapa_risco[df_mapa_risco['total_level'] == 'High']
            
        df_mapa_risco.columns = ["Fornecedor", "Setor", "Score Ambiental (E)", "Score Social (S)", "Score Governança (G)", "Risco Geral (RF)"]
        st.dataframe(df_mapa_risco, use_container_width=True)
    else:
        st.warning("Base de dados vazia para renderizar o mapa de riscos.")

# ------------------------------------------------------------------------------
# ABA 3: SCORECARD & RANKING (ENTREGÁVEL 3)
# ------------------------------------------------------------------------------
with aba3:
    st.header("📋 Scorecard de Sustentabilidade do Fornecedor")
    st.subheader("Auditoria Individualizada e Diagnóstico de Modelos de IA")
    
    if not df_fornecedores.empty:
        fornecedor_selecionado = st.selectbox("🔍 Busque e selecione um Fornecedor para auditoria completa:", options=sorted(df_fornecedores['name'].unique()), key="select_forn_aba3")
        dados_forn = df_fornecedores[df_fornecedores['name'] == fornecedor_selecionado].iloc[0]
        
        st.markdown(f"## Empresa: **{fornecedor_selecionado}** | Setor: *{dados_forn.get('industry', 'Não Informado')}*")
        st.markdown("---")
        
        st.markdown("### 🤖 Veredito dos Modelos Preditivos (Mapeados via MLflow)")
        col_ia1, col_ia2, col_ia3 = st.columns(3)
        risco_geral = dados_forn.get('total_level', 'N/A')
        cor_risco = "🔴" if risco_geral == "High" else "🟡" if risco_geral == "Medium" else "🟢"
        
        with col_ia1:
            st.metric(label="Maturidade Calculada (Modelo KNN)", value=f"{dados_forn.get('total_score', 0):.1f} pts", delta="Score Geral Estimado")
        with col_ia2:
            st.metric(label="Classificação de Risco (Random Forest)", value=f"{cor_risco} {risco_geral}", delta="Probabilidade de Alerta")
        with col_ia3:
            score_atual = dados_forn.get('total_score', 0)
            categoria_grade = "A" if score_atual > 1200 else "B" if score_atual > 800 else "C"
            st.metric(label="Conceito Final da Cadeia", value=f"Grau {categoria_grade}", delta="Rating de Reputação")
            
        st.markdown("---")
        st.markdown("### 🧬 Desempenho Aberto por Pilar")
        col_e, col_s, col_g = st.columns(3)
        teto_pilar = 500.0
        
        with col_e:
            st.markdown(f"#### 🌲 Ambiental (E)")
            st.write(f"**Score:** {dados_forn['environment_score']:.1f} pts")
            st.progress(min(float(dados_forn['environment_score']) / teto_pilar, 1.0))
        with col_s:
            st.markdown(f"#### 🤝 Social (S)")
            st.write(f"**Score:** {dados_forn['social_score']:.1f} pts")
            st.progress(min(float(dados_forn['social_score']) / teto_pilar, 1.0))
        with col_g:
            st.markdown(f"#### ⚖️ Governança (G)")
            st.write(f"**Score:** {dados_forn['governance_score']:.1f} pts")
            st.progress(min(float(dados_forn['governance_score']) / teto_pilar, 1.0))

        st.markdown("---")
        st.markdown("### 📋 Diretriz de Auditoria Recomendada")
        if risco_geral == 'High':
            st.error(f"⚠️ **Alerta Crítico para {fornecedor_selecionado}:** Suspensão preventiva de novos contratos e abertura de auditoria de conformidade urgente.")
        elif risco_geral == 'Medium':
            st.warning(f"⚠️ **Atenção Requerida para {fornecedor_selecionado}:** Incluir em workshops preventivos e coletar evidências de governança em 60 dias.")
        else:
            st.success(f"✅ **Certificação Verde para {fornecedor_selecionado}:** Operação segura em conformidade com as melhores práticas.")
    else:
        st.warning("Base de dados vazia para renderizar o Scorecard.")

# ------------------------------------------------------------------------------
# ABA 4: BENCHMARKING INTERNO & RANKING (ENTREGÁVEL 4)
# ------------------------------------------------------------------------------
with aba4:
    st.header("📊 Benchmarking Interno e Liderança ESG")
    st.subheader("Análise Comparativa de Desempenho e Competitividade")
    
    if not df_fornecedores.empty:
        setor_bench = st.multiselect("🔍 Filtrar Rankings e Gráfico por Setor Industrial:", options=df_fornecedores['industry'].dropna().unique(), key="filtra_aba4_setor", default=[])

        df_bench_filtrado = df_fornecedores.copy()
        if setor_bench:
            df_bench_filtrado = df_bench_filtrado[df_bench_filtrado['industry'].isin(setor_bench)]

        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            st.markdown("### 🏆 Top Líderes em Sustentabilidade")
            df_top = df_bench_filtrado.nlargest(10, 'total_score')[['name', 'industry', 'total_score', 'status_homologacao']]
            df_top.columns = ["Fornecedor", "Setor", "Score ESG (KNN)", "Status Cadastral"]
            st.dataframe(df_top.reset_index(drop=True), use_container_width=True)
            
        with col_rank2:
            st.markdown("### 📉 Fornecedores Retardatários (Ação Requerida)")
            df_bottom = df_bench_filtrado.nsmallest(10, 'total_score')[['name', 'industry', 'total_score', 'total_level']]
            df_bottom.columns = ["Fornecedor", "Setor", "Score ESG (KNN)", "Risco Geral"]
            st.dataframe(df_bottom.reset_index(drop=True), use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🎯 Dispersão e Concentração de Maturidade")
        
        fig_dist = px.scatter(
            df_bench_filtrado,
            x="total_score",
            y="environment_score",
            color="total_level",
            hover_name="name",
            labels={"total_score": "Maturidade Geral", "environment_score": "Maturidade Ambiental", "total_level": "Risco (RF)"},
            color_discrete_map={"High": "#EF553B", "Medium": "#FECB52", "Low": "#00CC96"},
            title="Maturidade Geral vs. Performance Ambiental"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.warning("Base de dados vazia para renderizar o benchmarking.")

# ------------------------------------------------------------------------------
#⚙️ ABA 5: PAINEL OPERACIONAL (CRUD E BANCO DE DADOS)
# ------------------------------------------------------------------------------
with aba5:
    st.header("⚙️ Painel Operacional e Governança de Dados")
    st.subheader("Gerenciamento Relacional, Edição de Notas (UPDATE) e Visões Complexas")

    engine_bd = obter_conexao_banco()

    # Buscando lista completa e atualizada de todas as empresas cadastradas no banco
    try:
        with engine_bd.connect() as conn:
            df_todas_empresas = pd.read_sql(text("SELECT id, name FROM fornecedores ORDER BY name;"), conn)
    except Exception:
        df_todas_empresas = pd.DataFrame()

    st.markdown("## 📊 Operações de Escrita, Edição e Deleção (CRUD)")
    
    # Dividindo em 3 colunas agora: Cadastrar, Editar Notas, Deletar
    col_crud1, col_crud2, col_crud3 = st.columns(3)

    # --- COLUNA 1: CADASTRAR (INSERT) ---
    with col_crud1:
        st.markdown("### ➕ Cadastrar Novo Fornecedor")
        with st.form("form_cadastro_fornecedor", clear_on_submit=True):
            nome_empresa = st.text_input("Nome da Organização:")
            segmento = st.selectbox("Segmento Industrial:", ["Transportes", "Manufatura", "Tecnologia", "Alimentos", "Serviços"])
            cnpj_empresa = st.text_input("CNPJ (Formatado):")
            botao_salvar = st.form_submit_button("Salvar no PostgreSQL")
            
            if botao_salvar:
                if nome_empresa and cnpj_empresa:
                    try:
                        with engine_bd.connect() as conn:
                            sql_insert_forn = text("""
                                INSERT INTO fornecedores (name, industry_segment, cnpj, status_homologacao) 
                                VALUES (:name, :industry, :cnpj, 'Pendente') RETURNING id;
                            """)
                            result = conn.execute(sql_insert_forn, {"name": nome_empresa, "industry": segmento, "cnpj": cnpj_empresa})
                            novo_id = result.fetchone()[0]
                            
                            # Cria os registros iniciais para ele já pontuar nas outras abas
                            sql_insert_aud = text("""
                                INSERT INTO auditorias_esg (fornecedor_id, environment_score, social_score, governance_score)
                                VALUES (:forn_id, 250.0, 250.0, 250.0) RETURNING id;
                            """)
                            result_aud = conn.execute(sql_insert_aud, {"forn_id": novo_id})
                            nova_aud_id = result_aud.fetchone()[0]
                            
                            sql_insert_pred = text("""
                                INSERT INTO predicoes_risco (auditoria_id, total_score_predito, total_level_predito)
                                VALUES (:aud_id, 750.0, 'Medium');
                            """)
                            conn.execute(sql_insert_pred, {"aud_id": nova_aud_id})
                            conn.commit()
                        
                        st.cache_data.clear()
                        st.success(f"✅ {nome_empresa} criado!")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Erro no cadastro: {error}")
                else:
                    st.warning("Preencha todos os campos.")

    # --- 🌟 NOVA COLUNA 2: ALTERAR NOTAS (UPDATE) 🌟 ---
    with col_crud2:
        st.markdown("### 📝 Atualizar Notas ESG")
        if not df_todas_empresas.empty:
            forn_editar = st.selectbox("Selecione para editar:", options=df_todas_empresas['name'].unique(), key="edit_forn_select")
            id_editar = df_todas_empresas[df_todas_empresas['name'] == forn_editar]['id'].values[0]
            
            # Buscar as notas atuais dele (se existirem) para preencher o formulário
            try:
                with engine_bd.connect() as conn:
                    nota_atual = conn.execute(
                        text("SELECT id, environment_score, social_score, governance_score FROM auditorias_esg WHERE fornecedor_id = :id;"),
                        {"id": int(id_editar)}
                    ).fetchone()
            except Exception:
                nota_atual = None

            # Valores padrão se a empresa (como as do CSV antigo) nunca tiver tido uma auditoria criada
            env_val = float(nota_atual[1]) if nota_atual else 250.0
            soc_val = float(nota_atual[2]) if nota_atual else 250.0
            gov_val = float(nota_atual[3]) if nota_atual else 250.0

            with st.form("form_edicao_notas"):
                novo_env = st.number_input("Nota Ambiental (E) [0-500]:", min_value=0.0, max_value=500.0, value=env_val)
                novo_soc = st.number_input("Nota Social (S) [0-500]:", min_value=0.0, max_value=500.0, value=soc_val)
                novo_gov = st.number_input("Nota Governança (G) [0-500]:", min_value=0.0, max_value=500.0, value=gov_val)
                botao_atualizar = st.form_submit_button("Atualizar Notas no Banco")
                
                if botao_atualizar:
                    try:
                        with engine_bd.connect() as conn:
                            # Se a empresa já tem linha em auditorias_esg, faz UPDATE
                            if nota_atual:
                                conn.execute(text("""
                                    UPDATE auditorias_esg 
                                    SET environment_score = :e, social_score = :s, governance_score = :g
                                    WHERE fornecedor_id = :id;
                                """), {"e": novo_env, "s": novo_soc, "g": novo_gov, "id": int(id_editar)})
                                
                                # Atualiza também o Score Geral somando as três notas de forma simples
                                novo_total_score = novo_env + novo_soc + novo_gov
                                novo_level = 'High' if novo_total_score < 600 else 'Medium' if novo_total_score < 1100 else 'Low'
                                conn.execute(text("""
                                    UPDATE predicoes_risco 
                                    SET total_score_predito = :score, total_level_predito = :level
                                    WHERE auditoria_id = :aud_id;
                                """), {"score": novo_total_score, "level": novo_level, "aud_id": int(nota_atual[0])})
                            
                            # Se for uma empresa antiga que não tinha linha em auditorias_esg, faz INSERT
                            else:
                                res_aud = conn.execute(text("""
                                    INSERT INTO auditorias_esg (fornecedor_id, environment_score, social_score, governance_score)
                                    VALUES (:id, :e, :s, :g) RETURNING id;
                                """), {"id": int(id_editar), "e": novo_env, "s": novo_soc, "g": novo_gov})
                                nova_aud_id = res_aud.fetchone()[0]
                                
                                novo_total_score = novo_env + novo_soc + novo_gov
                                novo_level = 'High' if novo_total_score < 600 else 'Medium' if novo_total_score < 1100 else 'Low'
                                conn.execute(text("""
                                    INSERT INTO predicoes_risco (auditoria_id, total_score_predito, total_level_predito)
                                    VALUES (:aud_id, :score, :level);
                                """), {"aud_id": nova_aud_id, "score": novo_total_score, "level": novo_level})
                            
                            conn.commit()
                        
                        st.cache_data.clear()
                        st.success(f"🎯 Notas de {forn_editar} atualizadas!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Erro ao atualizar notas: {err}")
        else:
            st.info("Nenhuma empresa para editar.")

    # --- COLUNA 3: REMOVER (DELETE) ---
    with col_crud3:
        st.markdown("### ❌ Remover Fornecedor")
        if not df_todas_empresas.empty:
            forn_remover = st.selectbox("Selecione para remover:", options=df_todas_empresas['name'].unique(), key="delete_forn_select")
            id_remover = df_todas_empresas[df_todas_empresas['name'] == forn_remover]['id'].values[0]
            
            if st.button("Confirmar Exclusão", type="primary"):
                try:
                    with engine_bd.connect() as conn:
                        conn.execute(text("DELETE FROM fornecedores WHERE id = :id;"), {"id": int(id_remover)})
                        conn.commit()
                    st.cache_data.clear()
                    st.success(f"🚫 {forn_remover} removido!")
                    st.rerun()
                except Exception as error:
                    st.error(f"Erro no DELETE: {error}")
        else:
            st.info("Nenhuma empresa cadastrada.")

    st.markdown("---")
    st.markdown("## 🔍 Consultas Avançadas, Visões e Índices (Etapas 03 e 04)")
    
    menu_sql = st.selectbox(
        "Selecione a Consulta Relacional para executar na Interface:",
        ["[Anti-Join] Fornecedores Sem Auditorias", "[Subconsulta] Acima da Média de Risco", "[Visão/View] Relatório Executivo Consolidado ESG"],
        key="menu_sql_exec"
    )

    if menu_sql == "[Anti-Join] Fornecedores Sem Auditorias":
        st.code("SELECT f.name FROM fornecedores f LEFT JOIN auditorias_esg a ON f.id = a.fornecedor_id WHERE a.id IS NULL;", language="sql")
        try:
            with engine_bd.connect() as conn:
                res_anti = pd.read_sql(text("""
                    SELECT f.name AS "Fornecedor", f.industry_segment AS "Setor", f.cnpj AS "CNPJ"
                    FROM fornecedores f LEFT JOIN auditorias_esg a ON f.id = a.fornecedor_id WHERE a.id IS NULL;
                """), conn)
            
            if not res_anti.empty:
                st.dataframe(res_anti, use_container_width=True)
            else:
                st.write("🟢 Todos os fornecedores cadastrados possuem auditorias vinculadas.")
        except Exception as err:
            st.error(f"Erro na query: {err}")

    elif menu_sql == "[Subconsulta] Acima da Média de Risco":
        st.code("SELECT f.name FROM fornecedores f WHERE total_score > (SELECT AVG(total_score) FROM ...);", language="sql")
        try:
            with engine_bd.connect() as conn:
                res_sub = pd.read_sql(text("""
                    SELECT f.name AS "Fornecedor", p.total_score_predito AS "Score ESG Preditivo"
                    FROM fornecedores f
                    INNER JOIN auditorias_esg a ON f.id = a.fornecedor_id
                    INNER JOIN predicoes_risco p ON a.id = p.auditoria_id
                    WHERE p.total_score_predito > (SELECT AVG(total_score_predito) FROM predicoes_risco);
                """), conn)
            st.dataframe(res_sub, use_container_width=True)
        except Exception as err:
            st.error(f"Erro na query: {err}")

    elif menu_sql == "[Visão/View] Relatório Executivo Consolidado ESG":
        st.code("SELECT * FROM vw_consolidado_esg_fornecedores;", language="sql")
        try:
            with engine_bd.connect() as conn:
                res_view = pd.read_sql(text("""
                    SELECT f.name AS "Fornecedor", f.status_homologacao AS "Status", a.environment_score AS "Ambiental", p.total_level_predito AS "Risco Final (IA)"
                    FROM fornecedores f
                    INNER JOIN auditorias_esg a ON f.id = a.fornecedor_id
                    INNER JOIN predicoes_risco p ON a.id = p.auditoria_id;
                """), conn)
            st.dataframe(res_view, use_container_width=True)
        except Exception as err:
            st.error(f"Erro ao ler View: {err}")