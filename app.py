"""Aplicacao de apoio as Etapas 03 a 06 da disciplina de Modelagem de Banco de Dados.

Estrutura das secoes:
    Modelo       - descricao do esquema relacional e das decisoes de projeto.
    Dashboard    - leitura analitica consolidada sobre a view principal.
    CRUD         - manutencao manual das quatro tabelas-base.
    Consultas    - oito consultas SQL com justificativa semantica.
    Visoes       - as duas views definidas na Etapa 04.
    Rotinas      - funcoes e procedimentos da Etapa 05.
    Trigger      - inspecao da tabela de auditoria mantida pelo trigger.
"""
from __future__ import annotations

import os
import statistics
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

import db

PORT = 8501


def _bootstrap(argv: list[str] | None = None) -> None:
    if not st.runtime.exists():
        import streamlit.web.cli as stcli
        sys.argv = [
            "streamlit", "run", os.path.abspath(__file__),
            "--server.runOnSave", "true",
            "--server.headless", "true",
            "--server.port", str(PORT),
            *(argv or []),
        ]
        sys.exit(stcli.main())


_bootstrap()

st.set_page_config(
    page_title="Prisma Key - Modelagem de BD",
    layout="wide",
    initial_sidebar_state="expanded",
)

PX_TEMPLATE = "plotly_dark"
PILAR_COLORS = {"AMBIENTAL": "#34d399", "SOCIAL": "#60a5fa", "GOVERNANCA": "#a78bfa"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      div[data-testid="stMetric"] {
          background: #1a2129;
          padding: 0.75rem 1rem;
          border-radius: 8px;
          border-left: 3px solid #34d399;
      }
      div[data-testid="stExpander"] {border-radius: 8px;}
      code {font-size: 0.85em;}
    </style>
    """,
    unsafe_allow_html=True,
)

db.init_db()

PAGES = [
    "Modelo Relacional",
    "Dashboard Analitico",
    "Manutencao de Dados",
    "Consultas SQL",
    "Visoes",
    "Rotinas Procedurais",
    "Auditoria via Trigger",
]

with st.sidebar:
    st.title("Prisma Key")
    st.caption("Modelagem de Banco de Dados")
    page = st.radio("Secao", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Volumetria do banco**")
    counts = db.query_df(
        """SELECT
              (SELECT COUNT(*) FROM Empresa)   AS empresas,
              (SELECT COUNT(*) FROM Registro)  AS registros,
              (SELECT COUNT(*) FROM Auditoria) AS auditorias"""
    ).iloc[0]
    st.metric("Empresas",   int(counts.empresas))
    st.metric("Registros",  int(counts.registros))
    st.metric("Auditorias", int(counts.auditorias))
    st.markdown("---")
    if st.button("Reinicializar base", help="Recria o esquema e recarrega o dataset"):
        db.init_db(force=True)
        st.success("Base reinicializada.")


def chart(fig):
    fig.update_layout(template=PX_TEMPLATE, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------
# Modelo
# ------------------------------------------------------------
if page == "Modelo Relacional":
    st.title("Modelo Relacional")
    st.write(
        "O dominio modela a coleta de indicadores ambientais, sociais e de governanca "
        "realizada em unidades operacionais de empresas e submetida a auditoria por "
        "profissionais habilitados. A modelagem segue as exigencias normativas de "
        "rastreabilidade e separacao por pilar ESG."
    )

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.subheader("Esquema relacional")
        st.code(
            """EMPRESA   (id_empresa, nome_fantasia, cnpj, cidade, industria, id_empresa_mae)
UNIDADE   (id_unidade, id_empresa, nome_unidade, localizacao)
CATEGORIA (id_categoria, descricao, tipo)
METRICA   (id_metrica, nome, descricao, id_categoria)
REGISTRO  (id_registro, data_hora, valor_medido, status,
           id_unidade, id_empresa, id_metrica)
AUDITOR   (cpf, nome, registro_profissional)
AUDITORIA (id_auditoria, data_realizacao, parecer_final, cpf_auditor)
AUDITORIA_REGISTRO (id_auditoria, id_registro)
LOG_STATUS_REGISTRO (id_log, id_registro, status_antigo, status_novo,
                      data_alteracao, usuario_sistema)""",
            language="sql",
        )
        st.caption(
            "Auto-relacionamento em EMPRESA (id_empresa_mae) modela holdings. "
            "UNIDADE e entidade fraca, com PK composta (id_unidade, id_empresa). "
            "AUDITORIA_REGISTRO resolve o N:M entre auditorias e registros."
        )

    with col2:
        st.subheader("Decisoes de projeto")
        st.markdown(
            """
- CATEGORIA foi extraida de METRICA para evitar redundancia do pilar ESG em cada coleta (3FN).
- Indices sobre Registro(id_empresa, id_unidade) e Registro(data_hora) cobrem
  os filtros mais frequentes do dashboard.
- Views materializam joins recorrentes, escondendo a complexidade do esquema
  para consumidores analiticos.
- Trigger AFTER UPDATE mantem log_status_registro como tabela de auditoria,
  atendendo a requisitos de rastreabilidade ESG.
- Procedimentos encapsulam operacoes em lote e reavaliacoes que exigem
  percorrer linhas uma a uma (CURSOR).
            """
        )

    st.subheader("Origem dos dados")
    st.write(
        "Base datakaggle.csv (Kaggle / Finnhub) contendo 722 empresas listadas em bolsa. "
        "Para cada empresa sao gerados quatro registros (environment, social, governance "
        "e total score) vinculados a uma unidade sede. O grade ESG informado pelo dataset "
        "(A, BBB, BB, B) e mapeado para o atributo status do registro (VALIDADO, "
        "PENDENTE ou REJEITADO)."
    )


# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
elif page == "Dashboard Analitico":
    st.title("Dashboard Analitico Integrado")
    st.caption(
        "Tres camadas analiticas calculadas em tempo real sobre o banco: visao "
        "transacional (Modelagem de BD), inferencia estatistica e aprendizado de "
        "maquina. Todas as agregacoes sao reexecutadas a cada alteracao dos filtros."
    )

    base = db.query_df(
        """SELECT v.*, e.industria
             FROM vw_resumo_metricas_empresas v
             JOIN Empresa e ON e.nome_fantasia = v.nome_empresa"""
    )
    base["data_hora"] = pd.to_datetime(base["data_hora"])

    with st.container(border=True):
        st.markdown("**Filtros analiticos**")
        c1, c2, c3, c4 = st.columns(4)
        industrias = sorted(base["industria"].dropna().unique().tolist())
        empresas   = sorted(base["nome_empresa"].dropna().unique().tolist())
        sel_ind    = c1.multiselect("Setor industrial", industrias)
        sel_emp    = c2.multiselect("Empresa", empresas)
        sel_pilar  = c3.multiselect("Pilar ESG", ["AMBIENTAL", "SOCIAL", "GOVERNANCA"])
        sel_status = c4.multiselect("Status", ["VALIDADO", "PENDENTE", "REJEITADO"])
        if base["data_hora"].notna().any():
            dmin, dmax = base["data_hora"].min().date(), base["data_hora"].max().date()
            if dmin < dmax:
                rng = st.slider("Periodo de coleta", min_value=dmin, max_value=dmax,
                                 value=(dmin, dmax))
                base = base[(base["data_hora"].dt.date >= rng[0]) &
                             (base["data_hora"].dt.date <= rng[1])]

    if sel_ind:    base = base[base["industria"].isin(sel_ind)]
    if sel_emp:    base = base[base["nome_empresa"].isin(sel_emp)]
    if sel_pilar:  base = base[base["tipo_esg"].isin(sel_pilar)]
    if sel_status: base = base[base["status_registro"].isin(sel_status)]

    if base.empty:
        st.warning("Sem dados para os filtros selecionados.")
        st.stop()

    # ----- Indicadores resumidos -----
    st.subheader("Indicadores resumidos")
    valores = base["valor_medido"]
    n = len(valores)
    media = valores.mean()
    sd = valores.std(ddof=1)
    ep = sd / np.sqrt(n) if n > 1 else 0
    ic = stats.t.interval(0.95, n - 1, loc=media, scale=ep) if n > 1 else (media, media)
    try:
        moda = statistics.mode(valores.round(1).tolist())
    except statistics.StatisticsError:
        moda = float(valores.iloc[0])
    validados = (base["status_registro"] == "VALIDADO").sum()

    k = st.columns(4)
    k[0].metric("Registros no recorte", f"{n:,}")
    k[1].metric("Empresas",             f"{base['nome_empresa'].nunique():,}")
    k[2].metric("Setores industriais",  f"{base['industria'].nunique():,}")
    k[3].metric("Taxa de validacao",    f"{validados/n*100:.1f}%")
    k = st.columns(4)
    k[0].metric("Media",          f"{media:.2f}")
    k[1].metric("Mediana",        f"{valores.median():.2f}")
    k[2].metric("Moda (1 casa)",  f"{moda:.2f}")
    k[3].metric("IC 95% da media", f"[{ic[0]:.2f}; {ic[1]:.2f}]")
    k = st.columns(4)
    k[0].metric("Desvio padrao", f"{sd:.2f}")
    k[1].metric("Variancia",     f"{valores.var(ddof=1):.2f}")
    k[2].metric("Assimetria",    f"{stats.skew(valores):.3f}")
    k[3].metric("Curtose",       f"{stats.kurtosis(valores):.3f}")

    st.divider()

    # ----- 1 e 2: composicoes -----
    st.subheader("Distribuicao por categoria e status")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Registros por pilar ESG**")
        df_p = base.groupby("tipo_esg").size().reset_index(name="qtd")
        chart(px.bar(df_p, x="tipo_esg", y="qtd", color="tipo_esg",
                      color_discrete_map=PILAR_COLORS))
    with g2:
        st.markdown("**Composicao por status de validacao**")
        df_s = base.groupby("status_registro").size().reset_index(name="qtd")
        chart(px.pie(df_s, names="status_registro", values="qtd", hole=0.45))

    # ----- 3 e 4: frequencia e dispersao -----
    st.subheader("Distribuicao do valor medido e dispersao")
    g3, g4 = st.columns(2)
    with g3:
        st.markdown("**Histograma de frequencia (com KDE de referencia)**")
        chart(px.histogram(base, x="valor_medido", nbins=30, color="tipo_esg",
                           color_discrete_map=PILAR_COLORS, barmode="overlay",
                           opacity=0.75, marginal="rug"))
    with g4:
        st.markdown("**Boxplot por pilar - visualiza IQR e outliers**")
        chart(px.box(base, x="tipo_esg", y="valor_medido", color="tipo_esg",
                      color_discrete_map=PILAR_COLORS, points="outliers"))

    # ----- 5: serie temporal -----
    st.subheader("Tendencia temporal")
    if base["data_hora"].notna().any():
        tmp = base.assign(mes=base["data_hora"].dt.to_period("M").astype(str)) \
                  .groupby(["mes", "tipo_esg"])["valor_medido"].mean().reset_index()
        chart(px.line(tmp, x="mes", y="valor_medido", color="tipo_esg",
                       color_discrete_map=PILAR_COLORS, markers=True))

    # ----- 6: setores -----
    st.subheader("Comparacao por setor industrial")
    g5, g6 = st.columns(2)
    with g5:
        st.markdown("**Dispersao por setor (top 15)**")
        top15 = base["industria"].value_counts().head(15).index
        chart(px.box(base[base["industria"].isin(top15)],
                     x="industria", y="valor_medido", color="tipo_esg",
                     color_discrete_map=PILAR_COLORS).update_xaxes(tickangle=-30))
    with g6:
        st.markdown("**Composicao de status por setor (top 10)**")
        top10 = base["industria"].value_counts().head(10).index
        df_si = base[base["industria"].isin(top10)] \
                    .groupby(["industria", "status_registro"]).size().reset_index(name="qtd")
        chart(px.bar(df_si, x="industria", y="qtd", color="status_registro")
                .update_xaxes(tickangle=-30))

    # ----- 7: heatmap setor x metrica -----
    st.subheader("Perfil ESG agregado")
    g7, g8 = st.columns(2)
    with g7:
        st.markdown("**Radar de medias por categoria**")
        df_c = base.groupby("categoria_descricao")["valor_medido"].mean().reset_index()
        fig = px.line_polar(df_c, r="valor_medido", theta="categoria_descricao",
                             line_close=True)
        fig.update_traces(fill="toself", line_color="#34d399")
        chart(fig)
    with g8:
        st.markdown("**Mapa de calor: setor x metrica (media)**")
        top15 = base["industria"].value_counts().head(15).index
        piv_hm = base[base["industria"].isin(top15)] \
                    .pivot_table(index="industria", columns="nome_metrica",
                                 values="valor_medido", aggfunc="mean").fillna(0)
        chart(px.imshow(piv_hm, aspect="auto", color_continuous_scale="Viridis"))

    # ----- correlacao entre pilares -----
    st.subheader("Correlacao entre pilares ESG")
    st.caption(
        "Construida pivotando o valor medido por empresa e pilar. "
        "Apenas empresas com observacoes nos tres pilares entram no calculo."
    )
    pivot_pilar = base.pivot_table(index="nome_empresa", columns="tipo_esg",
                                    values="valor_medido", aggfunc="mean").dropna()
    if len(pivot_pilar) >= 3:
        corr = pivot_pilar.corr().round(3)
        c1, c2 = st.columns([1.1, 1])
        with c1:
            chart(px.imshow(corr, text_auto=True, zmin=-1, zmax=1,
                             color_continuous_scale="RdBu_r"))
        with c2:
            pares = []
            cols = list(corr.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pares.append({"Pilar A": cols[i], "Pilar B": cols[j],
                                  "r": corr.iloc[i, j]})
            st.markdown("**Pares e coeficientes**")
            st.dataframe(pd.DataFrame(pares), width="stretch", hide_index=True)
    else:
        st.info("Filtro atual produziu poucos pontos para correlacao.")

    # ----- teste de hipoteses -----
    st.subheader("Comparacao formal entre dois setores")
    st.caption(
        "Teste de Welch (parametrico) e Mann-Whitney (nao parametrico) com "
        "alfa = 0,05. H0: medias iguais; H1: medias diferentes."
    )
    ind_disp = base["industria"].value_counts()
    ind_disp = ind_disp[ind_disp >= 5].index.tolist()
    if len(ind_disp) >= 2:
        c1, c2 = st.columns(2)
        ia = c1.selectbox("Setor A", ind_disp, index=0)
        ib = c2.selectbox("Setor B", ind_disp, index=1 if len(ind_disp) > 1 else 0)
        ga = base.loc[base["industria"] == ia, "valor_medido"].dropna()
        gb = base.loc[base["industria"] == ib, "valor_medido"].dropna()
        if ia != ib and len(ga) >= 3 and len(gb) >= 3:
            lev = stats.levene(ga, gb)
            tt = stats.ttest_ind(ga, gb, equal_var=False)
            mw = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            k = st.columns(4)
            k[0].metric(f"Media {ia}", f"{ga.mean():.2f}", f"n = {len(ga)}")
            k[1].metric(f"Media {ib}", f"{gb.mean():.2f}", f"n = {len(gb)}")
            k[2].metric("t de Welch",    f"{tt.statistic:.3f}",
                         f"p = {tt.pvalue:.2e}")
            k[3].metric("Mann-Whitney U", f"{mw.statistic:.1f}",
                         f"p = {mw.pvalue:.2e}")
            decisao = "Rejeita H0 - medias diferentes" if tt.pvalue < 0.05 \
                      else "Nao rejeita H0"
            st.info(f"Levene p = {lev.pvalue:.4f} (homogeneidade de variancias). "
                     f"Decisao em alfa = 0,05: **{decisao}**.")
        else:
            st.info("Selecione dois setores diferentes com pelo menos 3 observacoes.")
    else:
        st.info("Filtro atual nao tem setores com tamanho amostral suficiente.")

    # ----- modelo preditivo -----
    st.subheader("Modelo preditivo treinado sobre os dados filtrados")
    st.caption(
        "KNN para regressao do total ESG e Random Forest para classificacao em grades "
        "(A, BBB, BB, B) derivados por quartil. Validacao por hold-out 80/20 e CV "
        "5-fold. Permite inferencia ao vivo abaixo."
    )
    wide = base.pivot_table(index=["nome_empresa", "industria"], columns="tipo_esg",
                             values="valor_medido", aggfunc="mean").dropna()
    wide.columns = [c.lower() for c in wide.columns]
    if {"ambiental", "social", "governanca"}.issubset(wide.columns):
        wide["total"] = wide[["ambiental", "social", "governanca"]].sum(axis=1)
        wide = wide.reset_index()
        n_amostras = len(wide)
        if n_amostras >= 40:
            wide["grade"] = pd.qcut(wide["total"], q=4,
                                     labels=["B", "BB", "BBB", "A"]).astype(str)
            X = np.asarray(wide[["ambiental", "social", "governanca"]], dtype=float)
            y_reg = np.asarray(wide["total"], dtype=float)
            y_cls = np.asarray(wide["grade"].astype(str))
            X_tr, X_te, yr_tr, yr_te = train_test_split(
                X, y_reg, test_size=0.2, random_state=42)
            _, _, yc_tr, yc_te = train_test_split(
                X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

            knn = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(X_tr, yr_tr)
            yhat = knn.predict(X_te)
            rmse = float(np.sqrt(((yhat - yr_te) ** 2).mean()))
            r2   = float(r2_score(yr_te, yhat))
            cv_r2 = cross_val_score(knn, X, y_reg, cv=5, scoring="r2").mean()

            rf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                         class_weight="balanced",
                                         random_state=42).fit(X_tr, yc_tr)
            yhat_c = rf.predict(X_te)
            acc = accuracy_score(yc_te, yhat_c)
            f1  = f1_score(yc_te, yhat_c, average="macro")
            cv_f1 = cross_val_score(rf, X, y_cls, cv=5, scoring="f1_macro").mean()

            k = st.columns(4)
            k[0].metric("n treino",            f"{len(X_tr)}")
            k[1].metric("n teste",             f"{len(X_te)}")
            k[2].metric("RMSE (KNN)",          f"{rmse:.2f}")
            k[3].metric("R^2 hold-out / 5-fold", f"{r2:.3f} / {cv_r2:.3f}")
            k = st.columns(4)
            k[0].metric("Acuracia (RF)",            f"{acc:.3f}")
            k[1].metric("F1 macro hold-out / 5-fold", f"{f1:.3f} / {cv_f1:.3f}")
            k[2].metric("Profundidade max",          "12")
            k[3].metric("n arvores",                  "300")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Importancia das features (Random Forest)**")
                imp = pd.DataFrame({
                    "Pilar": ["AMBIENTAL", "SOCIAL", "GOVERNANCA"],
                    "Importancia": rf.feature_importances_,
                }).sort_values("Importancia", ascending=True)
                chart(px.bar(imp, x="Importancia", y="Pilar", orientation="h",
                              color="Pilar", color_discrete_map=PILAR_COLORS))
            with c2:
                st.markdown("**Predito x observado (hold-out)**")
                df_pred = pd.DataFrame({"observado": yr_te, "predito": yhat})
                fig = px.scatter(df_pred, x="observado", y="predito")
                fig.add_shape(type="line",
                               x0=float(df_pred["observado"].min()),
                               y0=float(df_pred["observado"].min()),
                               x1=float(df_pred["observado"].max()),
                               y1=float(df_pred["observado"].max()),
                               line=dict(dash="dash", color="#9ca3af"))
                chart(fig)

            st.markdown("**Inferencia ao vivo**")
            c1, c2, c3 = st.columns(3)
            v_amb = c1.number_input("Ambiental",
                                     min_value=0.0, value=float(wide["ambiental"].median()))
            v_soc = c2.number_input("Social",
                                     min_value=0.0, value=float(wide["social"].median()))
            v_gov = c3.number_input("Governanca",
                                     min_value=0.0, value=float(wide["governanca"].median()))
            x_new = np.array([[v_amb, v_soc, v_gov]])
            k = st.columns(2)
            k[0].metric("ESG total predito (KNN)", f"{float(knn.predict(x_new)[0]):.2f}")
            k[1].metric("Grade classificado (RF)", str(rf.predict(x_new)[0]))
        else:
            st.info(f"Sao necessarias pelo menos 40 empresas com os tres pilares "
                     f"para treinar o modelo. Recorte atual: {n_amostras}.")
    else:
        st.info("Filtro atual nao cobre os tres pilares ESG simultaneamente.")


# ------------------------------------------------------------
# CRUD
# ------------------------------------------------------------
elif page == "Manutencao de Dados":
    st.title("Manutencao de Dados")
    st.caption(
        "Operacoes de CRUD sobre quatro entidades. Tentativas de violar restricoes de "
        "integridade (CHECK, chaves estrangeiras ou triggers) sao recusadas pelo SGBD "
        "e exibidas como mensagens de erro nesta interface."
    )

    tabs = st.tabs(["Empresa", "Unidade", "Metrica", "Registro"])

    # Empresa
    with tabs[0]:
        st.subheader("Empresa")
        st.caption("Auto-relacionamento via id_empresa_mae modela holdings.")
        st.dataframe(db.query_df("SELECT * FROM Empresa LIMIT 200"), width="stretch")

        with st.expander("Inserir empresa"):
            c1, c2 = st.columns(2)
            nome = c1.text_input("Nome fantasia", key="e_nome")
            cnpj = c2.text_input("CNPJ (unique)", key="e_cnpj")
            c3, c4 = st.columns(2)
            cidade    = c3.text_input("Exchange / cidade", key="e_cid")
            industria = c4.text_input("Industria", key="e_ind")
            mae = st.number_input("ID empresa mae (0 = nenhuma)", min_value=0, value=0, step=1, key="e_mae")
            if st.button("Inserir", key="b_e_ins"):
                try:
                    db.execute(
                        """INSERT INTO Empresa
                              (nome_fantasia, cnpj, cidade, industria, id_empresa_mae)
                           VALUES (?,?,?,?,?)""",
                        (nome, cnpj, cidade, industria, mae or None),
                    )
                    st.success("Empresa inserida.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        with st.expander("Alterar ou excluir"):
            ids = db.query_df("SELECT id_empresa, nome_fantasia FROM Empresa ORDER BY nome_fantasia")
            eid = st.selectbox("Empresa", ids["id_empresa"],
                                format_func=lambda i: f"{i} - {ids[ids.id_empresa==i].nome_fantasia.values[0]}")
            c1, c2 = st.columns(2)
            novo_nome   = c1.text_input("Novo nome fantasia", key="e_upd_nome")
            nova_cidade = c2.text_input("Nova cidade", key="e_upd_cid")
            b1, b2 = st.columns(2)
            if b1.button("Atualizar empresa"):
                db.execute(
                    """UPDATE Empresa SET
                          nome_fantasia = COALESCE(NULLIF(?,''), nome_fantasia),
                          cidade        = COALESCE(NULLIF(?,''), cidade)
                       WHERE id_empresa = ?""",
                    (novo_nome, nova_cidade, int(eid)),
                )
                st.success("Atualizado.")
                st.rerun()
            if b2.button("Excluir empresa", type="secondary"):
                try:
                    db.execute("DELETE FROM Empresa WHERE id_empresa = ?", (int(eid),))
                    st.success("Excluida.")
                    st.rerun()
                except Exception as e:
                    st.error(f"FK impediu exclusao: {e}")

    # Unidade
    with tabs[1]:
        st.subheader("Unidade")
        st.caption("Entidade fraca. PK composta (id_unidade, id_empresa).")
        st.dataframe(db.query_df(
            """SELECT u.id_unidade, u.id_empresa, e.nome_fantasia, u.nome_unidade, u.localizacao
                 FROM Unidade u JOIN Empresa e ON e.id_empresa = u.id_empresa
                 LIMIT 200"""
        ), width="stretch")

        with st.expander("Inserir unidade"):
            emps = db.query_df("SELECT id_empresa, nome_fantasia FROM Empresa ORDER BY nome_fantasia")
            eid = st.selectbox("Empresa", emps["id_empresa"], key="u_eid",
                                format_func=lambda i: emps[emps.id_empresa==i].nome_fantasia.values[0])
            c1, c2 = st.columns(2)
            uid = c1.number_input("ID unidade", min_value=1, step=1, key="u_uid")
            nome = c2.text_input("Nome unidade", key="u_nome")
            loc = st.text_input("Localizacao", key="u_loc")
            if st.button("Inserir unidade"):
                try:
                    db.execute(
                        "INSERT INTO Unidade (id_unidade, id_empresa, nome_unidade, localizacao) VALUES (?,?,?,?)",
                        (int(uid), int(eid), nome, loc),
                    )
                    st.success("Unidade inserida.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    # Metrica
    with tabs[2]:
        st.subheader("Metrica")
        st.caption("Cada metrica pertence a uma categoria; a categoria define o pilar ESG.")
        st.dataframe(db.query_df(
            """SELECT m.id_metrica, m.nome, c.descricao AS categoria, c.tipo AS pilar
                 FROM Metrica m JOIN Categoria c ON c.id_categoria = m.id_categoria"""
        ), width="stretch")
        cats = db.query_df("SELECT id_categoria, descricao FROM Categoria")
        with st.expander("Inserir metrica"):
            cid = st.selectbox("Categoria", cats["id_categoria"],
                                format_func=lambda i: cats[cats.id_categoria==i].descricao.values[0])
            nome = st.text_input("Nome", key="m_nome")
            desc = st.text_area("Descricao", key="m_desc")
            if st.button("Inserir metrica"):
                db.execute("INSERT INTO Metrica (nome, descricao, id_categoria) VALUES (?,?,?)",
                            (nome, desc, int(cid)))
                st.success("Metrica inserida.")
                st.rerun()

    # Registro
    with tabs[3]:
        st.subheader("Registro")
        st.caption(
            "Tabela transacional principal. INSERT com valor_medido < 0 e bloqueado por "
            "trg_valida_valor_medido (BEFORE INSERT). UPDATE de status grava em "
            "log_status_registro via trg_logar_mudanca_status (AFTER UPDATE)."
        )
        st.dataframe(db.query_df(
            """SELECT r.id_registro, r.data_hora, r.valor_medido, r.status,
                      e.nome_fantasia, u.nome_unidade, m.nome AS metrica
                 FROM Registro r
                 JOIN Empresa  e ON e.id_empresa = r.id_empresa
                 JOIN Unidade  u ON u.id_unidade = r.id_unidade AND u.id_empresa = r.id_empresa
                 JOIN Metrica  m ON m.id_metrica = r.id_metrica
                 ORDER BY r.data_hora DESC LIMIT 200"""
        ), width="stretch")

        with st.expander("Inserir registro"):
            uns = db.query_df(
                """SELECT u.id_unidade, u.id_empresa, u.nome_unidade, e.nome_fantasia
                     FROM Unidade u JOIN Empresa e ON e.id_empresa=u.id_empresa
                     LIMIT 200"""
            )
            ms = db.query_df("SELECT id_metrica, nome FROM Metrica")
            idx = st.selectbox("Unidade", uns.index,
                                format_func=lambda i: f"{uns.loc[i,'nome_fantasia']} - {uns.loc[i,'nome_unidade']}")
            mid = st.selectbox("Metrica", ms["id_metrica"],
                                format_func=lambda i: ms[ms.id_metrica==i].nome.values[0], key="r_m")
            c1, c2, c3 = st.columns(3)
            valor  = c1.number_input("Valor medido", value=100.0, step=0.01)
            status = c2.selectbox("Status", ["PENDENTE", "VALIDADO", "REJEITADO"])
            data   = c3.date_input("Data", datetime.now())
            if st.button("Inserir registro"):
                try:
                    db.execute(
                        """INSERT INTO Registro
                              (data_hora, valor_medido, status, id_unidade, id_empresa, id_metrica)
                           VALUES (?,?,?,?,?,?)""",
                        (f"{data} 09:00:00", float(valor), status,
                         int(uns.loc[idx,"id_unidade"]), int(uns.loc[idx,"id_empresa"]), int(mid)),
                    )
                    st.success("Registro inserido.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Restricao acionada: {e}")

        with st.expander("Alterar status"):
            regs = db.query_df("SELECT id_registro, status FROM Registro ORDER BY id_registro DESC LIMIT 50")
            rid = st.selectbox("Registro", regs["id_registro"], key="r_alt")
            novo_status = st.selectbox("Novo status", ["PENDENTE", "VALIDADO", "REJEITADO"], key="r_st")
            if st.button("Atualizar status"):
                db.execute("UPDATE Registro SET status=? WHERE id_registro=?", (novo_status, int(rid)))
                st.success("Status atualizado. A entrada correspondente foi gravada em log_status_registro.")
                st.rerun()


# ------------------------------------------------------------
# Consultas
# ------------------------------------------------------------
elif page == "Consultas SQL":
    st.title("Consultas SQL")
    st.caption(
        "Cada consulta e apresentada com a etapa correspondente, a tecnica relacional "
        "empregada e a justificativa semantica que sustenta sua existencia no dominio."
    )

    CONSULTAS = {
        "Q1 - Coletas com pilar ESG (3 joins)": dict(
            etapa="Etapa 03",
            tecnica="INNER JOIN multi-tabela (Registro x Empresa x Metrica x Categoria)",
            porque=(
                "Visao operacional: relatorio de quem coletou o que e em qual pilar. "
                "Resolve a desnormalizacao logica sem perder a normalizacao fisica."),
            sql="""SELECT r.id_registro, r.data_hora, r.valor_medido, r.status,
       e.nome_fantasia, m.nome AS metrica, c.tipo AS pilar_esg
  FROM Registro r
  JOIN Empresa   e ON e.id_empresa  = r.id_empresa
  JOIN Metrica   m ON m.id_metrica  = r.id_metrica
  JOIN Categoria c ON c.id_categoria = m.id_categoria
 ORDER BY r.data_hora DESC
 LIMIT 200""",
            chart="bar",
        ),
        "Q2 - Media de valor por pilar ESG": dict(
            etapa="Etapa 03",
            tecnica="JOIN com agregacao COUNT e AVG agrupando por pilar",
            porque="Sintese: em qual pilar a empresa concentra coletas e qual e o nivel medio.",
            sql="""SELECT c.tipo AS pilar, COUNT(*) AS total, AVG(r.valor_medido) AS media
  FROM Registro r
  JOIN Metrica   m ON m.id_metrica   = r.id_metrica
  JOIN Categoria c ON c.id_categoria = m.id_categoria
 GROUP BY c.tipo""",
            chart="bar",
        ),
        "Q3 - Empresas com mais coletas pendentes": dict(
            etapa="Etapa 03",
            tecnica="JOIN + filtro WHERE + GROUP BY ordenado",
            porque="Aponta gargalos no fluxo de validacao para o time de compliance priorizar.",
            sql="""SELECT e.nome_fantasia, COUNT(*) AS pendentes
  FROM Registro r
  JOIN Empresa  e ON e.id_empresa = r.id_empresa
 WHERE r.status = 'PENDENTE'
 GROUP BY e.nome_fantasia
 ORDER BY pendentes DESC
 LIMIT 20""",
            chart="bar",
        ),
        "Q4 - Auditorias por auditor": dict(
            etapa="Etapa 03",
            tecnica="LEFT JOIN preservando auditores sem auditoria",
            porque="Gestao de equipe: identifica auditores ociosos ou sobrecarregados.",
            sql="""SELECT a.nome, COUNT(aud.id_auditoria) AS qtd_auditorias
  FROM Auditor a
  LEFT JOIN Auditoria aud ON aud.cpf_auditor = a.cpf
 GROUP BY a.nome
 ORDER BY qtd_auditorias DESC""",
            chart="bar",
        ),
        "Q5 - Empresas sem unidades (anti-join)": dict(
            etapa="Etapa 04",
            tecnica="Anti-join via LEFT JOIN com filtro IS NULL",
            porque=(
                "Qualidade de cadastro: detecta empresas registradas sem nenhuma unidade. "
                "Situacao invalida em producao, util para limpeza."),
            sql="""SELECT e.id_empresa, e.nome_fantasia, e.cnpj
  FROM Empresa e
  LEFT JOIN Unidade u ON e.id_empresa = u.id_empresa
 WHERE u.id_unidade IS NULL""",
        ),
        "Q6 - Auditores x Auditorias (full outer)": dict(
            etapa="Etapa 04",
            tecnica="FULL OUTER JOIN simulado por LEFT JOIN UNION LEFT JOIN invertido",
            porque=(
                "Conciliacao: lista qualquer auditor sem auditorias e qualquer auditoria orfa, "
                "garantindo cobertura dos dois lados."),
            sql="""SELECT a.nome AS auditor_nome, a.cpf, aud.id_auditoria, aud.data_realizacao
  FROM Auditor a
  LEFT JOIN Auditoria aud ON a.cpf = aud.cpf_auditor
UNION
SELECT a.nome, a.cpf, aud.id_auditoria, aud.data_realizacao
  FROM Auditoria aud
  LEFT JOIN Auditor a ON aud.cpf_auditor = a.cpf""",
        ),
        "Q7 - Metricas usadas em coletas validadas": dict(
            etapa="Etapa 04",
            tecnica="Subconsulta com IN sobre JOIN interno",
            porque=(
                "Curadoria do catalogo de metricas: lista somente as metricas que ja passaram "
                "por auditoria com status VALIDADO."),
            sql="""SELECT m.id_metrica, m.nome AS nome_metrica, m.descricao
  FROM Metrica m
 WHERE m.id_metrica IN (
       SELECT r.id_metrica
         FROM Registro r
         JOIN Auditoria_Registro ar ON r.id_registro = ar.id_registro
        WHERE r.status = 'VALIDADO'
 )""",
        ),
        "Q8 - Empresas acima da media geral": dict(
            etapa="Etapa 04",
            tecnica="Agregacao com HAVING comparando contra subconsulta escalar",
            porque=(
                "Benchmark: destaca empresas cuja media de valor medido supera a media "
                "global de todas as coletas."),
            sql="""SELECT e.id_empresa, e.nome_fantasia,
       AVG(r.valor_medido) AS media_medida_empresa
  FROM Empresa e
  JOIN Registro r ON e.id_empresa = r.id_empresa
 GROUP BY e.id_empresa, e.nome_fantasia
HAVING AVG(r.valor_medido) > (SELECT AVG(valor_medido) FROM Registro)
 ORDER BY media_medida_empresa DESC
 LIMIT 25""",
            chart="bar",
        ),
    }

    escolha = st.selectbox("Consulta", list(CONSULTAS.keys()))
    info = CONSULTAS[escolha]

    with st.container(border=True):
        c1, c2 = st.columns([1, 4])
        c1.markdown(f"**{info['etapa']}**")
        c2.markdown(f"**Tecnica empregada:** {info['tecnica']}")
        st.markdown(f"**Justificativa semantica:** {info['porque']}")

    st.code(info["sql"], language="sql")
    df = db.query_df(info["sql"])
    st.markdown(f"Conjunto resultado: {len(df)} tupla(s).")
    st.dataframe(df, width="stretch")

    if info.get("chart") == "bar" and not df.empty:
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num:
            fig = px.bar(df.head(25), x=df.columns[0], y=num[-1])
            fig.update_xaxes(tickangle=-30)
            chart(fig)


# ------------------------------------------------------------
# Views
# ------------------------------------------------------------
elif page == "Visoes":
    st.title("Visoes")
    st.caption(
        "Encapsulam joins recorrentes em um modelo de leitura simplificado. As duas "
        "visoes definidas reunem ao menos tres joins e atendem a finalidades distintas "
        "do dominio."
    )

    VIEWS = {
        "vw_resumo_metricas_empresas": dict(
            porque=(
                "Concentra a leitura analitica de coletas ESG: une Registro, Empresa, "
                "Metrica e Categoria, devolvendo ja o pilar ESG ao consumidor. "
                "Evita repetir o mesmo JOIN em diversas consultas."),
            sql="""CREATE VIEW vw_resumo_metricas_empresas AS
SELECT r.id_registro, r.data_hora, r.valor_medido,
       r.status              AS status_registro,
       e.nome_fantasia       AS nome_empresa,
       e.cnpj                AS documento_empresa,
       m.nome                AS nome_metrica,
       c.descricao           AS categoria_descricao,
       c.tipo                AS tipo_esg
  FROM Registro  r
  JOIN Empresa   e ON r.id_empresa  = e.id_empresa
  JOIN Metrica   m ON r.id_metrica  = m.id_metrica
  JOIN Categoria c ON m.id_categoria = c.id_categoria;""",
        ),
        "vw_detalhes_auditoria_completa": dict(
            porque=(
                "Suporta o painel de auditoria: une Auditoria, Auditor, Auditoria_Registro "
                "e Registro com LEFT JOIN para nao perder auditorias sem registros vinculados. "
                "Substrato natural para o relatorio de conformidade."),
            sql="""CREATE VIEW vw_detalhes_auditoria_completa AS
SELECT aud.id_auditoria, aud.data_realizacao, aud.parecer_final,
       a.nome                  AS auditor_responsavel,
       a.registro_profissional,
       r.id_registro, r.valor_medido,
       r.status                AS status_atual_registro
  FROM Auditoria aud
  LEFT JOIN Auditor             a  ON aud.cpf_auditor = a.cpf
  LEFT JOIN Auditoria_Registro  ar ON aud.id_auditoria = ar.id_auditoria
  LEFT JOIN Registro            r  ON ar.id_registro = r.id_registro;""",
        ),
    }

    view = st.radio("Visao selecionada", list(VIEWS), horizontal=True)
    with st.container(border=True):
        st.markdown(f"**Justificativa semantica:** {VIEWS[view]['porque']}")
    st.code(VIEWS[view]["sql"], language="sql")
    df = db.query_df(f"SELECT * FROM {view} LIMIT 500")
    st.markdown(f"Amostra exibida: {len(df)} tuplas.")
    st.dataframe(df, width="stretch")

    if view == "vw_resumo_metricas_empresas" and not df.empty:
        df["data_hora"] = pd.to_datetime(df["data_hora"])
        fig = px.scatter(df, x="data_hora", y="valor_medido", color="tipo_esg",
                         color_discrete_map=PILAR_COLORS,
                         hover_data=["nome_empresa", "nome_metrica"])
        chart(fig)


# ------------------------------------------------------------
# Rotinas
# ------------------------------------------------------------
elif page == "Rotinas Procedurais":
    st.title("Funcoes e Procedimentos")
    st.caption(
        "Rotinas da Etapa 05. As definicoes originais foram escritas em PL/SQL para MySQL. "
        "Para preservar a portabilidade da aplicacao em SQLite, a mesma logica e executada "
        "em Python mantendo integralmente a semantica das estruturas declaradas "
        "(IF/ELSEIF/ELSE, CURSOR com FETCH e HANDLER NOT FOUND)."
    )

    st.subheader("Funcao 1 - classificar_impacto_registro")
    st.write(
        "Classifica um valor_medido em tres faixas de impacto a partir de estrutura "
        "condicional IF/ELSEIF/ELSE. Aplicada em relatorios para sinalizar coletas "
        "que demandam investigacao adicional."
    )
    st.code(
        """CREATE FUNCTION classificar_impacto_registro(valor DECIMAL(10,2))
RETURNS VARCHAR(50)
BEGIN
  IF valor <= 100      THEN RETURN 'IMPACTO BAIXO / CONFORME';
  ELSEIF valor <= 500  THEN RETURN 'IMPACTO MODERADO / ATENCAO';
  ELSE                       RETURN 'IMPACTO CRITICO / ALERTA';
  END IF;
END;""",
        language="sql",
    )
    valor = st.number_input("Valor medido para classificar", min_value=0.0, value=150.0, step=10.0)
    st.success(f"Resultado: {db.classificar_impacto_registro(valor)}")

    st.divider()
    st.subheader("Funcao 2 - obter_nome_categoria_da_metrica")
    st.write(
        "Encapsula o JOIN entre Metrica e Categoria em uma funcao escalar, permitindo "
        "obter a descricao da categoria associada a uma metrica em qualquer SELECT "
        "sem reescrever o join."
    )
    st.code(
        """CREATE FUNCTION obter_nome_categoria_da_metrica(p_id_metrica INT)
RETURNS VARCHAR(255)
BEGIN
  DECLARE v_descricao VARCHAR(255);
  SELECT c.descricao INTO v_descricao
    FROM Metrica m
    JOIN Categoria c ON m.id_categoria = c.id_categoria
   WHERE m.id_metrica = p_id_metrica;
  RETURN IFNULL(v_descricao, 'Categoria Nao Encontrada');
END;""",
        language="sql",
    )
    ms = db.query_df("SELECT id_metrica, nome FROM Metrica")
    mid = st.selectbox("Metrica", ms["id_metrica"],
                        format_func=lambda i: ms[ms.id_metrica==i].nome.values[0])
    st.success(f"Categoria: {db.obter_nome_categoria_da_metrica(int(mid))}")

    st.divider()
    st.subheader("Procedimento 1 - atualizar_status_coletas_unidade")
    st.write(
        "Operacao de atualizacao em lote. Permite que o revisor altere de uma so vez o "
        "status de todos os registros pendentes de uma unidade especifica, mantendo a "
        "regra de negocio centralizada no banco."
    )
    st.code(
        """CREATE PROCEDURE atualizar_status_coletas_unidade(
    IN p_id_unidade INT, IN p_id_empresa INT, IN p_novo_status VARCHAR(20))
BEGIN
  UPDATE Registro
     SET status = p_novo_status
   WHERE id_unidade = p_id_unidade
     AND id_empresa = p_id_empresa
     AND status     = 'PENDENTE';
END;""",
        language="sql",
    )
    uns = db.query_df(
        """SELECT u.id_unidade, u.id_empresa, u.nome_unidade, e.nome_fantasia
             FROM Unidade u JOIN Empresa e ON e.id_empresa = u.id_empresa
             LIMIT 300"""
    )
    c1, c2 = st.columns(2)
    idx = c1.selectbox("Unidade", uns.index,
                        format_func=lambda i: f"{uns.loc[i,'nome_fantasia']} - {uns.loc[i,'nome_unidade']}")
    novo_status = c2.selectbox("Novo status para PENDENTEs", ["VALIDADO", "REJEITADO"])
    if st.button("Executar procedimento 1"):
        n = db.proc_atualizar_status_coletas_unidade(
            int(uns.loc[idx,"id_unidade"]), int(uns.loc[idx,"id_empresa"]), novo_status)
        st.success(f"Linhas afetadas: {n}")

    st.divider()
    st.subheader("Procedimento 2 - reavaliar_registros_rejeitados")
    st.write(
        "Justificativa para o uso de CURSOR: a regra de negocio exige inspecionar cada "
        "registro rejeitado individualmente. Quando o valor medido excede o limite tecnico "
        "(5000), o registro deve ser zerado e devolvido ao status PENDENTE. Trata-se de "
        "uma transformacao orientada a linha, que nao pode ser expressa como um unico "
        "UPDATE sem violar a regra."
    )
    st.code(
        """CREATE PROCEDURE reavaliar_registros_rejeitados()
BEGIN
  DECLARE v_id INT; DECLARE v_valor DECIMAL(10,2);
  DECLARE done INT DEFAULT 0;
  DECLARE cur CURSOR FOR
     SELECT id_registro, valor_medido FROM Registro WHERE status = 'REJEITADO';
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

  OPEN cur;
  FETCH cur INTO v_id, v_valor;
  WHILE done = 0 DO
    IF v_valor > 5000 THEN
      UPDATE Registro
         SET status = 'PENDENTE', valor_medido = 0.00
       WHERE id_registro = v_id;
    END IF;
    FETCH cur INTO v_id, v_valor;
  END WHILE;
  CLOSE cur;
END;""",
        language="sql",
    )
    if st.button("Executar procedimento 2"):
        n = db.proc_reavaliar_registros_rejeitados()
        st.success(f"Registros normalizados pelo cursor: {n}")


# ------------------------------------------------------------
# Trigger
# ------------------------------------------------------------
elif page == "Auditoria via Trigger":
    st.title("Tabela log_status_registro")
    st.caption(
        "Tabela populada exclusivamente pelo trigger trg_logar_mudanca_status, definido "
        "como AFTER UPDATE sobre Registro. Implementa o requisito de rastreabilidade "
        "ESG: cada transicao de status e historizada de forma automatica e nao pode ser "
        "manipulada pela aplicacao."
    )

    st.code(
        """CREATE TRIGGER trg_logar_mudanca_status
AFTER UPDATE ON Registro
FOR EACH ROW
WHEN OLD.status <> NEW.status
BEGIN
  INSERT INTO log_status_registro
     (id_registro, status_antigo, status_novo, data_alteracao, usuario_sistema)
  VALUES
     (OLD.id_registro, OLD.status, NEW.status, CURRENT_TIMESTAMP, 'app_streamlit');
END;""",
        language="sql",
    )

    df = db.query_df("SELECT * FROM log_status_registro ORDER BY id_log DESC")
    if df.empty:
        st.info(
            "Nenhuma entrada registrada. Para observar a atuacao do trigger, atualize "
            "o status de um registro na secao de Manutencao de Dados."
        )
    else:
        c1, c2 = st.columns([1, 2])
        c1.metric("Transicoes registradas", len(df))
        c1.metric("Registros distintos afetados", df["id_registro"].nunique())
        with c2:
            df_g = df.groupby("status_novo").size().reset_index(name="qtd")
            fig = px.pie(df_g, names="status_novo", values="qtd", hole=0.45)
            chart(fig)
        st.dataframe(df, width="stretch")
