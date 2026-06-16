"""Camada de acesso ao banco SQLite para a aplicacao Prisma Key."""
from __future__ import annotations

import hashlib
import random
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "prismakey.db"
SQL_DIR = BASE_DIR / "sql"
CSV_PATH = SQL_DIR / "datakaggle.csv"


def _run_script(conn: sqlite3.Connection, path: Path) -> None:
    conn.executescript(path.read_text(encoding="utf-8"))


def init_db(force: bool = False) -> None:
    if DB_PATH.exists() and not force:
        return
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        _run_script(conn, SQL_DIR / "create_db.sql")
        _run_script(conn, SQL_DIR / "insert_data.sql")
        conn.commit()
    _load_csv()


def _fake_cnpj(seed: str) -> str:
    h = hashlib.sha1(seed.encode()).hexdigest()
    digits = "".join(c for c in h if c.isdigit()).ljust(14, "0")[:14]
    return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:14]}"


def _grade_to_status(grade: str) -> str:
    g = (grade or "").strip().upper()
    if g in ("A", "AA", "AAA", "BBB"):
        return "VALIDADO"
    if g == "BB":
        return "PENDENTE"
    return "REJEITADO"


_METRICA_MAP = {
    "environment_score": (1, "environment_grade"),
    "social_score":      (2, "social_grade"),
    "governance_score":  (3, "governance_grade"),
    "total_score":       (4, "total_grade"),
}


def _load_csv() -> None:
    """Carrega Empresa/Unidade/Registro/Auditoria a partir do datakaggle.csv."""
    if not CSV_PATH.exists():
        return

    df = (
        pd.read_csv(CSV_PATH)
          .drop_duplicates(subset="ticker")
          .reset_index(drop=True)
          .assign(
              id_empresa = lambda d: d.index + 1,
              data       = lambda d: pd.to_datetime(
                  d["last_processing_date"], format="%d-%m-%Y", errors="coerce"
              ).fillna(pd.Timestamp("2022-04-19")),
          )
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        df.rename(columns={"name": "nome_fantasia", "exchange": "cidade",
                            "industry": "industria"}) \
          .assign(cnpj=df["ticker"].map(_fake_cnpj)) \
          [["id_empresa", "nome_fantasia", "cnpj", "cidade", "industria"]] \
          .to_sql("Empresa", conn, if_exists="append", index=False)

        df.assign(id_unidade=1,
                   nome_unidade="Sede " + df["ticker"].str.upper(),
                   localizacao=df["exchange"]) \
          [["id_unidade", "id_empresa", "nome_unidade", "localizacao"]] \
          .to_sql("Unidade", conn, if_exists="append", index=False)

        scores = df.melt(id_vars=["id_empresa", "data"],
                         value_vars=list(_METRICA_MAP),
                         var_name="metrica", value_name="valor_medido")
        grades = df.melt(id_vars="id_empresa",
                         value_vars=[g for _, g in _METRICA_MAP.values()],
                         var_name="metrica", value_name="grade")
        grades["metrica"] = grades["metrica"].str.replace("_grade", "_score")

        registros = (
            scores.merge(grades, on=["id_empresa", "metrica"])
                  .assign(
                      id_metrica = lambda d: d["metrica"].map(lambda m: _METRICA_MAP[m][0]),
                      status     = lambda d: d["grade"].map(_grade_to_status),
                      id_unidade = 1,
                      data_hora  = lambda d: d["data"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                  )
        )[["data_hora", "valor_medido", "status", "id_unidade", "id_empresa", "id_metrica"]]
        registros.to_sql("Registro", conn, if_exists="append", index=False)

        # Auditorias: 1 a cada 6 empresas
        auditores = pd.read_sql("SELECT cpf FROM Auditor", conn)["cpf"].tolist()
        rng = random.Random(42)
        sample = df.iloc[::6].reset_index(drop=True).assign(
            id_auditoria = lambda d: d.index + 1,
            data_realizacao = lambda d: (d["data"] + pd.Timedelta(days=30)).dt.strftime("%Y-%m-%d"),
            parecer_final   = lambda d: "Auditoria ESG " + d["name"],
            cpf_auditor     = lambda d: [rng.choice(auditores) for _ in range(len(d))],
        )
        sample[["id_auditoria", "data_realizacao", "parecer_final", "cpf_auditor"]] \
            .to_sql("Auditoria", conn, if_exists="append", index=False)

        reg_ids = pd.read_sql(
            "SELECT id_registro, id_empresa, id_metrica FROM Registro", conn)
        sample[["id_auditoria", "id_empresa"]] \
            .merge(reg_ids[reg_ids["id_metrica"].isin([1, 2])], on="id_empresa") \
            [["id_auditoria", "id_registro"]] \
            .to_sql("Auditoria_Registro", conn, if_exists="append", index=False)
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def query_df(sql: str, params: tuple | dict | None = None) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params or ())


def execute(sql: str, params: tuple | dict | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(sql, params or ())
        return cur.lastrowid or cur.rowcount


# ============================================================
# FUNCOES (equivalentes SQL no Python pois SQLite nao tem
# CREATE FUNCTION com IF/ELSE igual MySQL)
# ============================================================
def classificar_impacto_registro(valor: float) -> str:
    if valor <= 100.00:
        return "IMPACTO BAIXO / CONFORME"
    if valor <= 500.00:
        return "IMPACTO MODERADO / ATENCAO"
    return "IMPACTO CRITICO / ALERTA"


def obter_nome_categoria_da_metrica(id_metrica: int) -> str:
    df = query_df(
        """SELECT c.descricao FROM Metrica m
           JOIN Categoria c ON m.id_categoria = c.id_categoria
           WHERE m.id_metrica = ?""",
        (id_metrica,),
    )
    if df.empty:
        return "Categoria Nao Encontrada"
    return df.iloc[0, 0]


# ============================================================
# PROCEDIMENTOS
# ============================================================
def proc_atualizar_status_coletas_unidade(id_unidade: int, id_empresa: int, novo_status: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """UPDATE Registro SET status = ?
               WHERE id_unidade = ? AND id_empresa = ? AND status = 'PENDENTE'""",
            (novo_status, id_unidade, id_empresa),
        )
        return cur.rowcount


def proc_reavaliar_registros_rejeitados() -> int:
    """Procedimento com CURSOR: percorre rejeitados e zera valores absurdos."""
    afetados = 0
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id_registro, valor_medido FROM Registro WHERE status = 'REJEITADO'"
        )
        for row in cur.fetchall():
            if row["valor_medido"] > 5000.00:
                conn.execute(
                    "UPDATE Registro SET status='PENDENTE', valor_medido=0.00 WHERE id_registro=?",
                    (row["id_registro"],),
                )
                afetados += 1
    return afetados
