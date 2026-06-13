import pandas as pd
from sqlalchemy import create_engine

# String de conexão (ajusta com as tuas credenciais do docker-compose)
DB_URL = "mysql+pymysql://user:password@db:3306/prismakey_db"
engine = create_engine(DB_URL)

def executar_query(query):
    """Executa uma consulta SELECT e retorna um DataFrame do Pandas"""
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def executar_comando(sql, params=None):
    """Executa comandos de INSERT, UPDATE, DELETE"""
    with engine.connect() as conn:
        conn.execute(sql, params)