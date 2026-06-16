-- ========================================================
-- PRISMA KEY - ESG Compliance Database
-- Schema em SQLite (portavel). Para MySQL ver create_db_mysql.sql
-- ========================================================

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS log_status_registro;
DROP TABLE IF EXISTS Auditoria_Registro;
DROP TABLE IF EXISTS Auditoria;
DROP TABLE IF EXISTS Auditor;
DROP TABLE IF EXISTS Registro;
DROP TABLE IF EXISTS Metrica;
DROP TABLE IF EXISTS Categoria;
DROP TABLE IF EXISTS Unidade;
DROP TABLE IF EXISTS Empresa;

CREATE TABLE Empresa (
    id_empresa       INTEGER PRIMARY KEY AUTOINCREMENT,
    nome_fantasia    VARCHAR(150) NOT NULL,
    cnpj             VARCHAR(20)  NOT NULL UNIQUE,
    cidade           VARCHAR(120),
    industria        VARCHAR(120),
    id_empresa_mae   INTEGER,
    FOREIGN KEY (id_empresa_mae) REFERENCES Empresa(id_empresa)
);

CREATE TABLE Unidade (
    id_unidade     INTEGER NOT NULL,
    id_empresa     INTEGER NOT NULL,
    nome_unidade   VARCHAR(120) NOT NULL,
    localizacao    VARCHAR(150),
    PRIMARY KEY (id_unidade, id_empresa),
    FOREIGN KEY (id_empresa) REFERENCES Empresa(id_empresa)
);

CREATE TABLE Categoria (
    id_categoria INTEGER PRIMARY KEY AUTOINCREMENT,
    descricao    VARCHAR(150) NOT NULL,
    tipo         VARCHAR(20)  NOT NULL CHECK (tipo IN ('AMBIENTAL','SOCIAL','GOVERNANCA'))
);

CREATE TABLE Metrica (
    id_metrica   INTEGER PRIMARY KEY AUTOINCREMENT,
    nome         VARCHAR(150) NOT NULL,
    descricao    TEXT,
    id_categoria INTEGER NOT NULL,
    FOREIGN KEY (id_categoria) REFERENCES Categoria(id_categoria)
);

CREATE TABLE Registro (
    id_registro   INTEGER PRIMARY KEY AUTOINCREMENT,
    data_hora     DATETIME NOT NULL,
    valor_medido  DECIMAL(10,2) NOT NULL,
    status        VARCHAR(20)  NOT NULL CHECK (status IN ('PENDENTE','VALIDADO','REJEITADO')),
    id_unidade    INTEGER NOT NULL,
    id_empresa    INTEGER NOT NULL,
    id_metrica    INTEGER NOT NULL,
    FOREIGN KEY (id_unidade, id_empresa) REFERENCES Unidade(id_unidade, id_empresa),
    FOREIGN KEY (id_metrica) REFERENCES Metrica(id_metrica)
);

CREATE TABLE Auditor (
    cpf                   VARCHAR(14) PRIMARY KEY,
    nome                  VARCHAR(120) NOT NULL,
    registro_profissional VARCHAR(50)
);

CREATE TABLE Auditoria (
    id_auditoria     INTEGER PRIMARY KEY AUTOINCREMENT,
    data_realizacao  DATE NOT NULL,
    parecer_final    TEXT,
    cpf_auditor      VARCHAR(14) NOT NULL,
    FOREIGN KEY (cpf_auditor) REFERENCES Auditor(cpf)
);

CREATE TABLE Auditoria_Registro (
    id_auditoria INTEGER NOT NULL,
    id_registro  INTEGER NOT NULL,
    PRIMARY KEY (id_auditoria, id_registro),
    FOREIGN KEY (id_auditoria) REFERENCES Auditoria(id_auditoria),
    FOREIGN KEY (id_registro)  REFERENCES Registro(id_registro)
);

-- Tabela de log do trigger
CREATE TABLE log_status_registro (
    id_log          INTEGER PRIMARY KEY AUTOINCREMENT,
    id_registro     INTEGER,
    status_antigo   VARCHAR(20),
    status_novo     VARCHAR(20),
    data_alteracao  DATETIME DEFAULT CURRENT_TIMESTAMP,
    usuario_sistema VARCHAR(100)
);

-- ============== INDICES (Etapa 04) ==============
CREATE INDEX idx_registro_empresa   ON Registro(id_empresa, id_unidade);
CREATE INDEX idx_registro_data_hora ON Registro(data_hora);

-- ============== VIEWS (Etapa 04) ==============
DROP VIEW IF EXISTS vw_resumo_metricas_empresas;
CREATE VIEW vw_resumo_metricas_empresas AS
SELECT
    r.id_registro,
    r.data_hora,
    r.valor_medido,
    r.status                 AS status_registro,
    e.nome_fantasia          AS nome_empresa,
    e.cnpj                   AS documento_empresa,
    m.nome                   AS nome_metrica,
    c.descricao              AS categoria_descricao,
    c.tipo                   AS tipo_esg
FROM Registro  r
JOIN Empresa   e ON r.id_empresa  = e.id_empresa
JOIN Metrica   m ON r.id_metrica  = m.id_metrica
JOIN Categoria c ON m.id_categoria = c.id_categoria;

DROP VIEW IF EXISTS vw_detalhes_auditoria_completa;
CREATE VIEW vw_detalhes_auditoria_completa AS
SELECT
    aud.id_auditoria,
    aud.data_realizacao,
    aud.parecer_final,
    a.nome                    AS auditor_responsavel,
    a.registro_profissional,
    r.id_registro,
    r.valor_medido,
    r.status                  AS status_atual_registro
FROM Auditoria aud
LEFT JOIN Auditor a            ON aud.cpf_auditor = a.cpf
LEFT JOIN Auditoria_Registro ar ON aud.id_auditoria = ar.id_auditoria
LEFT JOIN Registro r           ON ar.id_registro = r.id_registro;

-- ============== TRIGGERS (Etapa 05) ==============
DROP TRIGGER IF EXISTS trg_valida_valor_medido;
CREATE TRIGGER trg_valida_valor_medido
BEFORE INSERT ON Registro
FOR EACH ROW
WHEN NEW.valor_medido < 0
BEGIN
    SELECT RAISE(ABORT, 'Erro: valor medido nao pode ser negativo');
END;

DROP TRIGGER IF EXISTS trg_logar_mudanca_status;
CREATE TRIGGER trg_logar_mudanca_status
AFTER UPDATE OF status ON Registro
FOR EACH ROW
WHEN OLD.status <> NEW.status
BEGIN
    INSERT INTO log_status_registro (id_registro, status_antigo, status_novo, data_alteracao, usuario_sistema)
    VALUES (OLD.id_registro, OLD.status, NEW.status, CURRENT_TIMESTAMP, 'app_streamlit');
END;
