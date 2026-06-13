-- 1. APAGAR TABELAS SE JÁ EXISTIREM (Garante que pode rodar o script várias vezes para resetar o teste)
DROP TABLE IF EXISTS planos_acao CASCADE;
DROP TABLE IF EXISTS predicoes_risco CASCADE;
DROP TABLE IF EXISTS auditorias_esg CASCADE;
DROP TABLE IF EXISTS fornecedores CASCADE;

-- 2. CRIAÇÃO DAS 04 TABELAS OBRIGATÓRIAS
CREATE TABLE fornecedores (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry_segment VARCHAR(100) NOT NULL,
    cnpj VARCHAR(18) UNIQUE NOT NULL,
    status_homologacao VARCHAR(50) DEFAULT 'Pendente'
);

CREATE TABLE auditorias_esg (
    id SERIAL PRIMARY KEY,
    fornecedor_id INT,
    environment_score DECIMAL(5,2) NOT NULL,
    social_score DECIMAL(5,2) NOT NULL,
    governance_score DECIMAL(5,2) NOT NULL,
    data_auditoria TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fornecedor_id) REFERENCES fornecedores(id) ON DELETE CASCADE
);

CREATE TABLE predicoes_risco (
    id SERIAL PRIMARY KEY,
    auditoria_id INT,
    total_score_predito DECIMAL(6,2),
    total_level_predito VARCHAR(20),
    run_id_mlflow VARCHAR(100),
    data_calculo TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (auditoria_id) REFERENCES auditorias_esg(id) ON DELETE CASCADE
);

CREATE TABLE planos_acao (
    id SERIAL PRIMARY KEY,
    fornecedor_id INT,
    descricao_correcao TEXT NOT NULL,
    prazo DATE NOT NULL,
    status_plano VARCHAR(50) DEFAULT 'Aberto',
    FOREIGN KEY (fornecedor_id) REFERENCES fornecedores(id) ON DELETE CASCADE
);

-- 3. INJEÇÃO DE DADOS FICTÍCIOS DE TESTE (Para os gráficos da estatística não nascerem vazios)
INSERT INTO fornecedores (name, industry_segment, cnpj, status_homologacao) VALUES
('Alpha Logística', 'Transportes', '11.111.111/0001-11', 'Aprovado'),
('Beta Componentes', 'Manufatura', '22.222.222/0001-22', 'Pendente'),
('Gamma Tech Soluções', 'Tecnologia', '33.333.333/0001-33', 'Bloqueado'),
('Delta Alimentos', 'Alimentos', '44.444.444/0001-44', 'Aprovado');

INSERT INTO auditorias_esg (fornecedor_id, environment_score, social_score, governance_score) VALUES
(1, 450.00, 310.00, 290.00),
(2, 210.00, 180.00, 150.00),
(3, 580.00, 420.00, 390.00),
(4, 390.00, 280.00, 270.00);

-- Simulando o que a IA salvaria após rodar (Estatísticas prontas para os gráficos Streamlit)
INSERT INTO predicoes_risco (auditoria_id, total_score_predito, total_level_predito, run_id_mlflow) VALUES
(1, 1050.00, 'Medium', 'f225e9f6f741452dabf24783620d5ac7'),
(2, 540.00, 'High', '5e11b880be8b43a4acb6a85b432bc3b4'),
(3, 1390.00, 'Low', '7a29531676db41f692e2c4f7f49b149f'),
(4, 940.00, 'Medium', 'c508212b659b4b9784a2d02e58d8e612');

INSERT INTO planos_acao (fornecedor_id, descricao_correcao, prazo, status_plano) VALUES
(2, 'Implementar política de descarte de resíduos eletrônicos.', '2026-08-12', 'Aberto'),
(3, 'Revisar adequação de segurança cibernética e LGPD.', '2026-07-20', 'Em Andamento');