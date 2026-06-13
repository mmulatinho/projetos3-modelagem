-- TABELA 1: Cadastro dos Fornecedores
CREATE TABLE fornecedores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry_segment VARCHAR(100) NOT NULL,
    cnpj VARCHAR(18) UNIQUE NOT NULL,
    status_homologacao VARCHAR(50) DEFAULT 'Pendente'
);

-- TABELA 2: Histórico de Auditorias (Guarda os inputs para a IA)
CREATE TABLE auditorias_esg (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fornecedor_id INT,
    environment_score DECIMAL(5,2) NOT NULL,
    social_score DECIMAL(5,2) NOT NULL,
    governance_score DECIMAL(5,2) NOT NULL,
    data_auditoria TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fornecedor_id) REFERENCES fornecedores(id) ON DELETE CASCADE
);

-- TABELA 3: Predições de Risco (Guarda os outputs gerados pelo KNN e Random Forest)
CREATE TABLE predicoes_risco (
    id INT AUTO_INCREMENT PRIMARY KEY,
    auditoria_id INT,
    total_score_predito DECIMAL(6,2),
    total_level_predito VARCHAR(20),
    run_id_mlflow VARCHAR(100),
    data_calculo TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (auditoria_id) REFERENCES auditorias_esg(id) ON DELETE CASCADE
);

-- TABELA 4: Planos de Ação (Garante o CRUD completo exigido pelo edital)
CREATE TABLE planos_acao (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fornecedor_id INT,
    descricao_correcao TEXT NOT NULL,
    prazo DATE NOT NULL,
    status_plano VARCHAR(50) DEFAULT 'Aberto',
    FOREIGN KEY (fornecedor_id) REFERENCES fornecedores(id) ON DELETE CASCADE
);