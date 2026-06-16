-- ========================================================
-- POPULACAO INICIAL - elementos fixos do dominio
-- Empresas, Unidades e Registros sao carregados do CSV
-- (datakaggle.csv) pelo loader Python em db.py.
-- ========================================================

INSERT INTO Categoria (id_categoria, descricao, tipo) VALUES
(1, 'Desempenho Ambiental',  'AMBIENTAL'),
(2, 'Desempenho Social',     'SOCIAL'),
(3, 'Governanca Corporativa','GOVERNANCA');

INSERT INTO Metrica (id_metrica, nome, descricao, id_categoria) VALUES
(1, 'Environment Score',  'Score ambiental ESG (0-1000)',  1),
(2, 'Social Score',       'Score social ESG (0-1000)',     2),
(3, 'Governance Score',   'Score de governanca (0-1000)',  3),
(4, 'Total ESG Score',    'Score total agregado (0-3000)', 3);

INSERT INTO Auditor (cpf, nome, registro_profissional) VALUES
('111.222.333-44', 'Carla Mendes',   'CRC-SP 12345'),
('222.333.444-55', 'Joao Lima',      'CRC-RJ 23456'),
('333.444.555-66', 'Beatriz Souza',  'CRC-MG 34567'),
('444.555.666-77', 'Eduardo Reis',   'CRC-PR 45678'),
('555.666.777-88', 'Helena Castro',  'CRC-SC 56789');
