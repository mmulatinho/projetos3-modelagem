-- ========================================================
-- CONSULTAS UTILIZADAS NA INTERFACE
-- Etapas 03 e 04
-- ========================================================

-- ---------- ETAPA 03: 4 consultas obrigatorias ----------

-- Q1 (Etapa 03 - INNER JOIN multi-tabela): registros completos com empresa, metrica e categoria
SELECT r.id_registro, r.data_hora, r.valor_medido, r.status,
       e.nome_fantasia, m.nome AS metrica, c.tipo AS pilar_esg
FROM Registro  r
JOIN Empresa   e ON e.id_empresa = r.id_empresa
JOIN Metrica   m ON m.id_metrica = r.id_metrica
JOIN Categoria c ON c.id_categoria = m.id_categoria
ORDER BY r.data_hora DESC;

-- Q2 (Etapa 03 - agregacao): media de valor medido por pilar ESG
SELECT c.tipo AS pilar, COUNT(*) AS total, AVG(r.valor_medido) AS media
FROM Registro  r
JOIN Metrica   m ON m.id_metrica   = r.id_metrica
JOIN Categoria c ON c.id_categoria = m.id_categoria
GROUP BY c.tipo;

-- Q3 (Etapa 03 - filtro por status): registros pendentes por empresa
SELECT e.nome_fantasia, COUNT(*) AS pendentes
FROM Registro r
JOIN Empresa  e ON e.id_empresa = r.id_empresa
WHERE r.status = 'PENDENTE'
GROUP BY e.nome_fantasia
ORDER BY pendentes DESC;

-- Q4 (Etapa 03 - auditores ativos): auditorias por auditor
SELECT a.nome, COUNT(aud.id_auditoria) AS qtd_auditorias
FROM Auditor a
LEFT JOIN Auditoria aud ON aud.cpf_auditor = a.cpf
GROUP BY a.nome
ORDER BY qtd_auditorias DESC;

-- ---------- ETAPA 04: 4 consultas avancadas ----------

-- Q5 (Anti-join): empresas SEM unidades cadastradas
SELECT e.id_empresa, e.nome_fantasia, e.cnpj
FROM Empresa e
LEFT JOIN Unidade u ON e.id_empresa = u.id_empresa
WHERE u.id_unidade IS NULL;

-- Q6 (Full outer join via UNION): auditores x auditorias (incluindo auditores sem registros)
SELECT a.nome AS auditor_nome, a.cpf, aud.id_auditoria, aud.data_realizacao
FROM Auditor a
LEFT JOIN Auditoria aud ON a.cpf = aud.cpf_auditor
UNION
SELECT a.nome AS auditor_nome, a.cpf, aud.id_auditoria, aud.data_realizacao
FROM Auditoria aud
LEFT JOIN Auditor a ON aud.cpf_auditor = a.cpf;

-- Q7 (Subconsulta com IN): metricas que foram auditadas em registros VALIDADOS
SELECT m.id_metrica, m.nome AS nome_metrica, m.descricao
FROM Metrica m
WHERE m.id_metrica IN (
    SELECT r.id_metrica
    FROM Registro r
    JOIN Auditoria_Registro ar ON r.id_registro = ar.id_registro
    WHERE r.status = 'VALIDADO'
);

-- Q8 (Subconsulta com HAVING): empresas com media de valor acima da media geral
SELECT e.id_empresa, e.nome_fantasia, AVG(r.valor_medido) AS media_medida_empresa
FROM Empresa e
JOIN Registro r ON e.id_empresa = r.id_empresa
GROUP BY e.id_empresa, e.nome_fantasia
HAVING AVG(r.valor_medido) > (SELECT AVG(valor_medido) FROM Registro);
