-- Análises exploratórias de fraude em SQL (DuckDB; sintaxe compatível com BigQuery
-- com pequenos ajustes). Tabela: transacoes (data/transacoes.csv)

-- 1. Visão geral: volume, taxa de fraude e valor exposto
SELECT
  COUNT(*)                                        AS transacoes,
  SUM(fraude)                                     AS fraudes,
  ROUND(AVG(fraude) * 100, 2)                     AS taxa_fraude_pct,
  ROUND(SUM(CASE WHEN fraude = 1 THEN valor END), 2) AS valor_fraudado
FROM transacoes;

-- 2. Taxa de fraude por sinal de dispositivo
SELECT
  dispositivo_novo,
  emulador,
  COUNT(*)                    AS transacoes,
  ROUND(AVG(fraude) * 100, 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY 1, 2
ORDER BY taxa_fraude_pct DESC;

-- 3. Taxa de fraude por faixa de distância do local habitual
SELECT
  CASE
    WHEN distancia_casa_km < 10  THEN '1. < 10 km'
    WHEN distancia_casa_km < 50  THEN '2. 10-50 km'
    WHEN distancia_casa_km < 300 THEN '3. 50-300 km'
    ELSE '4. 300+ km'
  END                         AS faixa_distancia,
  COUNT(*)                    AS transacoes,
  ROUND(AVG(fraude) * 100, 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY 1
ORDER BY 1;

-- 4. Velocidade: rajadas de transações (indício de card testing)
SELECT
  LEAST(tx_ultima_1h, 6)      AS tx_ultima_1h_cap,
  COUNT(*)                    AS transacoes,
  ROUND(AVG(fraude) * 100, 2) AS taxa_fraude_pct,
  ROUND(AVG(valor), 2)        AS ticket_medio
FROM transacoes
GROUP BY 1
ORDER BY 1;

-- 5. Usuários com comportamento anômalo: muitas transações pequenas em sequência
SELECT
  user_id,
  COUNT(*)                         AS tx_ate_15_reais,
  MIN(data_hora)                   AS primeira,
  MAX(data_hora)                   AS ultima,
  MAX(tx_ultima_1h)                AS pico_tx_1h
FROM transacoes
WHERE valor <= 15
GROUP BY user_id
HAVING MAX(tx_ultima_1h) >= 4
ORDER BY pico_tx_1h DESC
LIMIT 20;

-- 6. Fraude por hora do dia
SELECT hora, COUNT(*) AS transacoes, ROUND(AVG(fraude) * 100, 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY hora
ORDER BY hora;

-- 7. Combinação de sinais de account takeover
SELECT
  (dispositivo_novo = 1)       AS disp_novo,
  (distancia_casa_km >= 300)   AS longe_de_casa,
  (falhas_login_24h >= 3)      AS falhas_login,
  COUNT(*)                     AS transacoes,
  ROUND(AVG(fraude) * 100, 2)  AS taxa_fraude_pct
FROM transacoes
GROUP BY 1, 2, 3
ORDER BY taxa_fraude_pct DESC;
