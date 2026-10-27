"""
Gera um dataset sintético de transações digitais com sinais de dispositivo,
localização e comportamento, incluindo três padrões de fraude:

- account_takeover: conta legítima acessada por um fraudador (dispositivo novo,
  localização distante, falhas de login, valor alto, madrugada)
- card_testing: rajadas de transações de baixo valor em poucos minutos
- conta_falsa: contas recém-criadas, frequentemente em emuladores
- camuflada: fraudes sem sinais evidentes (o modelo não deve pegar todas)

Os dados são 100% fictícios e servem apenas para estudo.
Uso: python src/gerar_dados.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_USUARIOS = 5_000
N_TRANSACOES_LEGIT = 58_000
INICIO = pd.Timestamp("2026-07-01")
DIAS = 90

rng = np.random.default_rng(SEED)
CATEGORIAS = ["delivery", "mercado", "eletronicos", "transporte", "farmacia", "games", "moda", "recarga"]
PESOS_CAT = [0.25, 0.18, 0.06, 0.17, 0.09, 0.07, 0.10, 0.08]


def usuarios():
    return pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(N_USUARIOS)],
        "idade_conta_dias": np.where(rng.random(N_USUARIOS) < 0.05, rng.integers(0, 30, N_USUARIOS),
                                     rng.gamma(2.2, 220, N_USUARIOS).astype(int) + 1),
        "ticket_medio": rng.lognormal(4.0, 0.6, N_USUARIOS),
    })


def legitimas(u):
    idx = rng.integers(0, N_USUARIOS, N_TRANSACOES_LEGIT)
    base = u.iloc[idx].reset_index(drop=True)
    hora = rng.choice(24, N_TRANSACOES_LEGIT, p=_perfil_hora(noturno=False))
    ts = INICIO + pd.to_timedelta(rng.integers(0, DIAS, N_TRANSACOES_LEGIT), "D") \
        + pd.to_timedelta(hora, "h") + pd.to_timedelta(rng.integers(0, 60, N_TRANSACOES_LEGIT), "m")
    novo = rng.random(N_TRANSACOES_LEGIT) < 0.07
    return pd.DataFrame({
        "user_id": base.user_id,
        "data_hora": ts,
        "valor": np.round(base.ticket_medio * rng.lognormal(0, 0.5, N_TRANSACOES_LEGIT), 2),
        "categoria": rng.choice(CATEGORIAS, N_TRANSACOES_LEGIT, p=PESOS_CAT),
        "canal": rng.choice(["app", "web"], N_TRANSACOES_LEGIT, p=[0.8, 0.2]),
        "dispositivo_novo": novo.astype(int),
        "idade_dispositivo_dias": np.where(novo, 0, rng.gamma(2, 120, N_TRANSACOES_LEGIT).astype(int) + 1),
        "emulador": (rng.random(N_TRANSACOES_LEGIT) < 0.008).astype(int),
        "distancia_casa_km": np.round(rng.exponential(6, N_TRANSACOES_LEGIT) + (rng.random(N_TRANSACOES_LEGIT) < 0.09) * rng.exponential(500, N_TRANSACOES_LEGIT), 1),
        "falhas_login_24h": rng.choice([0, 1, 2, 3], N_TRANSACOES_LEGIT, p=[0.84, 0.10, 0.04, 0.02]),
        "idade_conta_dias": base.idade_conta_dias,
        "tipo_fraude": "nenhuma",
        "fraude": 0,
    })


def account_takeover(u, n=420):
    base = u.sample(n, random_state=1, replace=True).reset_index(drop=True)
    hora = rng.choice(24, n, p=_perfil_hora(noturno=True))
    ts = INICIO + pd.to_timedelta(rng.integers(0, DIAS, n), "D") + pd.to_timedelta(hora, "h") \
        + pd.to_timedelta(rng.integers(0, 60, n), "m")
    novo = rng.random(n) < 0.55
    return pd.DataFrame({
        "user_id": base.user_id, "data_hora": ts,
        "valor": np.round(base.ticket_medio * rng.lognormal(0.7, 0.9, n), 2),
        "categoria": rng.choice(CATEGORIAS, n, p=[0.05, 0.05, 0.40, 0.05, 0.02, 0.18, 0.15, 0.10]),
        "canal": rng.choice(["app", "web"], n, p=[0.55, 0.45]),
        "dispositivo_novo": novo.astype(int),
        "idade_dispositivo_dias": np.where(novo, 0, rng.integers(1, 30, n)),
        "emulador": (rng.random(n) < 0.06).astype(int),
        "distancia_casa_km": np.round(np.where(rng.random(n) < 0.5, rng.exponential(600, n) + 50, rng.exponential(8, n)), 1),
        "falhas_login_24h": rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.40, 0.18, 0.14, 0.13, 0.09, 0.06]),
        "idade_conta_dias": base.idade_conta_dias,
        "tipo_fraude": "account_takeover", "fraude": 1,
    })


def card_testing(u, n_rajadas=45):
    linhas = []
    for _ in range(n_rajadas):
        user = u.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        inicio = INICIO + pd.Timedelta(days=int(rng.integers(0, DIAS)), hours=int(rng.integers(0, 24)))
        k = int(rng.integers(2, 9))
        dist = float(np.round(rng.exponential(300), 1))
        for j in range(k):
            linhas.append({
                "user_id": user.user_id,
                "data_hora": inicio + pd.Timedelta(minutes=float(j * rng.uniform(0.3, 2.5))),
                "valor": float(np.round(rng.uniform(1, 35), 2)),
                "categoria": rng.choice(["recarga", "games", "delivery"]),
                "canal": rng.choice(["web", "app"], p=[0.7, 0.3]), "dispositivo_novo": int(rng.random() < 0.7), "idade_dispositivo_dias": int(rng.integers(0, 60)),
                "emulador": int(rng.random() < 0.15), "distancia_casa_km": dist,
                "falhas_login_24h": int(rng.integers(0, 3)),
                "idade_conta_dias": int(user.idade_conta_dias),
                "tipo_fraude": "card_testing", "fraude": 1,
            })
    return pd.DataFrame(linhas)


def contas_falsas(n=260):
    ids = [f"F{i:05d}" for i in rng.integers(0, 150, n)]
    hora = rng.choice(24, n, p=_perfil_hora(noturno=True))
    ts = INICIO + pd.to_timedelta(rng.integers(0, DIAS, n), "D") + pd.to_timedelta(hora, "h")
    return pd.DataFrame({
        "user_id": ids, "data_hora": ts,
        "valor": np.round(rng.lognormal(5.2, 0.7, n), 2),
        "categoria": rng.choice(CATEGORIAS, n, p=[0.05, 0.05, 0.45, 0.05, 0.02, 0.15, 0.15, 0.08]),
        "canal": rng.choice(["app", "web"], n, p=[0.6, 0.4]),
        "dispositivo_novo": (rng.random(n) < 0.5).astype(int),
        "idade_dispositivo_dias": rng.integers(0, 30, n),
        "emulador": (rng.random(n) < 0.25).astype(int),
        "distancia_casa_km": np.round(rng.exponential(30, n), 1),
        "falhas_login_24h": rng.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]),
        "idade_conta_dias": rng.integers(0, 45, n),
        "tipo_fraude": "conta_falsa", "fraude": 1,
    })


def fraude_camuflada(u, n=150):
    """Fraudes sem sinais claros (ex.: golpe com o próprio dispositivo da vítima)."""
    d = legitimas(u).sample(n, random_state=7).reset_index(drop=True)
    d["valor"] = np.round(d["valor"] * rng.lognormal(0.5, 0.5, n), 2)
    d["tipo_fraude"] = "camuflada"
    d["fraude"] = 1
    return d


def _perfil_hora(noturno):
    p = np.array([1, .5, .3, .3, .3, .5, 1.5, 3, 4, 4.5, 5, 6, 7, 6, 5, 5, 5, 6, 7, 8, 7, 5, 3, 2], float)
    if noturno:
        p[0:6] *= 6
    return p / p.sum()


def velocidade(df):
    """Nº de transações do mesmo usuário na última 1h e minutos desde a anterior."""
    df = df.sort_values(["user_id", "data_hora"]).reset_index(drop=True)
    df["min_desde_ultima"] = df.groupby("user_id")["data_hora"].diff().dt.total_seconds().div(60).round(1)
    df["tx_ultima_1h"] = (
        df.set_index("data_hora").groupby("user_id")["valor"]
        .rolling("1h").count().values.astype(int)
    )
    return df


def main():
    u = usuarios()
    df = pd.concat([legitimas(u), account_takeover(u), card_testing(u), contas_falsas(), fraude_camuflada(u)], ignore_index=True)
    df = velocidade(df).sort_values("data_hora").reset_index(drop=True)
    df.insert(0, "transacao_id", [f"T{i:06d}" for i in range(len(df))])
    df["hora"] = df.data_hora.dt.hour
    out = Path(__file__).resolve().parents[1] / "data" / "transacoes.csv"
    df.to_csv(out, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"{len(df):,} transações | taxa de fraude {df.fraude.mean():.2%} -> {out}")


if __name__ == "__main__":
    main()
