"""Sinais vitais reais do PhysioNet — MIMIC-III Waveform Database Matched Subset.

Usamos os registros *numéricos* (sufixo `n`): uma amostra por minuto de FC,
pressão arterial invasiva, frequência respiratória e SpO2, extraída do monitor
de beira de leito de pacientes de UTI. É acesso aberto (licença ODbL v1.0) e traz o que o
enunciado pede — "batimentos, pressão arterial, oxigenação" — com horas de
duração e com os artefatos de verdade de um monitor (sensor solto = zero).

Os registros são baixados uma vez e guardados em `data/external/physionet/`
como CSV no formato canônico (`fc`, `pas`, `pad`, `pam`, `fr`, `spo2`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR

BASE_PN = "mimic3wdb-matched/1.0"
CACHE_DIR = EXTERNAL_DIR / "physionet"

# Nome no monitor → nome canônico. A PA não invasiva (NBP) é medida a cada
# 15–60 min; entra só quando não há linha arterial.
MAPA_SINAIS = {
    "HR": "fc",
    "ABPSys": "pas",
    "ABPDias": "pad",
    "ABPMean": "pam",
    "RESP": "fr",
    "SpO2": "spo2",
}
MAPA_NBP = {"NBPSys": "pas", "NBPDias": "pad", "NBPMean": "pam"}
SINAIS = ("fc", "pas", "pad", "pam", "fr", "spo2")


@dataclass(frozen=True)
class Registro:
    paciente: str       # diretório do paciente no PhysioNet, ex.: "p00/p000020"
    nome: str           # registro numérico, ex.: "p000020-2183-04-28-17-47n"

    @property
    def id(self) -> str:
        return self.nome

    @property
    def pn_dir(self) -> str:
        return f"{BASE_PN}/{self.paciente}"


# Curadoria: um registro por paciente, todos com FC, PA invasiva, FR e SpO2 e
# pelo menos 8 h de duração (varredura dos primeiros pacientes do subset).
REGISTROS = (
    Registro("p00/p000020", "p000020-2183-04-28-17-47n"),   # 21,9 h
    Registro("p00/p000079", "p000079-2175-09-26-12-28n"),   # 38,2 h
    Registro("p00/p000123", "p000123-2161-10-29-03-01n"),   # 34,0 h
    Registro("p00/p000124", "p000124-2160-07-05-21-51n"),   # 16,1 h
    Registro("p00/p000138", "p000138-2131-10-31-13-39n"),   # 20,6 h
    Registro("p00/p000177", "p000177-2125-11-29-12-35n"),   # 26,4 h
    Registro("p00/p000208", "p000208-2188-05-18-00-06n"),   # 41,6 h
    Registro("p00/p000214", "p000214-2188-10-24-16-23n"),   # 71,8 h
    Registro("p00/p000217", "p000217-2126-09-30-12-28n"),   # 23,6 h
    Registro("p00/p000263", "p000263-2160-01-08-23-52n"),   # 42,2 h
)


def _para_canonico(df_monitor: pd.DataFrame) -> pd.DataFrame:
    saida = pd.DataFrame(index=df_monitor.index)
    for origem, destino in MAPA_SINAIS.items():
        if origem in df_monitor:
            saida[destino] = df_monitor[origem]
    for origem, destino in MAPA_NBP.items():
        if origem in df_monitor:
            nbp = df_monitor[origem]
            saida[destino] = saida[destino].fillna(nbp) if destino in saida else nbp
    for sinal in SINAIS:
        if sinal not in saida:
            saida[sinal] = np.nan
    return saida[list(SINAIS)].astype(float)


def baixar(registro: Registro, forcar: bool = False) -> pd.DataFrame:
    """Baixa (ou lê do cache) um registro e devolve a série canônica minuto a minuto."""
    caminho = CACHE_DIR / f"{registro.nome}.csv"
    if caminho.exists() and not forcar:
        return carregar_csv(caminho)

    import wfdb  # dependência do extra `anomalias`

    rec = wfdb.rdrecord(registro.nome, pn_dir=registro.pn_dir)
    inicio = pd.Timestamp.combine(rec.base_date, rec.base_time) if rec.base_date else pd.Timestamp("2100-01-01")
    passo = pd.Timedelta(seconds=round(1 / rec.fs))
    # o base_time do MIMIC tem fração de segundo; arredondar deixa os achados legíveis
    indice = pd.date_range(inicio.floor("s"), periods=rec.sig_len, freq=passo, name="momento")
    bruto = pd.DataFrame(rec.p_signal, columns=rec.sig_name, index=indice)
    serie = _para_canonico(bruto)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    serie.to_csv(caminho)
    return serie


def carregar_csv(caminho) -> pd.DataFrame:
    df = pd.read_csv(caminho, index_col="momento", parse_dates=["momento"])
    df.index = df.index.floor("s")
    return df


def carregar(id_registro: str) -> pd.DataFrame:
    """Carrega um registro curado pelo nome (baixando se preciso)."""
    for r in REGISTROS:
        if r.nome == id_registro:
            return baixar(r)
    raise KeyError(f"registro fora da curadoria: {id_registro}")
