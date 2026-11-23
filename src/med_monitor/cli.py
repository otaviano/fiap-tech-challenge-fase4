"""CLI `med-monitor`. Os subcomandos de cada modalidade entram à medida que forem implementados."""

from __future__ import annotations

import argparse
import json

from . import __version__
from .config import ANOMALIAS_RESULTS, HOSPITAL, AzureSettings, VideoSettings


def _cmd_info(_: argparse.Namespace) -> int:
    azure = AzureSettings()
    video = VideoSettings()
    ok = lambda b: "configurado" if b else "NÃO configurado (modo offline)"  # noqa: E731
    print(f"med-monitor {__version__} — {HOSPITAL}")
    print(f"  Azure Speech  : {ok(azure.speech_configurado)} [{azure.speech_region}, {azure.speech_language}]")
    print(f"  Azure Language: {ok(azure.language_configurado)} [Text Analytics for Health: {azure.text_language}]")
    print(f"  YOLO          : {video.yolo_model} | pose: {video.pose_model}")
    return 0


# ---------------------------------------------------------------- vitais

def _cmd_vitais_baixar(_: argparse.Namespace) -> int:
    from .dados import physionet

    for r in physionet.REGISTROS:
        df = physionet.baixar(r)
        print(f"{r.nome}: {len(df) / 60:.1f} h")
    print(f"cache: {physionet.CACHE_DIR}")
    return 0


def _cmd_vitais_analisar(args: argparse.Namespace) -> int:
    from .anomalias.graficos import plotar
    from .anomalias.sinais_vitais import MonitorSinaisVitais, achados_para_df
    from .audit import get_audit_trail
    from .dados import physionet, sintetico

    eventos = []
    if args.registro:
        serie = physionet.carregar(args.registro)
        nome = args.registro
    else:
        serie = sintetico.gerar_basal(horas=args.horas, semente=args.semente)
        serie, eventos = sintetico.injetar_varios(
            serie, list(sintetico.TIPOS_CLINICOS) + ["artefato", "artefato"], semente=args.semente)
        nome = f"sintetico-s{args.semente}"
    paciente = args.paciente or nome

    trilha = get_audit_trail()
    with trilha.trace("sinais-vitais", f"série {nome} ({len(serie)} min)", paciente_id=paciente) as tid:
        achados = MonitorSinaisVitais(paciente).processar_serie(serie)
        for a in achados:
            trilha.event(tid, "achado", **a.as_dict())

    tabela = achados_para_df(achados)
    if not tabela.empty:
        tabela["evidencia"] = tabela["evidencia"].map(
            lambda e: ", ".join(f"{k}={v}" for k, v in e.items() if k in ("sinal", "valor", "basal", "z",
                                                                          "news2_parcial", "principais_desvios")))
    print(f"{nome}: {len(serie) / 60:.1f} h, {len(achados)} achados")
    for ev in eventos:
        print(f"  evento injetado: {ev.tipo:<24} {ev.inicio} → {ev.fim}")
    if not tabela.empty:
        print(tabela.to_string(index=False, max_colwidth=70))

    ANOMALIAS_RESULTS.mkdir(parents=True, exist_ok=True)
    js = ANOMALIAS_RESULTS / f"achados_{nome}.json"
    js.write_text(json.dumps({"serie": nome, "paciente_id": paciente,
                              "eventos_injetados": [e.as_dict() for e in eventos],
                              "achados": [a.as_dict() for a in achados]},
                             ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"achados: {js}")
    if args.grafico:
        png = plotar(serie, achados, ANOMALIAS_RESULTS / f"sinais_vitais_{nome}.png",
                     f"{HOSPITAL} — sinais vitais · {nome}", eventos)
        print(f"gráfico: {png}")
    return 0


def _cmd_vitais_avaliar(args: argparse.Namespace) -> int:
    from .anomalias import avaliacao

    params_if = {k: v for k, v in (("quantil", args.if_quantil), ("persistencia", args.if_persistencia))
                 if v is not None}
    resumo = avaliacao.avaliar(sementes=tuple(args.sementes), params_if=params_if or None)
    print(avaliacao.tabela_markdown(resumo))
    if args.saida:
        destino = ANOMALIAS_RESULTS / args.saida
    else:
        destino = ANOMALIAS_RESULTS
    js, md = avaliacao.salvar(resumo, destino)
    print(f"salvo em {js} e {md}")
    return 0


# ------------------------------------------------------------- alertas

def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"painel da equipe médica: http://{args.host}:{args.port}")
    uvicorn.run("med_monitor.serving.api:app", host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_monitorar(args: argparse.Namespace) -> int:
    """Plantão no terminal: replay dos leitos imprimindo cada alerta (sem painel)."""
    import asyncio

    from .alertas.despacho import Despachante
    from .serving.simulador import LEITOS_PADRAO, Simulador

    despachante = Despachante()
    sim = Simulador(despachante, velocidade=args.velocidade)
    icone = {"critico": "!!!", "alto": "!! ", "moderado": "!  ", "informativo": "i  "}

    async def rodar():
        fila = despachante.assinar(100_000)
        tarefa = asyncio.create_task(sim.executar(LEITOS_PADRAO))
        limite = None
        while not (tarefa.done() and fila.empty()):
            try:
                tipo, dados = await asyncio.wait_for(fila.get(), timeout=0.5)
            except TimeoutError:
                continue
            if tipo == "vitais" and sim.relogio and limite is None:
                limite = sim.relogio.timestamp() + args.horas * 3600
            if limite and sim.relogio and sim.relogio.timestamp() >= limite:
                sim.parar()
            if tipo == "alerta":
                print(f"{dados['momento'][11:16]} {icone[dados['severidade']]} {dados['contexto'].get('leito')} "
                      f"[{dados['severidade']}] {dados['mensagem']}  → {dados['destinatario']}", flush=True)
        await tarefa

    asyncio.run(rodar())
    print(f"{len(despachante.historico)} alertas · eventos injetados: "
          + ", ".join(f"{e['leito']} {e['tipo']} {e['inicio'][11:16]}" for e in sim.eventos_injetados))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="med-monitor", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="comando", required=True)

    info = sub.add_parser("info", help="mostra a configuração ativa (serviços Azure, modelos)")
    info.set_defaults(func=_cmd_info)

    vitais = sub.add_parser("vitais", help="anomalias em sinais vitais (PhysioNet / sintético)")
    vsub = vitais.add_subparsers(dest="acao", required=True)

    baixar = vsub.add_parser("baixar", help="baixa os registros curados do MIMIC-III (PhysioNet)")
    baixar.set_defaults(func=_cmd_vitais_baixar)

    analisar = vsub.add_parser("analisar", help="roda o monitor numa série e lista os achados")
    fonte = analisar.add_mutually_exclusive_group()
    fonte.add_argument("--registro", help="registro do MIMIC-III (ver `vitais baixar`)")
    fonte.add_argument("--sintetico", action="store_true", help="série sintética com eventos injetados (padrão)")
    analisar.add_argument("--horas", type=float, default=24, help="duração da série sintética")
    analisar.add_argument("--semente", type=int, default=42)
    analisar.add_argument("--paciente", help="id do paciente nos achados (padrão: nome da série)")
    analisar.add_argument("--grafico", action="store_true", help="salva PNG com a série e os achados")
    analisar.set_defaults(func=_cmd_vitais_analisar)

    avaliar = vsub.add_parser("avaliar", help="recall, latência e carga de alarme com anomalias injetadas")
    avaliar.add_argument("--sementes", type=int, nargs="+", default=[1, 2, 3])
    avaliar.add_argument("--if-quantil", type=float, help="quantil do limiar do Isolation Forest")
    avaliar.add_argument("--if-persistencia", type=int, help="persistência do Isolation Forest")
    avaliar.add_argument("--saida", help="subpasta de results/anomalias (para variações)")
    avaliar.set_defaults(func=_cmd_vitais_avaliar)

    serve = sub.add_parser("serve", help="API + painel da equipe médica (http://localhost:8000)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    monitorar = sub.add_parser("monitorar", help="plantão no terminal: replay dos leitos com os alertas")
    monitorar.add_argument("--velocidade", type=float, default=0, help="0 = o mais rápido possível")
    monitorar.add_argument("--horas", type=float, default=6, help="horas de plantão a simular")
    monitorar.set_defaults(func=_cmd_monitorar)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
