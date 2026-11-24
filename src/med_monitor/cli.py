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


# ----------------------------------------------------------------- áudio

def _cmd_audio_amostras(_: argparse.Namespace) -> int:
    from .audio import amostras

    for a, basal, atual in amostras.gerar_padrao():
        print(f"{a.paciente_id}: basal {basal.name} ({a.basal[1]}) · atual {atual.name} ({a.atual[1]})")
    print(f"em {amostras.DESTINO_PADRAO} (cada .wav com o roteiro .txt ao lado)")
    return 0


def _cmd_audio_analisar(args: argparse.Namespace) -> int:
    import json as _json
    import urllib.request
    from pathlib import Path

    from .audio import consulta
    from .config import AUDIO_RESULTS

    paciente = args.paciente or Path(args.arquivo).stem.split("_")[0]
    rel = consulta.analisar_consulta(args.arquivo, paciente, basal=args.basal, usar_azure=not args.sem_azure)
    print(f"{paciente} · {Path(args.arquivo).name}" + (f" vs basal {Path(args.basal).name}" if args.basal else ""))
    print(f"  serviços: transcrição = {rel.servicos['speech']} · texto = {rel.servicos['language']}")
    for erro in rel.erros:
        print(f"  ! {erro}")
    f = rel.features
    print(f"  fala: {f['taxa_articulacao']:.1f} síl/s · grupos de {f['grupo_fala_medio_s']:.1f} s · "
          f"{f['pausas_por_min']:.0f} pausas/min · {f['inspiracoes_por_min']:.0f} inspirações/min · "
          f"F0 {f['f0_mediana_hz'] or 0:.0f} Hz")
    if rel.transcricao:
        print(f"  transcrição: {rel.transcricao['texto'][:160]}{'…' if len(rel.transcricao['texto']) > 160 else ''}")
    negados = sorted({t["termo"] for t in rel.termos if t["negado"]})
    if negados:
        print(f"  termos negados (ignorados): {', '.join(negados)}")
    if not rel.achados:
        print("  nenhum achado")
    for a in rel.achados:
        ev = a.evidencia
        detalhe = ev.get("criterios_atendidos") or ev.get("termos") or ev.get("sintomas") or ""
        print(f"  ▸ {a.modalidade.value}/{a.tipo} escore {a.escore:.2f} [{a.fonte}] {detalhe}")
    print(f"  relatório: {consulta.salvar(rel, AUDIO_RESULTS)}")

    if args.api:
        for a in rel.achados:
            corpo = {k: v for k, v in a.as_dict().items() if k in ("paciente_id", "modalidade", "tipo", "escore",
                                                                    "fonte", "evidencia")}
            req = urllib.request.Request(args.api.rstrip("/") + "/achados", method="POST",
                                         data=_json.dumps(corpo, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                for alerta in _json.load(resp)["alertas"]:
                    print(f"  → alerta [{alerta['severidade']}] {alerta['mensagem']} → {alerta['destinatario']}")
    return 0


def _cmd_audio_avaliar(_: argparse.Namespace) -> int:
    from .audio import avaliacao

    res = avaliacao.avaliar()
    print(avaliacao.tabela_markdown(res))
    js, md = avaliacao.salvar(res)
    print(f"salvo em {js} e {md}")
    return 0


# ----------------------------------------------------------------- vídeo

def _cmd_video_baixar(args: argparse.Namespace) -> int:
    from .dados import rehab24

    zr = rehab24.RemoteZip()
    for video_id, camera in rehab24.selecao(tuple(args.exercicios)):
        caminho = rehab24.baixar_videos([video_id], camera, zr)[0]
        print(f"{video_id} câmera {camera}: {caminho.stat().st_size / 1e6:.0f} MB", flush=True)
    return 0


def _cmd_video_avaliar(args: argparse.Namespace) -> int:
    from .video import avaliacao

    df = avaliacao.tabela_repeticoes(tuple(args.exercicios), passo=args.passo)
    if df.empty:
        print("nenhum vídeo baixado — rode `med-monitor video baixar` antes")
        return 1
    res = avaliacao.avaliar(df)
    print(avaliacao.tabela_markdown(res))
    for caminho in avaliacao.salvar(df, res):
        print(f"salvo: {caminho}")
    return 0


def _cmd_video_relatorio(args: argparse.Namespace) -> int:
    import json as _json
    import urllib.request

    import pandas as pd

    from .config import VIDEO_RESULTS
    from .video import relatorio

    tabela = VIDEO_RESULTS / "metricas_repeticoes.csv"
    if not tabela.exists():
        print("rode `med-monitor video avaliar` antes (gera as métricas das repetições)")
        return 1
    rel = relatorio.gerar(args.video, pd.read_csv(tabela), paciente_id=args.paciente,
                          com_objetos=not args.sem_objetos)
    print(relatorio.markdown(rel))
    for caminho in relatorio.salvar(rel):
        print(f"salvo: {caminho}")
    for a in rel.achados:
        print(f"  ▸ {a.modalidade.value}/{a.tipo} escore {a.escore:.2f}")
        if args.api:
            corpo = {k: v for k, v in a.as_dict().items()
                     if k in ("paciente_id", "modalidade", "tipo", "escore", "fonte", "evidencia")}
            req = urllib.request.Request(args.api.rstrip("/") + "/achados", method="POST",
                                         data=_json.dumps(corpo, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                for alerta in _json.load(resp)["alertas"]:
                    print(f"    → alerta [{alerta['severidade']}] {alerta['mensagem']} → {alerta['destinatario']}")
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

    audio = sub.add_parser("audio", help="consultas gravadas: alterações vocais + termos críticos (Azure)")
    asub = audio.add_subparsers(dest="acao", required=True)
    am = asub.add_parser("amostras", help="gera consultas sintéticas (espeak-ng) com basal e consulta atual")
    am.set_defaults(func=_cmd_audio_amostras)
    an = asub.add_parser("analisar", help="analisa uma consulta: fala, transcrição, termos e sentimento")
    an.add_argument("arquivo", help="áudio da consulta (.wav)")
    an.add_argument("--basal", help="consulta anterior do mesmo paciente, para comparação")
    an.add_argument("--paciente", help="id do paciente (padrão: prefixo do nome do arquivo)")
    an.add_argument("--sem-azure", action="store_true", help="não chama o Azure mesmo se configurado")
    an.add_argument("--api", help="envia os achados para a fusão, ex.: http://localhost:8000")
    an.set_defaults(func=_cmd_audio_analisar)
    av = asub.add_parser("avaliar", help="detecção de fadiga/dispneia em consultas sintéticas (~1 min)")
    av.set_defaults(func=_cmd_audio_avaliar)

    video = sub.add_parser("video", help="sessões de fisioterapia (REHAB24-6): pose, desvios, áreas críticas")
    vdsub = video.add_subparsers(dest="acao", required=True)
    vb = vdsub.add_parser("baixar", help="baixa só os vídeos dos exercícios avaliados (download seletivo do zip)")
    vb.add_argument("--exercicios", type=int, nargs="+", default=[1, 6])
    vb.set_defaults(func=_cmd_video_baixar)
    va = vdsub.add_parser("avaliar", help="pose + métricas + classificador validado por pessoa")
    va.add_argument("--exercicios", type=int, nargs="+", default=[1, 6])
    va.add_argument("--passo", type=int, default=2, help="processa 1 a cada N quadros")
    va.set_defaults(func=_cmd_video_avaliar)
    vr = vdsub.add_parser("relatorio", help="relatório automático de uma sessão (ex.: PM_001)")
    vr.add_argument("video")
    vr.add_argument("--paciente", help="id do paciente nos achados (padrão: REHAB-Pnn)")
    vr.add_argument("--sem-objetos", action="store_true", help="pula o YOLOv8 de objetos (mais rápido)")
    vr.add_argument("--api", help="envia os achados para a fusão, ex.: http://localhost:8000")
    vr.set_defaults(func=_cmd_video_relatorio)

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
