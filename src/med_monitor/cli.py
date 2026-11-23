"""CLI `med-monitor`. Os subcomandos de cada modalidade entram à medida que forem implementados."""

from __future__ import annotations

import argparse

from . import __version__
from .config import HOSPITAL, AzureSettings, VideoSettings


def _cmd_info(_: argparse.Namespace) -> int:
    azure = AzureSettings()
    video = VideoSettings()
    ok = lambda b: "configurado" if b else "NÃO configurado (modo offline)"  # noqa: E731
    print(f"med-monitor {__version__} — {HOSPITAL}")
    print(f"  Azure Speech  : {ok(azure.speech_configurado)} [{azure.speech_region}, {azure.speech_language}]")
    print(f"  Azure Language: {ok(azure.language_configurado)}")
    print(f"  YOLO          : {video.yolo_model} | pose: {video.pose_model}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="med-monitor", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="comando", required=True)

    info = sub.add_parser("info", help="mostra a configuração ativa (serviços Azure, modelos)")
    info.set_defaults(func=_cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
