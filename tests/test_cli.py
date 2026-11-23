from med_monitor.cli import main


def test_info_sem_credenciais_indica_modo_offline(monkeypatch, capsys):
    for var in ("AZURE_SPEECH_KEY", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    assert main(["info"]) == 0
    saida = capsys.readouterr().out
    assert "modo offline" in saida


def test_vitais_analisar_sintetico_grava_achados(monkeypatch, tmp_path, capsys):
    import json

    from med_monitor import audit, cli

    monkeypatch.setattr(cli, "ANOMALIAS_RESULTS", tmp_path)
    audit.get_audit_trail(tmp_path / "aud.jsonl")
    assert main(["vitais", "analisar", "--sintetico", "--horas", "24", "--semente", "5"]) == 0
    dados = json.loads((tmp_path / "achados_sintetico-s5.json").read_text())
    assert dados["achados"] and len(dados["eventos_injetados"]) == 8
    eventos = [e["evento"] for e in audit.get_audit_trail().read(limite=1000)]
    assert eventos[0] == "interacao_iniciada" and eventos[-1] == "interacao_concluida"
    assert "achado" in eventos
