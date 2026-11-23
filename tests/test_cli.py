from med_monitor.cli import main


def test_info_sem_credenciais_indica_modo_offline(monkeypatch, capsys):
    for var in ("AZURE_SPEECH_KEY", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    assert main(["info"]) == 0
    saida = capsys.readouterr().out
    assert "modo offline" in saida
