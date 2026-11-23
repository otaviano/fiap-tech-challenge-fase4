from med_monitor.audit import AuditTrail


def test_trilha_append_only_com_trace(tmp_path):
    trilha = AuditTrail(tmp_path / "aud.jsonl")
    with trilha.trace(usuario="sistema", pergunta="janela sinais vitais", paciente_id="PRT-1") as tid:
        trilha.event(tid, "achado_registrado", tipo="taquicardia")
    eventos = trilha.read(tid)
    assert [e["evento"] for e in eventos] == ["interacao_iniciada", "achado_registrado", "interacao_concluida"]
