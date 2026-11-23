"""Fusão multimodal (late fusion).

Consome `Achado`s de qualquer modalidade, agrupa por paciente numa janela de
tempo e decide alertas por regras explícitas. Achados concordantes de
modalidades diferentes (ex.: SpO2 caindo + fala ofegante + menor amplitude de
movimento) elevam a severidade — `Severidade.elevar()`.
"""
