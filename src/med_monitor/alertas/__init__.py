"""Despacho de alertas para a equipe médica.

Recebe `Alerta`s da fusão, registra na trilha de auditoria e publica nos
canais: barramento em memória → painel em tempo real (SSE na API) e log. Um
canal externo (e-mail/Teams/webhook) fica atrás de uma interface, desligado por
padrão.
"""
