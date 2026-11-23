"""Clientes dos serviços gerenciados do Azure.

Isola o SDK do resto do código: os analisadores dependem de uma interface
pequena (`transcrever`, `analisar_texto`), não do SDK. Sem credenciais, os
clientes **não fingem**: `disponivel` é falso e o pipeline registra no
relatório que o serviço não foi chamado (decisão D4 do plano).

Os recursos são provisionados por Terraform em `infra/`.
"""
