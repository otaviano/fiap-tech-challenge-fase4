"""Clientes dos serviços gerenciados do Azure.

Isola o SDK do resto do código: os analisadores dependem de uma interface
pequena (`transcrever`, `analisar_texto`), não do SDK. Sem credenciais, os
clientes caem em modo offline declarado — o pipeline roda, mas o relatório diz
explicitamente que o Azure não foi usado.

Os recursos são provisionados por Terraform em `infra/`.
"""
