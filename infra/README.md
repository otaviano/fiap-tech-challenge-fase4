# Infraestrutura Azure (Terraform)

Provisiona os serviços gerenciados que o enunciado pede:

| Recurso | Kind | Uso no projeto |
|---|---|---|
| `medmonitor-speech` | `SpeechServices` | Azure Speech to Text — transcrição das consultas (pt-BR) |
| `medmonitor-language` | `TextAnalytics` | Azure Text Analytics — sentimento, termos críticos, Text Analytics for Health |

Camada **F0 (gratuita)** por padrão: suficiente para desenvolvimento e para a
demonstração. Só é permitido um recurso F0 de cada tipo por assinatura.

## Uso

```bash
az login
cp terraform.tfvars.example terraform.tfvars   # preencha o subscription_id
terraform init
terraform apply
```

Depois, preencha o `.env` na raiz do projeto:

```bash
echo "AZURE_SPEECH_KEY=$(terraform output -raw speech_key)" >> ../.env
echo "AZURE_SPEECH_REGION=$(terraform output -raw speech_region)" >> ../.env
echo "AZURE_LANGUAGE_ENDPOINT=$(terraform output -raw language_endpoint)" >> ../.env
echo "AZURE_LANGUAGE_KEY=$(terraform output -raw language_key)" >> ../.env
```

> Chaves, `terraform.tfvars` e o state **não são versionados** (ver `.gitignore`).
> Ao fim do projeto: `terraform destroy`.

> **Text Analytics for Health:** confirmar disponibilidade em `brazilsouth` e
> suporte a pt-BR antes de depender dele; se não houver, a análise de saúde
> roda em `eastus` (basta mudar `location`) e o sentimento continua em pt-BR.
