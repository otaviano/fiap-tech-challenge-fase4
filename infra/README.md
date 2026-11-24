# Infraestrutura Azure (Terraform)

Provisiona os serviços gerenciados que o enunciado pede:

| Recurso | Kind | Região | Uso no projeto |
|---|---|---|---|
| `medmonitor-speech` | `SpeechServices` | `brazilsouth` | Azure Speech to Text — transcrição das consultas (pt-BR) |
| `medmonitor-language` | `TextAnalytics` | `brazilsouth` | Azure Text Analytics — sentimento, termos críticos, Text Analytics for Health (`language="pt"`) |
| `budget-medmonitor` | orçamento | assinatura | alerta por e-mail em 50% e 90% do teto (opcional) |

Camada **F0 (gratuita)** por padrão: suficiente para desenvolvimento e para a
demonstração, sem cobrança excedente (no F0 a chamada é recusada ao bater o
limite). Só é permitido um recurso F0 de cada tipo por assinatura. Custo
estimado e riscos estão na decisão D7 de [`../docs/plano.md`](../docs/plano.md).

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

> **Assinatura Azure for Students:** uma política só permite 5 regiões
> (`brazilsouth`, `westus3`, `francecentral`, `spaincentral`,
> `southafricanorth`). Fora delas, o `apply` falha com `RequestDisallowedByAzure`.
> Consulte as regiões liberadas com
> `az policy assignment list --query "[].parameters.listOfAllowedLocations.value"`.

> Chaves, `terraform.tfvars` e o state **não são versionados** (ver `.gitignore`).
> Ao fim do projeto: `terraform destroy`.

> **Text Analytics for Health:** aceita português pela tag `pt`. A documentação
> não confirma se ele entra na cota F0. Se a primeira chamada for recusada, use
> `language_sku = "S"` no `terraform.tfvars` e rode `terraform apply` de novo.
