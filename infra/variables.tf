variable "subscription_id" {
  description = "ID da assinatura Azure (az account show --query id -o tsv)"
  type        = string
}

variable "prefix" {
  description = "Prefixo dos nomes dos recursos"
  type        = string
  default     = "medmonitor"
}

variable "location" {
  description = "Região do resource group e do Speech"
  type        = string
  default     = "brazilsouth"
}

variable "language_location" {
  description = "Região do Language. eastus porque a docs não confirma Text Analytics for Health no Brazil South (plano D7)"
  type        = string
  default     = "eastus"
}

variable "speech_sku" {
  description = "F0 = gratuito (5 h/mês de STT em tempo real); S0 = pago"
  type        = string
  default     = "F0"
}

variable "language_sku" {
  description = "F0 = gratuito (5.000 registros/mês); S = pago. Trocar para S se o F0 recusar Text Analytics for Health"
  type        = string
  default     = "F0"
}

variable "budget_amount" {
  description = "Teto mensal do orçamento de alerta, em USD"
  type        = number
  default     = 20
}

variable "budget_alert_email" {
  description = "E-mail que recebe os alertas de orçamento (50% e 90%). Vazio = sem orçamento"
  type        = string
  default     = ""
}
