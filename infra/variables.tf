variable "subscription_id" {
  description = "ID da assinatura Azure (az account show --query id -o tsv)"
  type        = string
}

variable "location" {
  description = "Região dos recursos. brazilsouth tem Speech e Language."
  type        = string
  default     = "brazilsouth"
}

variable "prefix" {
  description = "Prefixo dos nomes dos recursos"
  type        = string
  default     = "medmonitor"
}

variable "sku" {
  description = "F0 = camada gratuita (1 recurso F0 por tipo por assinatura); S0 = pago"
  type        = string
  default     = "F0"
}
