locals {
  tags = {
    Project = "tech-challenge-fase4"
    Purpose = "monitoramento-multimodal"
    Managed = "terraform"
  }
}

resource "azurerm_resource_group" "this" {
  name     = "rg-${var.prefix}"
  location = var.location
  tags     = local.tags
}

# Azure AI Speech — Speech to Text das gravações de consulta
resource "azurerm_cognitive_account" "speech" {
  name                = "${var.prefix}-speech"
  location            = var.location
  resource_group_name = azurerm_resource_group.this.name
  kind                = "SpeechServices"
  sku_name            = var.speech_sku
  tags                = local.tags
}

# Azure AI Language — Text Analytics (sentimento, entidades, Text Analytics for Health)
resource "azurerm_cognitive_account" "language" {
  name                  = "${var.prefix}-language"
  location              = var.language_location
  resource_group_name   = azurerm_resource_group.this.name
  kind                  = "TextAnalytics"
  sku_name              = var.language_sku
  custom_subdomain_name = "${var.prefix}-language"
  tags                  = local.tags
}

# Rede de proteção do crédito: alerta por e-mail antes de gastar (não bloqueia nada).
data "azurerm_subscription" "current" {}

resource "azurerm_consumption_budget_subscription" "this" {
  count           = var.budget_alert_email == "" ? 0 : 1
  name            = "budget-${var.prefix}"
  subscription_id = data.azurerm_subscription.current.id
  amount          = var.budget_amount
  time_grain      = "Monthly"

  time_period {
    start_date = formatdate("YYYY-MM-01'T'00:00:00Z", plantimestamp())
  }

  notification {
    enabled        = true
    threshold      = 50
    operator       = "GreaterThan"
    contact_emails = [var.budget_alert_email]
  }

  notification {
    enabled        = true
    threshold      = 90
    operator       = "GreaterThan"
    contact_emails = [var.budget_alert_email]
  }

  lifecycle {
    ignore_changes = [time_period] # start_date só vale na criação
  }
}
