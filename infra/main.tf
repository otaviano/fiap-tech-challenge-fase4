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
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  kind                = "SpeechServices"
  sku_name            = var.sku
  tags                = local.tags
}

# Azure AI Language — Text Analytics (sentimento, entidades, Text Analytics for Health)
resource "azurerm_cognitive_account" "language" {
  name                  = "${var.prefix}-language"
  location              = azurerm_resource_group.this.location
  resource_group_name   = azurerm_resource_group.this.name
  kind                  = "TextAnalytics"
  sku_name              = var.sku
  custom_subdomain_name = "${var.prefix}-language"
  tags                  = local.tags
}
