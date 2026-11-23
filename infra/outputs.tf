# Preenchem o .env:  terraform output -raw speech_key  etc.
output "speech_region" {
  value = azurerm_cognitive_account.speech.location
}

output "speech_key" {
  value     = azurerm_cognitive_account.speech.primary_access_key
  sensitive = true
}

output "language_endpoint" {
  value = azurerm_cognitive_account.language.endpoint
}

output "language_key" {
  value     = azurerm_cognitive_account.language.primary_access_key
  sensitive = true
}
