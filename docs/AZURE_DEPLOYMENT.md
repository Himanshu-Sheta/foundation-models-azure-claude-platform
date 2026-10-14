# Microsoft Azure AI Services Deployment Guide

This guide details how to provision, configure, and deploy the **Enterprise Foundation Model Platform** on Microsoft Azure.

---

## 1. Prerequisites
* Azure Subscription with `Cognitive Services Contributor` role.
* Azure CLI (`az`) installed: `az login`.

---

## 2. Resource Provisioning via Azure CLI

### Step 1: Create Resource Group
```bash
az group create --name rg-enterprise-ai --location eastus
```

### Step 2: Deploy Azure OpenAI Service
```bash
az cognitiveservices account create \
  --name oai-enterprise-hub \
  --resource-group rg-enterprise-ai \
  --location eastus \
  --kind OpenAI \
  --sku S0

# Deploy GPT-4o Model
az cognitiveservices account deployment create \
  --name oai-enterprise-hub \
  --resource-group rg-enterprise-ai \
  --deployment-name gpt-4o \
  --model-name gpt-4o \
  --model-version "2024-05-13" \
  --model-format OpenAI \
  --sku-capacity 30 \
  --sku-name "Standard"
```

### Step 3: Deploy Azure AI Search
```bash
az search service create \
  --name search-enterprise-rag \
  --resource-group rg-enterprise-ai \
  --location eastus \
  --sku standard
```

### Step 4: Deploy Azure AI Content Safety
```bash
az cognitiveservices account create \
  --name safety-enterprise-guard \
  --resource-group rg-enterprise-ai \
  --location eastus \
  --kind ContentSafety \
  --sku S0
```

---

## 3. Containerized Deployment on Azure Container Apps

### Build and Push Container to Azure Container Registry (ACR)
```bash
az acr create --resource-group rg-enterprise-ai --name acrenterpriseai --sku Basic --admin-enabled true

az acr build --registry acrenterpriseai --image foundation-models-platform:v1 .
```

### Deploy to Azure Container App
```bash
az containerapp create \
  --name foundation-models-gateway \
  --resource-group rg-enterprise-ai \
  --environment managedEnvironment-enterprise \
  --image acrenterpriseai.azurecr.io/foundation-models-platform:v1 \
  --target-port 8000 \
  --ingress external \
  --env-vars \
    ANTHROPIC_API_KEY=secretref:anthropic-key \
    AZURE_OPENAI_API_KEY=secretref:azure-openai-key \
    AZURE_OPENAI_ENDPOINT=https://oai-enterprise-hub.openai.azure.com/ \
    AZURE_SEARCH_ENDPOINT=https://search-enterprise-rag.search.windows.net \
    AZURE_SEARCH_API_KEY=secretref:azure-search-key
```
