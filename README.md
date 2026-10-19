# 📄 DocIntelligence AI — Assistente Inteligente de Documentos com RAG

Plataforma de Inteligência Documental baseada em Arquitetura RAG (Retrieval-Augmented Generation) para análise, extração e consulta interativa de PDFs corporativos em tempo real.

---

## 🎯 Sobre o Projeto

O **DocIntelligence AI** foi desenvolvido para resolver o desafio de análise e busca de informações em documentos corporativos extensos (como contratos, relatórios financeiros, manuais e regulamentos). 

Utilizando a técnica de **RAG (Geração Aumentada por Recuperação)**, a aplicação fatia e vetoriza o conteúdo do documento localmente, consultando apenas os trechos estritamente relevantes para responder às perguntas do usuário. Isso reduz custos de contexto, evita alucinações e fornece **rastreabilidade total das respostas**.

---

## ✨ Diferenciais e Funcionalidades

- 🔍 **RAG com Rastreabilidade de Fontes:** Exibição expansível dos trechos exatos do PDF que fundamentaram a resposta da IA.
- ⚡ **Embeddings Locais:** Uso do modelo `all-MiniLM-L6-v2` (HuggingFace) para conversão vetorial eficiente sem dependência externa de API para vetores.
- 🧠 **LLM de Alta Performance:** Integração com o Google Gemini (`gemini-3.8-flash` / `gemini-2.5-flash`) via LangChain.
- 💬 **Interface de Chat Contínuo:** Manutenção de histórico de conversa interativo utilizando `st.chat_message` e `st.session_state`.
- 🛡️ **Segurança e Isolamento de Segredos:** Suporte a carregamento automático de chaves de API via `.env` local e integração nativa com *Streamlit Secrets* na nuvem.

---

## 🛠️ Arquitetura do Sistema

1. **Upload do PDF:** Leitura do arquivo via `PyPDF`.
2. **Chunking:** Fatiamento em blocos com `RecursiveCharacterTextSplitter`.
3. **Embeddings & Vector Store:** Geração de vetores locais com HuggingFace e armazenamento temporário no `ChromaDB`.
4. **Retrieval & LLM:** Busca por similaridade (`k=3`) e montagem de prompt contextualizado para a API do Google Gemini.

---

## 🚀 Como Executar o Projeto Localmente

### Pré-requisitos
- Python 3.10 ou superior
- Uma chave de API gratuita do [Google AI Studio](https://aistudio.google.com/)

### Passo a Passo

1. **Clonar o repositório:**
   ```bash
   git clone [https://github.com/clovisfilh0/doc-intelligence-ai.git](https://github.com/clovisfilh0/doc-intelligence-ai.git)
   cd doc-intelligence-ai