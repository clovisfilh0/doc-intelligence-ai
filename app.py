import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import streamlit as st

# Carregar variáveis de ambiente (.env local)
load_dotenv()

# 1. Configuração da Página
st.set_page_config(
    page_title="DocIntelligence AI | Análise Inteligente de Documentos",
    page_icon="📄",
    layout="wide",
)

# 2. Inicialização do Session State
if "messages" not in st.session_state:
  st.session_state["messages"] = []
if "banco_vetorial" not in st.session_state:
  st.session_state["banco_vetorial"] = None
if "arquivo_atual" not in st.session_state:
  st.session_state["arquivo_atual"] = None

# 3. Barra Lateral (Sidebar)
with st.sidebar:
  st.image(
      "https://img.icons8.com/color/96/artificial-intelligence.png", width=64
  )
  st.title("DocIntelligence AI")
  st.caption("Plataforma de Inteligência Documental com RAG")

  st.divider()

  st.subheader("📥 Upload do Documento")
  uploaded_file = st.file_uploader(
      "Carregue o seu PDF corporativo para análise",
      type=["pdf"],
      help="Suporta manuais, contratos, relatórios e documentos em formato PDF.",
  )

  st.divider()
  st.subheader("🔑 Configuração da IA (Google Gemini)")

  api_key_input = st.text_input(
      "Chave da API do Google AI Studio (Opcional)",
      type="password",
      placeholder="Deixe em branco para usar a chave padrão",
      help=(
          "Insira a sua chave do Google AI Studio caso queira usar a sua"
          " própria."
      ),
  )

  # Resolução oculta da chave
  if api_key_input:
    google_api_key = api_key_input
  else:
    try:
      google_api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
      google_api_key = os.getenv("GOOGLE_API_KEY")

  # Botão para limpar a conversa e resetar o arquivo
  if st.button("🗑️ Limpar Conversa"):
    st.session_state["messages"] = []
    st.session_state["banco_vetorial"] = None
    st.session_state["arquivo_atual"] = None
    st.rerun()

  st.divider()
  st.markdown("---")
  st.caption(
      "Desenvolvido por **Francisco Clóvis** | Tech Stack: Python • Streamlit •"
      " LangChain • Gemini"
  )


# Funções auxiliares
def extrair_texto_pdf(arquivo_pdf):
  leitor = PdfReader(arquivo_pdf)
  texto_total = ""
  for pagina in leitor.pages:
    texto_pagina = pagina.extract_text()
    if texto_pagina:
      texto_total += texto_pagina + "\n"
  return texto_total


def dividir_texto(texto):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len,
  )
  return text_splitter.split_text(texto)


def criar_banco_vetorial(chunks):
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  # Cria um banco vetorial novo em memória isolado na sessão do usuário
  return Chroma.from_texts(texts=chunks, embedding=embeddings)


# 4. Área Principal do Aplicativo
st.title("📄 Assistente Inteligente de Documentos")
st.markdown("""
Converse com os seus documentos em tempo real. Faça perguntas, peça resumos e analise informações com o poder da **IA Generativa**.
""")

st.divider()

# Processamento do arquivo PDF enviado
if uploaded_file:
  # Se for um arquivo novo ou diferente do anterior, reseta o banco e o histórico
  if st.session_state["arquivo_atual"] != uploaded_file.name:
    with st.spinner("🔄 Processando, fatiando e vetorizando o documento..."):
      texto_extraido = extrair_texto_pdf(uploaded_file)
      chunks = dividir_texto(texto_extraido)

      # Reseta completamente o banco antigo antes de criar o novo
      st.session_state["banco_vetorial"] = criar_banco_vetorial(chunks)
      st.session_state["arquivo_atual"] = uploaded_file.name
      st.session_state["messages"] = []  # Reseta o histórico de mensagens

    st.sidebar.success(f"✅ Processado: {uploaded_file.name}")

# Mensagens de orientação
if not uploaded_file:
  st.info(
      "👈 **Para começar:** Faça o upload de um arquivo PDF na barra lateral."
  )
elif not google_api_key:
  st.warning(
      "⚠️ **Atenção:** Nenhuma chave de API configurada no servidor e nenhuma"
      " inserida na barra lateral."
  )
else:
  # Renderizar o histórico de mensagens da sessão atual
  for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])
      if "sources" in message and message["sources"]:
        with st.expander("📌 Ver trechos originais consultados no PDF"):
          for idx, fonte in enumerate(message["sources"], 1):
            st.markdown(f"**Trecho {idx}:**\n_{fonte}_")

  # Campo de entrada do chat
  if user_prompt := st.chat_input(
      "Pergunte algo sobre o documento carregado..."
  ):
    with st.chat_message("user"):
      st.markdown(user_prompt)
    st.session_state["messages"].append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("assistant"):
      with st.spinner("🔍 Consultando o documento..."):
        try:
          banco_vetorial = st.session_state["banco_vetorial"]
          documentos_relacionados = banco_vetorial.similarity_search(
              user_prompt, k=3
          )

          trechos_fonte = [
              doc.page_content for doc in documentos_relacionados
          ]
          contexto = "\n\n".join(trechos_fonte)

          llm = ChatGoogleGenerativeAI(
              model="gemini-3.8-flash",
              temperature=0.3,
              google_api_key=google_api_key,
          )

          prompt_template = ChatPromptTemplate.from_messages([
              (
                  "system",
                  "Você é um assistente especialista em análise de documentos."
                  " Utiliza os trechos de contexto fornecidos para responder à"
                  " pergunta de forma clara, precisa e concisa. Se a resposta"
                  " não estiver no contexto, informa diretamente que o"
                  " documento não contém essa informação.\n\nContexto do"
                  " Documento:\n{context}",
              ),
              ("human", "{input}"),
          ])

          prompt_formatado = prompt_template.format_prompt(
              context=contexto, input=user_prompt
          )
          resposta = llm.invoke(prompt_formatado.to_messages())

          if isinstance(resposta.content, list):
            texto_resposta = "".join([
                item.get("text", "")
                for item in resposta.content
                if isinstance(item, dict) and "text" in item
            ])
          else:
            texto_resposta = str(resposta.content)

          st.markdown(texto_resposta)

          with st.expander("📌 Ver trechos originais consultados no PDF"):
            for idx, fonte in enumerate(trechos_fonte, 1):
              st.markdown(f"**Trecho {idx}:**\n_{fonte}_")

          st.session_state["messages"].append({
              "role": "assistant",
              "content": texto_resposta,
              "sources": trechos_fonte,
          })

        except Exception as e:
          erro_msg = f"Ocorreu um erro ao processar a pergunta: {e}"
          st.error(erro_msg)