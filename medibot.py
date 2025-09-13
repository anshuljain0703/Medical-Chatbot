# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# # Page configuration
# st.set_page_config(
#     page_title="AI Chatbot Assistant",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Custom CSS for beautiful UI with proper prompt text visibility
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

# /* Main app styling */
# .stApp {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     font-family: 'Poppins', sans-serif;
# }

# /* Title styling */
# .main-title {
#     font-size: 3rem;
#     font-weight: 700;
#     text-align: center;
#     color: white;
#     text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#     margin-bottom: 2rem;
#     padding: 2rem;
#     background: rgba(255,255,255,0.1);
#     backdrop-filter: blur(10px);
#     border-radius: 20px;
#     border: 1px solid rgba(255,255,255,0.2);
# }

# /* Chat messages container */
# .stChatMessage {
#     backdrop-filter: blur(10px) !important;
#     border-radius: 15px !important;
#     margin: 10px 0 !important;
#     padding: 15px !important;
#     box-shadow: 0 8px 32px rgba(31,38,135,0.37) !important;
# }

# /* User message styling */
# div[data-testid="user-message"] {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
#     border-radius: 20px 20px 5px 20px !important;
#     margin-left: 20% !important;
#     box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
# }
# div[data-testid="user-message"] p,
# div[data-testid="user-message"] span,
# div[data-testid="user-message"] div {
#     color: #ffffff !important;
#     font-weight: 500 !important;
# }

# /* Assistant message styling */
# div[data-testid="assistant-message"] {
#     background: #ffffff !important;
#     border-radius: 20px 20px 20px 5px !important;
#     margin-right: 20% !important;
#     border-left: 4px solid #667eea !important;
#     box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
# }
# div[data-testid="assistant-message"] p,
# div[data-testid="assistant-message"] span,
# div[data-testid="assistant-message"] div {
#     color: #111827 !important;
#     font-weight: 500 !important;
# }

# /* Chat input styling (FIXED TEXT VISIBILITY) */
# .stChatInput > div > div {
#     background: rgba(255,255,255,0.05) !important; /* slightly visible background */
#     backdrop-filter: blur(10px) !important;
#     border-radius: 25px !important;
#     border: 2px solid rgba(255,255,255,0.3) !important;
# }
# .stChatInput input {
#     background: transparent !important;
#     border: none !important;
#     font-size: 16px !important;
#     font-family: 'Poppins', sans-serif !important;
#     padding: 15px 20px !important;
#     color: #ffffff !important;  /* TEXT VISIBLE */
# }
# .stChatInput input::placeholder {
#     color: #d1d5db !important;  /* placeholder light gray */
#     font-style: italic !important;
# }

# /* Error message styling */
# .stError {
#     background: rgba(239,68,68,0.1) !important;
#     border: 1px solid rgba(239,68,68,0.3) !important;
#     border-radius: 15px !important;
#     backdrop-filter: blur(10px) !important;
#     color: #dc2626 !important;
#     font-weight: 500 !important;
# }

# /* Container padding */
# .block-container {
#     padding-top: 2rem !important;
#     max-width: 800px !important;
# }

# /* Hide streamlit branding */
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}

# /* Scrollbar styling */
# ::-webkit-scrollbar { width: 8px; }
# ::-webkit-scrollbar-track { background: rgba(255,255,255,0.1); border-radius: 4px; }
# ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); border-radius: 4px; }
# ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.5); }

# /* Animation for messages */
# @keyframes slideIn {
#     from {opacity: 0; transform: translateY(20px);}
#     to {opacity: 1; transform: translateY(0);}
# }
# .stChatMessage { animation: slideIn 0.3s ease-out; }

# /* Spinner styling */
# .stSpinner { text-align: center; }
# .stSpinner > div { border-top-color: #4a90e2 !important; }
# </style>
# """, unsafe_allow_html=True)

# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token": HF_TOKEN, "max_length": "512"}
#     )
#     return llm

# def main():
#     # Beautiful title
#     st.markdown('<div class="main-title">🤖 Ask Chatbot!</div>', unsafe_allow_html=True)

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt = st.chat_input("✨ Pass your prompt here...")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role': 'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#             Use the pieces of information provided in the context to answer user's question.
#             If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#             Dont provide anything out of the given context

#             Context: {context}
#             Question: {question}

#             Start the answer directly. No small talk please.
#         """

#         try:
#             with st.spinner('🧠 Processing your question...'):
#                 vectorstore = get_vectorstore()
#                 if vectorstore is None:
#                     st.error("Failed to load the vector store")

#                 qa_chain = RetrievalQA.from_chain_type(
#                     llm=ChatGroq(
#                         model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
#                         temperature=0.0,
#                         groq_api_key=os.environ["GROQ_API_KEY"],
#                     ),
#                     chain_type="stuff",
#                     retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#                     return_source_documents=True,
#                     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#                 )

#                 response = qa_chain.invoke({'query': prompt})

#                 result = response["result"]
#                 source_documents = response["source_documents"]
#                 result_to_show = result + "\n\n**Source Docs:**\n" + str(source_documents)

#                 st.chat_message('assistant').markdown(result_to_show)
#                 st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()


import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat input visibility and beautiful UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Poppins', sans-serif; }
.main-title { font-size: 3rem; font-weight: 700; text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin-bottom: 2rem; padding: 2rem; background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 20px; border: 1px solid rgba(255,255,255,0.2); }
.stChatMessage { backdrop-filter: blur(10px) !important; border-radius: 15px !important; margin: 10px 0 !important; padding: 15px !important; box-shadow: 0 8px 32px rgba(31,38,135,0.37) !important; color: #fff; }
div[data-testid="user-message"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border-radius: 20px 20px 5px 20px !important; margin-left: 20% !important; box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important; }
div[data-testid="user-message"] p, div[data-testid="user-message"] span, div[data-testid="user-message"] div { color: #ffffff !important; font-weight: 500 !important; }
div[data-testid="assistant-message"] { background: #ffffff !important; border-radius: 20px 20px 20px 5px !important; margin-right: 20% !important; border-left: 4px solid #667eea !important; box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important; }
div[data-testid="assistant-message"] p, div[data-testid="assistant-message"] span, div[data-testid="assistant-message"] div { color: #111827 !important; font-weight: 500 !important; }
.stChatInput > div > div { background: rgba(255,255,255,0.05) !important; backdrop-filter: blur(10px) !important; border-radius: 25px !important; border: 2px solid rgba(255,255,255,0.3) !important; }
.stChatInput input { background: transparent !important; border: none !important; font-size: 16px !important; font-family: 'Poppins', sans-serif !important; padding: 15px 20px !important; color: #ffffff !important; }
.stChatInput input::placeholder { color: #d1d5db !important; font-style: italic !important; }
.stError { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; border-radius: 15px !important; backdrop-filter: blur(10px) !important; color: #dc2626 !important; font-weight: 500 !important; }
.block-container { padding-top: 2rem !important; max-width: 800px !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: rgba(255,255,255,0.1); border-radius: 4px; } ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); border-radius: 4px; } ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.5); }
@keyframes slideIn { from {opacity: 0; transform: translateY(20px);} to {opacity: 1; transform: translateY(0);} }
.stChatMessage { animation: slideIn 0.3s ease-out; }
.stSpinner { text-align: center; } .stSpinner > div { border-top-color: #4a90e2 !important; }
</style>
""", unsafe_allow_html=True)

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def main():
    st.markdown('<div class="main-title">🤖 Ask Chatbot!</div>', unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("✨ Pass your prompt here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
            Dont provide anything out of the given context
            Context: {context}
            Question: {question}
            Start the answer directly. No small talk please.
        """

        try:
            with st.spinner('🧠 Processing your question...'):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                        temperature=0.0,
                        groq_api_key=os.environ["GROQ_API_KEY"],
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})

                result = response["result"]
                source_documents = response["source_documents"]

                # Summarize source docs in 2 lines
                source_summary = ""
                for doc in source_documents[:2]:  # first 2 docs only
                    source_summary += f"- {doc.metadata.get('source', 'Unknown source')}\n"

                result_to_show = result + "\n\n**Source Docs (summary):**\n" + source_summary

                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
