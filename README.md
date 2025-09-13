# 🩺 Medical AI-powered Question Answering Chatbot  

An **AI-powered Question Answering (QA) Chatbot** designed to provide **reliable medical information** using **Retrieval-Augmented Generation (RAG)**. Unlike traditional LLMs that may hallucinate, this chatbot grounds its responses in trusted sources like the *Gale Encyclopedia of Medicine*.  

---

## 🚀 Features  
- **LangChain** – conversational workflow & retrieval pipeline  
- **Hugging Face Embeddings** – semantic query understanding  
- **Mistral 7B** – LLM backend for high-quality natural language answers  
- **FAISS** – vector database for efficient similarity search  
- **Groq Cloud API** – enabling fast and scalable inference  
- **Streamlit UI** – user-friendly chatbot interface  
- **RAG Pipeline** – ensures context-aware and fact-grounded answers  

---

## 🛠️ Tech Stack  
- Python 3.9+  
- [LangChain](https://www.langchain.com/)  
- [Hugging Face](https://huggingface.co/)  
- [Mistral 7B](https://mistral.ai/)  
- [FAISS](https://faiss.ai/)  
- [Groq Cloud](https://groq.com/)  
- [Streamlit](https://streamlit.io/)  

---

## ⚙️ How It Works  
1. 📄 Load medical PDFs (e.g., Gale Encyclopedia of Medicine)  
2. ✂️ Split documents into chunks using LangChain text splitters  
3. 🔎 Generate embeddings with Hugging Face models  
4. 📦 Store embeddings in FAISS for similarity search  
5. 🔗 Retrieve relevant context when a user asks a question  
6. 🧠 Mistral LLM generates answers based only on retrieved context  
7. 💬 Streamlit chatbot UI delivers precise, reliable responses  


---

## 🔑 Environment Variables  
Create a `.env` file in the root directory with:  

```ini
HF_TOKEN=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key




