

# 🔍 **AI-Based Research Assistant**  
**An AI-powered research assistant that automates literature reviews, summarizes papers, analyzes PDFs, and conducts web-based research efficiently.**

---

## 🚀 **Project Overview**
This AI-powered application leverages cutting-edge technologies to assist researchers by:
- 📚 **Searching Research Papers:** Fetches research papers from Google Scholar based on a given topic.  
- 📝 **Generating Literature Reviews:** Summarizes key findings, methodologies, and implications from the retrieved papers.  
- 📄 **Analyzing PDF Research Papers:** Extracts text from uploaded PDFs, uses RAG (Retrieval-Augmented Generation), and answers user questions about the content.  
- 🌐 **Web-Based Research:** Conducts research on any topic using DuckDuckGo and Wikipedia APIs, providing a structured summary.

---

## 🧠 **Key Functionalities**
1. **Search for Research Papers**
   - Retrieves research papers using Google Scholar API (`scholarly`).
   - Generates structured literature reviews and comprehensive summaries.
2. **Analyze PDF Research Papers**
   - Extracts and processes PDF content using `PyMuPDF`.
   - Sets up a RAG system with FAISS to answer questions about the uploaded paper.
3. **Web Research**
   - Uses DuckDuckGo and Wikipedia APIs for topic-based web research.
   - Summarizes the results in a structured format using LangChain.

---

## 🛠️ **Tech Stack**
- **Backend:** Python, LangChain, FAISS, PyMuPDF  
- **Frontend:** Streamlit  
- **AI Models:** Claude-3 Sonnet (via Anthropic API), SentenceTransformer (`all-MiniLM-L6-v2`)  
- **APIs & Tools:**  
   - Google Scholar API (`scholarly`)  
   - DuckDuckGo API  
   - Wikipedia API  
- **Libraries:** LangChain, FAISS, anthropic, scholarly, pyttsx3, PyMuPDF, re, numpy, pandas

---

## 📄 **Installation & Setup**

### 1. **Clone the Repository**
```bash
git clone https://github.com/atharvad38/AI_Based_Research_Assistant.git
cd AI_Based_Research_Assistant
```

### 2. **Create a Virtual Environment**
```bash
# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. **Install Required Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Set API Keys**
Create a `.env` file in the root directory and add:
```
ANTHROPIC_API_KEY=your-anthropic-api-key
```

---

## ▶️ **Usage**

### 1. **Run the Application**
```bash
streamlit run main.py
```

### 2. **Explore Features**
- **Search for Research Papers:** Enter a research topic and retrieve papers.
- **Analyze PDF Paper:** Upload a research paper PDF and ask questions.
- **Web Research:** Search the web for any topic and get a structured summary.

---

## 📚 **How It Works**

### 📝 **PDF Analysis with RAG**
- Chunks text from PDFs using `PyMuPDF` and creates vector embeddings with `SentenceTransformer`.
- FAISS indexes these embeddings to enable semantic search.
- Retrieves relevant chunks and uses Claude AI to generate answers.

### 🔎 **Web Research with LangChain**
- Uses DuckDuckGo and Wikipedia APIs.
- Claude AI processes retrieved content and formats it as a structured summary.


## ⚠️ **Disclaimer**
This tool is intended for academic and research purposes. Always cross-check the information from multiple sources for critical research.

---

🔗 **GitHub Repository:** [AI-Based Research Assistant](https://github.com/atharvad38/AI_Based_Research_Assistant)

