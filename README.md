

# 🔍 **AI Based Research Assistant**  
**An advanced research assistant powered by AI that automates literature reviews, summarizes papers, analyzes PDFs, and conducts web-based research efficiently.**

---

## 🚀 **Project Overview**
This AI-powered application leverages cutting-edge technologies to assist researchers by:
- 📚 **Searching Research Papers:** Fetches papers from Google Scholar based on a topic.  
- 📝 **Generating Literature Reviews:** Summarizes key findings, methodologies, and implications from the retrieved papers.  
- 📄 **Analyzing PDF Research Papers:** Extracts text from uploaded PDFs, uses RAG (Retrieval-Augmented Generation), and answers user questions about the content.  
- 🌐 **Web-Based Research:** Conducts research on any topic using DuckDuckGo and Wikipedia APIs, providing a structured summary.

---

## 🧠 **Key Functionalities**
1. **Search for Research Papers**
   - Retrieves research papers using Google Scholar.
   - Generates structured literature reviews and comprehensive summaries.
2. **Analyze PDF Research Papers**
   - Extracts and processes PDF content using `PyMuPDF`.
   - Sets up a RAG system with FAISS to answer questions about the uploaded paper.
3. **Web Research**
   - Uses DuckDuckGo and Wikipedia APIs for topic-based web research.
   - Summarizes the results in a structured format.

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
git clone https://github.com/atharvad38/ai-research-assistant.git
cd ai-research-assistant
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

### 4. **Set API Keys (Optional but Recommended)**
Create a `.env` file in the root directory and add:
```
ANTHROPIC_API_KEY=your-anthropic-api-key
```

---

## ▶️ **Usage**

### 1. **Run the Application**
```bash
streamlit run app.py
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

---

## ⚡ **Future Improvements**
- 🔐 Secure API key handling with environment variables.
- 🎯 Improve exception handling for API errors and parsing.
- 🚀 Scale to handle large document processing with distributed architecture.
- 📊 Integrate citation management and reference formatting.

---

## 🤝 **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

---

## 📧 **Contact**
For any questions or suggestions, contact [your-email@example.com](mailto:your-email@example.com).

---

## ⚠️ **Disclaimer**
This tool is intended for academic and research purposes. Always cross-check the information from multiple sources for critical research.

---

🔗 **GitHub Repository:** [AI Research Assistant](https://github.com/your-username/ai-research-assistant)

---

Let me know if you want to tweak anything! 🚀
