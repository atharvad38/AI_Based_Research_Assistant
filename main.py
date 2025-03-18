import streamlit as st
import anthropic
import fitz  # PyMuPDF for PDF processing
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import textwrap
import re
from scholarly import scholarly
import time
import pyttsx3
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# Set API Key for Claude
ANTHROPIC_API_KEY = "YOUR_CLAUDE_API_KEY"

# Define response format for LangChain agent
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize LangChain components
@st.cache_resource
def load_langchain_components():
    # Initialize model
    model = ChatAnthropic(model='claude-3-sonnet-20240229', api_key=ANTHROPIC_API_KEY)

    # Set up response parser
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a research assistant that will help generate research summaries.
            Use the search or Wikipedia tool if necessary.
            Wrap the output in this format: {format_instructions}
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # Define tools
    duck = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="search",
        func=duck.run,
        description="Search the web for information based on a given query."
    )

    wiki_api = WikipediaAPIWrapper()
    wiki_tool = Tool(
        name="wikipedia",
        func=WikipediaQueryRun(api_wrapper=wiki_api).run,
        description="Search Wikipedia for information based on a given query."
    )

    # List of tools
    tools = [search_tool, wiki_tool]

    # Set up the agent
    agent = create_tool_calling_agent(
        llm=model,
        prompt=prompt,
        tools=tools
    )

    # Create AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return parser, agent_executor

# Function to create embeddings
def create_embeddings(texts, model):
    return model.encode(texts)

# Function to chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        # If we're not at the end and can look for a good break point
        if end < text_length and end - start > overlap:
            # Try to find a period, question mark, or newline to break on
            break_chars = ['.', '?', '!', '\n']
            for char in break_chars:
                last_break = text.rfind(char, start + chunk_size - overlap, end)
                if last_break != -1:
                    end = last_break + 1  # Include the break character
                    break

        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position, ensuring we don't create tiny chunks
        start = end - overlap if end < text_length else end

    return chunks

def fetch_google_scholar_papers(topic, max_results=5):
    papers = []

    try:
        # Query Google Scholar
        search_query = scholarly.search_pubs(topic)

        # Retrieve up to max_results papers
        for i in range(max_results):
            try:
                paper = next(search_query)

                # Extract relevant information
                title = paper.get('bib', {}).get('title', 'No title available')
                abstract = paper.get('bib', {}).get('abstract', 'No abstract available')
                url = paper.get('pub_url', '')
                year = paper.get('bib', {}).get('pub_year', 'Year unknown')
                authors = paper.get('bib', {}).get('author', 'Authors unknown')

                if isinstance(authors, list):
                    authors = ', '.join(authors)

                # Add to our list of papers
                papers.append({
                    "title": title,
                    "link": url,
                    "summary": abstract,
                    "year": year,
                    "authors": authors
                })

                # Add a small delay to avoid being rate-limited
                time.sleep(0.5)

            except StopIteration:
                break  # No more results
            except Exception as e:
                st.warning(f"Error retrieving paper: {e}")
                continue  # Try the next one

    except Exception as e:
        st.error(f"Error searching Google Scholar: {e}")

    return papers

def generate_literature_review(papers):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Format the papers data for input
    papers_text = ""
    for i, p in enumerate(papers):
        papers_text += f"Paper {i+1}:\n"
        papers_text += f"Title: {p['title']}\n"
        papers_text += f"Authors: {p.get('authors', 'Unknown')}\n"
        papers_text += f"Year: {p.get('year', 'N/A')}\n"
        papers_text += f"Summary: {p['summary']}\n\n"

    prompt = f"""You are a research assistant. Create a structured literature review table for the following papers.
    For each paper, provide:
    1. The full title
    2. Publication year
    3. Authors (first author et al. if more than 3)
    4. A concise but informative 2-3 sentence summary of key findings and methodology
    5. Research implications
    
    Format this as a scholarly literature review.
    
    Papers:
    {papers_text}"""

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def summarize_papers_claude(papers):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    text = "\n\n".join([f"Title: {p['title']}\nYear: {p.get('year', 'N/A')}\nSummary: {p['summary']}" for p in papers])
    prompt = f"""You are a research assistant. Summarize these research papers in detail, highlighting:
    1. Key findings
    2. Methodologies used
    3. Practical implications
    4. How they relate to each other

    Papers:
    {text}"""

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        # Clean the text - remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def setup_rag_from_pdf(pdf_text, embedding_model):
    # Create chunks
    chunks = chunk_text(pdf_text)
    if not chunks:
        return None, None

    # Create metadata to keep track of chunks
    metadata = [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]

    # Create embeddings
    embeddings = create_embeddings(chunks, embedding_model)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings)

    return index, metadata

def answer_question_with_rag(question, pdf_text, embedding_model):
    # Setup RAG if not already done
    if 'faiss_index' not in st.session_state or 'chunks_metadata' not in st.session_state:
        st.session_state.faiss_index, st.session_state.chunks_metadata = setup_rag_from_pdf(pdf_text, embedding_model)
        if st.session_state.faiss_index is None:
            return "Could not process the document properly. Please try uploading again."

    # Create query embedding
    query_embedding = embedding_model.encode([question])
    faiss.normalize_L2(query_embedding)

    # Search for similar chunks
    k = min(5, len(st.session_state.chunks_metadata))  # Get top k chunks
    distances, indices = st.session_state.faiss_index.search(query_embedding, k)

    # Retrieve relevant chunks
    relevant_chunks = [st.session_state.chunks_metadata[idx]["text"] for idx in indices[0]]

    # Combine chunks for context
    context = "\n\n".join(relevant_chunks)

    # Use Claude to answer the question
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""You are a research paper expert. Answer the following question based ONLY on the provided context from a research paper.
    
    Question: {question}
    
    Context from paper:
    {context}
    
    If the answer cannot be found in the context, simply state that you don't have enough information to answer accurately.
    """

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None

# Streamlit UI
st.title("🔍 AI Research Assistant")
st.write("Use this tool to search for papers, analyze PDFs, or research topics from the web.")

# Load resources
embedding_model = load_embedding_model()
parser, agent_executor = load_langchain_components()

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Search for Papers", "Analyze PDF", "Web Research"])

with tab1:
    st.header("Search for Research Papers")
    topic = st.text_input("Enter a research topic:")
    max_results = st.slider("Maximum number of papers to retrieve", 3, 15, 5)

    if st.button("Search Papers"):
        if topic:
            with st.spinner("Searching for papers on Google Scholar..."):
                papers = fetch_google_scholar_papers(topic, max_results)

            if papers:
                st.success(f"Found {len(papers)} papers on {topic}")
                st.write("### Research Papers Found:")

                # Display papers with collapsible summaries
                for i, p in enumerate(papers):
                    with st.expander(f"{i+1}. {p['title']} ({p.get('year', 'N/A')})"):
                        st.write(f"**Authors**: {p.get('authors', 'Not available')}")
                        st.write(f"**Summary**: {p['summary']}")
                        if p['link']:
                            st.write(f"**Link**: [{p['link']}]({p['link']})")
                        else:
                            st.write("**Link**: Not available")

                # Generate literature review
                with st.spinner("Generating literature review..."):
                    lit_review = generate_literature_review(papers)

                st.write("### Literature Review:")
                st.markdown(lit_review)

                # Generate comprehensive summary
                with st.spinner("Generating comprehensive summary of all papers..."):
                    summary = summarize_papers_claude(papers)

                st.write("### Comprehensive Summary of All Papers:")
                st.markdown(summary)

            else:
                st.warning("No papers found for this topic. Try modifying your search terms.")
        else:
            st.warning("Please enter a topic to search for papers.")

with tab2:
    st.header("Analyze Research Paper")
    uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Processing your PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.session_state.pdf_text = pdf_text

            # Reset RAG components when a new file is uploaded
            if 'faiss_index' in st.session_state:
                del st.session_state.faiss_index
            if 'chunks_metadata' in st.session_state:
                del st.session_state.chunks_metadata

        if pdf_text:
            st.success("PDF processed successfully!")

            # Initialize RAG in background
            with st.spinner("Setting up RAG system..."):
                st.session_state.faiss_index, st.session_state.chunks_metadata = setup_rag_from_pdf(pdf_text, embedding_model)

            # Preview of extracted text
            with st.expander("Preview extracted text"):
                st.text_area("First 2000 characters of paper content", pdf_text[:2000], height=200)

            # Question answering section
            st.write("### Ask Questions About the Paper")
            question = st.text_input("Enter your question about the paper:")

            if st.button("Get Answer"):
                if question:
                    with st.spinner("Analyzing the paper to answer your question..."):
                        answer = answer_question_with_rag(question, pdf_text, embedding_model)

                    st.write("### Answer:")
                    st.markdown(answer)

                    # Text-to-speech option
                    if st.button("Read Answer Aloud"):
                        text_to_speech(answer)
                else:
                    st.warning("Please enter a question about the paper.")
        else:
            st.error("Could not extract text from the uploaded PDF. Please try another file.")

# Replace the existing web research result parsing code in tab3 with this:
with tab3:
    st.header("Web Research")
    st.write("Enter a topic to research, and the AI will search the web for information.")

    query = st.text_input("Enter your research topic for web search:")
    if st.button("Research Web") and query:
        with st.spinner("Fetching information from the web..."):
            result = agent_executor.invoke({"input": query})

            try:
                # Handle the raw output from the agent
                raw_output = result.get("output", "")

                # Try to extract the JSON result from the output
                if isinstance(raw_output, list) and len(raw_output) > 0:
                    # If it's a list of outputs, get the last one which should contain the result
                    json_text = raw_output[-1].get("text", "")
                else:
                    # If it's a string, use it directly
                    json_text = raw_output if isinstance(raw_output, str) else str(raw_output)

                # Try to find the JSON result between the <result> tags
                result_match = re.search(r'<result>(.+?)</result>', json_text, re.DOTALL)
                if result_match:
                    json_text = result_match.group(1)
                    structured_response = parser.parse(json_text)

                    st.subheader("📌 Research Summary")
                    st.write(f"**Topic:** {structured_response.topic}")
                    st.write(f"**Summary:** {structured_response.summary}")
                    st.write("**Sources:**")
                    for source in structured_response.sources:
                        st.write(f"- {source}")
                    st.write(f"**Tools Used:** {', '.join(structured_response.tools_used)}")
                else:
                    # If no result tags found, try to parse the entire text as JSON
                    try:
                        structured_response = parser.parse(json_text)

                        st.subheader("📌 Research Summary")
                        st.write(f"**Topic:** {structured_response.topic}")
                        st.write(f"**Summary:** {structured_response.summary}")
                        st.write("**Sources:**")
                        for source in structured_response.sources:
                            st.write(f"- {source}")
                        st.write(f"**Tools Used:** {', '.join(structured_response.tools_used)}")
                    except:
                        st.error("Could not parse the response into the expected format.")
                        st.write("Raw response:")
                        st.json(result)
            except Exception as e:
                st.error(f"Error parsing response: {str(e)}")
                st.write("Raw response:")
                st.json(result)

# Add information about the app
st.sidebar.title("About")
st.sidebar.info(
    "This AI Research Assistant helps you:\n"
    "1. Search for research papers on a topic using Google Scholar\n"
    "2. Get comprehensive literature reviews and summaries\n"
    "3. Upload your own papers and ask questions\n"
    "4. Research topics from the web using search and Wikipedia\n"
    "\nPowered by Claude AI and FAISS for semantic search."
)

# Add a note about scholarly usage
st.sidebar.warning(
    "Note: For deep research, please don't rely solely on these findings and insights. "
    "Always verify information from multiple sources."
)
