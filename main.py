import os
import fitz
import streamlit as st
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()

# Azure AI Search setup
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX")

# Azure OpenAI setup
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Create search index
def create_index():
    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_key)
    )
    fields = [
        SimpleField(name="id", type="Edm.String", key=True),
        SearchableField(name="content", type="Edm.String")
    ]
    index = SearchIndex(name=index_name, fields=fields)
    index_client.create_or_update_index(index)

if "index_created" not in st.session_state:
    try:
        create_index()
        st.session_state.index_created = True
    except Exception:
        st.session_state.index_created = True

# Streamlit UI
st.title("📄 THE AURA RAG")
st.caption("Upload a PDF. Ask anything from it.")

# Step 1 - Upload PDF
st.subheader("Step 1 — Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # Split into chunks
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    # Index into Azure AI Search
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key)
    )
    documents = [{"id": str(i), "content": chunk} for i, chunk in enumerate(chunks)]
    search_client.upload_documents(documents)

    st.success(f"✅ Uploaded {len(chunks)} chunks from {uploaded_file.name}")

    # Step 2 - Ask a question
    st.subheader("Step 2 — Ask a question")
    question = st.text_input("Type your question here...")

    if st.button("Ask"):
        # Search Azure AI Search
        results = search_client.search(question, top=3)
        context = " ".join([r["content"] for r in results])

        # Ask Azure OpenAI
        response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": f"Answer only from this document: {context}"},
                {"role": "user", "content": question}
            ],
            max_tokens=300
        )

        st.write("**Answer:**")
        st.write(response.choices[0].message.content)