import os
import uuid
import fitz  # PyMuPDF
import streamlit as st
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX")

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# ---------------------------------------------------------
# Create / update the search index
# NOTE: we now also store the source filename so we can
# tell the user which document an answer came from.
# ---------------------------------------------------------
def create_index():
    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_key),
    )
    fields = [
        SimpleField(name="id", type="Edm.String", key=True),
        SearchableField(name="content", type="Edm.String"),
        SearchableField(name="source", type="Edm.String", filterable=True),
    ]
    index = SearchIndex(name=index_name, fields=fields)
    index_client.create_or_update_index(index)


if "index_created" not in st.session_state:
    try:
        create_index()
        st.session_state.index_created = True
    except Exception as e:
        st.session_state.index_created = False
        st.error(f"Failed to create/update index: {e}")

# Reusable, single search client
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_key),
)

# Track which files have already been indexed this session
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# ---------------------------------------------------------
# Helper: extract text from a PDF and chunk it
# ---------------------------------------------------------
def extract_and_chunk(uploaded_file, chunk_size=1000):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# ---------------------------------------------------------
# Helper: index a PDF into Azure AI Search
# Each chunk gets a globally unique id (uuid4) so the two
# PDFs' chunks never collide / overwrite each other, and a
# "source" field records which file it came from.
# ---------------------------------------------------------
def index_pdf(uploaded_file):
    chunks = extract_and_chunk(uploaded_file)
    documents = [
        {
            "id": str(uuid.uuid4()),
            "content": chunk,
            "source": uploaded_file.name,
        }
        for chunk in chunks
    ]
    if documents:
        search_client.upload_documents(documents)
    return len(chunks)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("📄 The Aura Rag")
st.caption("Upload two PDFs. Ask anything — the answer is drawn from whichever document is most relevant.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1 — Upload first PDF")
    file_one = st.file_uploader("Choose PDF #1", type="pdf", key="file_one")
    if file_one and file_one.name not in st.session_state.indexed_files:
        with st.spinner(f"Indexing {file_one.name}..."):
            n_chunks = index_pdf(file_one)
        st.session_state.indexed_files.add(file_one.name)
        st.success(f"✅ Indexed {n_chunks} chunks from {file_one.name}")
    elif file_one:
        st.info(f"{file_one.name} already indexed.")

with col2:
    st.subheader("Step 2 — Upload second PDF")
    file_two = st.file_uploader("Choose PDF #2", type="pdf", key="file_two")
    if file_two and file_two.name not in st.session_state.indexed_files:
        with st.spinner(f"Indexing {file_two.name}..."):
            n_chunks = index_pdf(file_two)
        st.session_state.indexed_files.add(file_two.name)
        st.success(f"✅ Indexed {n_chunks} chunks from {file_two.name}")
    elif file_two:
        st.info(f"{file_two.name} already indexed.")

st.divider()

# ---------------------------------------------------------
# Step 3 - Ask a question (independent of the upload blocks,
# so it works regardless of upload order and doesn't require
# re-uploading to re-trigger).
# ---------------------------------------------------------
st.subheader("Step 3 — Ask a question")

if len(st.session_state.indexed_files) == 0:
    st.warning("Upload at least one PDF before asking a question.")
else:
    question = st.text_input("Type your question here...")

    if st.button("Ask") and question:
        with st.spinner("Searching documents..."):
            # top=3 most relevant chunks across BOTH documents.
            # Azure AI Search ranks by relevance score, so if one
            # document is more relevant to the question, its chunks
            # will naturally dominate the top results.
            results = list(search_client.search(question, top=3))

        if not results:
            st.warning("No relevant content found in the indexed documents.")
        else:
            context = " ".join([r["content"] for r in results])
            sources_used = sorted({r.get("source") or "Unknown source" for r in results})

            with st.spinner("Asking Azure OpenAI..."):
                response = openai_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer only using the provided context. "
                                "If the answer is not contained in the context, say so.\n\n"
                                f"Context:\n{context}"
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                    max_tokens=300,
                )

            st.write("**Answer:**")
            st.write(response.choices[0].message.content)
            