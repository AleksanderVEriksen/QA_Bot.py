# Creating a simple QA bot using Document Loaders, Text Splitters, 
# Vector Stores, and RAG

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma


from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.chains import RetrievalQA

import gradio as gr

import warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os

# Get the environment variable for IBM Watsonx API key
WATSONX_APIKEY = os.environ.get("WATSONX_APIKEY")

# Define the model ID for the IBM Watsonx model
LLM_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"
EMBEDDING_MODEL_ID = "ibm/slate-125m-english-rtrvr"
PROJECT_ID = "1785fdb4-83bf-4bba-aa9b-8321ba695d6a"
URL = "https://eu-de.ml.cloud.ibm.com"


from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

parameters = {   
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 512,   
    GenTextParamsMetaNames.TEMPERATURE: 0.5,  
}  

# Initialize Watsonx credentials
def get_model():
    model_id = LLM_MODEL_ID
    project_id = PROJECT_ID
    watson_model = WatsonxLLM(
        model_id=model_id,
        project_id=project_id,
        url=URL,
        apikey=WATSONX_APIKEY,
        params=parameters,
    )
    return watson_model

# Initialize document loader
def load_documents(file_path, web_url:bool = False):
    """Load documents from a PDF file."""
    if web_url:
        loader = UnstructuredPDFLoader(file_path)
    else:
        loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

# Initialize chunking
def chunks(documents, lang:bool , language=Language.LATEX):
    """
    Split a document based on characters.
    """
    if lang:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
        )
        latex_doc = open(documents, "r", encoding="utf-8").read()
        docs = text_splitter.create_documents([latex_doc])
        return docs
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
        )
    
        splitted_documents = text_splitter.split_documents(documents)
        return splitted_documents


# Initialize embedding
def embedding():
    """
    Creates a Watsonx embeddings for documents.
    """
    embeddings = WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        project_id=PROJECT_ID,

        url=URL,
        apikey=WATSONX_APIKEY,
    )
    return embeddings

# Initialize vector store
def create_vectorDB(splitted_documents):
    """
    Creates a vector database for the prepared documents.

    Embeds the document before passing to the vector store.
    """
    ids = [str(i) for i in range(0, len(splitted_documents))]
    vector_db = Chroma.from_documents(
        documents=splitted_documents,
        embedding=embedding(),
        persist_directory="vector_db",
        ids=ids,
    )
    return vector_db

# Creates a Retriever
def retriever(file, lang:bool = False, k:int=0):
    """
    Creates a retriever for the vector store.

    Loads the documents, splits them, and creates a vector store.
    Retrieves the top k relevant documents based on the query.
    """
    docs = load_documents(file)
    splitted_documents = chunks(docs, lang)
    vector_db = create_vectorDB(splitted_documents)
    retriever = vector_db.as_retriever(search_type="similarity",search_kwargs={"k": k})
    return retriever

# Creates a RetrievalQA chain
def create_qa_chain(file, query, k: int = 5):
    """
    Sets up a QA based on the LLM and Retriever.
    """
    llm = get_model()
    ret = retriever(file, k=k)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ret,
        return_source_documents=False
    )
    # Run the QA chain with the provided query
    response = qa.run(query)
    return response

# Gradio interface for the QA bot
rag_application = gr.Interface(
    fn=create_qa_chain,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

def exercises():

    """Test loading documents from a PDF file."""
    url = "Test.pdf"
    web_documents = load_documents(url)
    print("\nLoaded web document from PDF:\n")
    print(web_documents[0].page_content[:1000])
    
    """Test chunking function with LaTeX text."""
    latex_text = "Test.tex"
    latex_docs = chunks(latex_text, lang=True)
    print("\nChunked LaTeX document:\n")
    for i, doc in enumerate(latex_docs):
        print(f"Chunk {i+1}:\n{doc.page_content[:200]}...\n")

    """Test embedding functionality."""
    query = "How are you?"
    embeddings_model = embedding()
    embedded_query = embeddings_model.embed_query(query)
    print(f"\nEmbedded query with: {query}\n")
    print(embedded_query[:5])

    """Test vectordb functionality."""
    doc = "new-Policies.txt"
    query = "Smoking Policy"
    doc = load_documents(doc)
    vector_db = create_vectorDB(doc)
    search_result = vector_db.similarity_search(query, k=5)
    print(f"\nRetrieved documents from DB with: {query}\n")
    for i, result in enumerate(search_result):
        print(f"Document {i+1}: {result.page_content[:200]}...")

    """Test retriever functionality."""
    doc = "new-Policies.txt"
    query = "Smoking Policy"
    retriever_instance = retriever(doc, k=2)
    embedded_doc = retriever_instance.invoke(query)
    print(f"\nRetrieved documents from retriever with: {query}:\n")
    for i, result in enumerate(embedded_doc):
        print(f"Document {i+1}: {result.page_content[:200]}...")


    """Test the QA chain functionality."""
    query = "Email policy"
    response = create_qa_chain(doc, query, k=5)
    print(f"\nResponse from QA chain with: {query}:\n")
    print(response)

if __name__ == "__main__":
    # Launch the app
    #exercises()
    rag_application.launch(server_name="0.0.0.0", server_port= 7860, debug=True)




