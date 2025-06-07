import unittest
from Qa_bot import load_documents, chunks, embedding, retriever, create_qa_chain

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestQA_Bot(unittest.TestCase):
    """Unit tests for the QA Bot functionality."""
    def test_load_documents(self):
        """Test loading documents from a PDF file."""
        pdf_file = "Test.pdf"
        documents = load_documents(pdf_file)

        url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/WgM1DaUn2SYPcCg_It57tA/A-Comprehensive-Review-of-Low-Rank-Adaptation-in-Large-Language-Models-for-Efficient-Parameter-Tuning-1.pdf"
        web_documents = load_documents(url, web_url=True)
        
        self.assertGreater(len(web_documents), 0, "No documents loaded from web URL.")
        self.assertGreater(len(documents), 0, "No documents loaded from PDF.")

    def test_chunks(self):
        """Test chunking function with LaTeX text."""
        latex_text = "Test.tex"
        latex_docs = chunks(latex_text, lang=True)

        self.assertGreater(len(latex_docs), 0, "No chunks created from LaTeX text.")

    def test_embeddings(self):
        """Test embedding functionality."""
        query = "How are you?"
        embeddings_model = embedding()
        embedded_query = embeddings_model.embed_query(query)

        self.assertIsInstance(embedded_query, list, "Embedded query is not a list.")

    def test_retriever(self):
        """Test retriever functionality."""
        doc = "new-Policies.txt"
        query = "Smoking Policy"
        retriever_instance = retriever(doc, k=5)
        embedded_doc = retriever_instance.get_relevant_documents(query)

        self.assertGreater(len(embedded_doc), 0, "No documents retrieved for the query.")

    def test_create_qa_chain(self):
        """Test the QA chain functionality."""
        doc = "new-Policies.txt"
        query = "Email policy"
        response = create_qa_chain(doc, query, k=2)

        self.assertIsInstance(response, str, "Response from QA chain is not a string.")

if __name__ == '__main__':
    unittest.main()