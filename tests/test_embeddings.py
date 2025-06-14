import unittest
from src.rag.retriever import RAGRetriever
from src.rag.embeddings import TFIDFEmbeddings
import numpy as np

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.embeddings = TFIDFEmbeddings()
        self.retriever = RAGRetriever(embeddings=self.embeddings)
        
        # Sample test documents
        self.test_documents = [
            {"text": "The quick brown fox jumps over the lazy dog.", "metadata": {"source": "test1"}},
            {"text": "A fast orange fox leaps across a sleepy canine.", "metadata": {"source": "test2"}},
            {"text": "Python is a popular programming language.", "metadata": {"source": "test3"}}
        ]

    def test_embedding_generation(self):
        """Test if embeddings are correctly generated for a text."""
        text = "The quick brown fox"
        embedding = self.embeddings.embed_query(text)
        
        # Check if embedding is a numpy array
        self.assertIsInstance(embedding, np.ndarray)
        # Check if embedding has the correct shape
        self.assertEqual(embedding.shape[0], len(self.embeddings.vectorizer.get_feature_names_out()))

    def test_document_embedding(self):
        """Test if documents are correctly embedded and stored."""
        # Add documents to the retriever
        self.retriever.add_documents(self.test_documents)
        
        # Test retrieval with a similar query
        query = "What does the fox do?"
        results = self.retriever.retrieve(query, self.test_documents)
        
        # Check if results are returned
        self.assertGreater(len(results), 0)
        # Check if results contain the expected documents
        self.assertTrue(any("fox" in doc["text"].lower() for doc in results))

    def test_embedding_consistency(self):
        """Test if the same text always generates the same embedding."""
        text = "Test text for consistency"
        embedding1 = self.embeddings.embed_query(text)
        embedding2 = self.embeddings.embed_query(text)
        
        # Check if embeddings are identical
        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        empty_doc = {"text": "", "metadata": {"source": "empty"}}
        self.retriever.add_documents([empty_doc])
        
        # Should not raise an error
        results = self.retriever.retrieve("test query", [empty_doc])
        self.assertIsInstance(results, list)

    def test_similarity_calculation(self):
        """Test if similarity calculations work correctly."""
        # Add documents
        self.retriever.add_documents(self.test_documents)
        
        # Test with similar and dissimilar queries
        similar_query = "fox jumping"
        dissimilar_query = "python programming"
        
        similar_results = self.retriever.retrieve(similar_query, self.test_documents)
        dissimilar_results = self.retriever.retrieve(dissimilar_query, self.test_documents)
        
        # Similar query should return fox-related documents first
        self.assertTrue(any("fox" in doc["text"].lower() for doc in similar_results[:2]))
        # Dissimilar query should return python-related document
        self.assertTrue(any("python" in doc["text"].lower() for doc in dissimilar_results[:1]))

if __name__ == '__main__':
    unittest.main() 