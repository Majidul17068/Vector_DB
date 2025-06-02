from sentence_transformers import SentenceTransformer

def test_embedding():
    try:
        # Load the model
        print("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Using a reliable, public model
        
        # Test embedding
        text = "Hello world"
        embedding = model.encode(text)
        
        print("✅ Embedding test successful!")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First few values: {embedding[:5].tolist()}")
        
    except Exception as e:
        print(f"❌ Error testing embeddings: {str(e)}")

if __name__ == "__main__":
    test_embedding()
