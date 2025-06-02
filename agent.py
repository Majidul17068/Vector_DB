from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import (
    PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME
)

class ChatAgent:
    def __init__(self):
        print("Initializing Chat Agent...")
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize Falcon model
        print("Loading Falcon model...")
        model_name = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize Pinecone
        print("Initializing Pinecone...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        
        print("✅ Chat Agent initialized successfully!")

    def get_relevant_context(self, query, top_k=5):
        """Search for relevant context in Pinecone"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            if not results.matches:
                return "No relevant information found in the document."
            
            # Extract and format contexts
            contexts = []
            for i, match in enumerate(results.matches, 1):
                text = match.metadata.get("text", "").strip()
                if text:
                    contexts.append(f"Context {i}: {text}")
            
            return "\n\n".join(contexts)
            
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return "Error retrieving context from the document."

    def generate_response(self, query, context):
        """Generate response using Falcon model"""
        try:
            # Create the prompt in Falcon's chat format
            prompt = f"""User: You are a helpful assistant that answers questions based on the provided document context. 
If the context doesn't contain enough information to answer the question, say so.
Only use information from the provided context to answer the question.

Document Context:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say "I don't have enough information in the document to answer this question completely"
3. Keep your answer concise and relevant
4. If the context contains multiple relevant points, include them all
5. Make sure your answer is complete and makes sense

Assistant:"""
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            
            # Generate response
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the model's response
            response = response.split("Assistant:")[-1].strip()
            
            # If response is too short or seems incomplete, add a note
            if len(response.split()) < 10:
                response = "I don't have enough information in the document to answer this question completely. Please try rephrasing your question or ask about a different topic."
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."

    def chat(self):
        """Main chat loop"""
        print("\n🤖 Chat Agent is ready! Type 'end chat' to exit.")
        print("----------------------------------------")
        print("I can help you find information from your document. Ask me anything!")
        
        while True:
            # Get user input
            query = input("\n👤 You: ").strip()
            
            # Check for exit command
            if query.lower() == "end chat":
                print("\n🤖 Chat Agent: Goodbye! Have a great day!")
                break
            
            try:
                # Get relevant context
                print("\n🔍 Searching document for relevant information...")
                context = self.get_relevant_context(query)
                print("\n📄 Found relevant information in the document:")
                print(context)
                
                # Generate response
                print("\n🤔 Generating response...")
                response = self.generate_response(query, context)
                
                print(f"\n🤖 Chat Agent: {response}")
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'end chat' to exit.")

if __name__ == "__main__":
    # Create and start chat agent
    agent = ChatAgent()
    agent.chat() 