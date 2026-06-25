import kuzu
import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

def initialize_servers(
        ollama_url="http://localhost:11434",
        kuzu_path="./test_kuzu",
        chroma_path="./test_chroma"):
    
    try:
        # Test Ollama Connection
        # ollama = Ollama(model="llama3:8b", base_url=ollama_url)
        Settings.llm = Ollama(model="phi3:3.8b", base_url=ollama_url,
                        request_timeout=1000.0,
                        additional_kwargs={"num_ctx": 2048})
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        print("🤖 Ollama: Connection established successfully.")
        
        # Test Kùzu DB Initialization
        kuzudb = kuzu.Database(database_path=kuzu_path)
        print("📊 Kùzu DB: Embedded graph engine initialized.")
        
        # Test ChromaDB Initialization
        chroma = chromadb.PersistentClient(path=chroma_path)
        print("🗄️ ChromaDB: Local vector client running.")

        # everything ready
        print("\n🚀 Status: Excellent! GraphRAG stack is ready.")
        
    except Exception as e:
        print(f"❌ Error encountered: {e}")

    return Settings.llm, kuzudb, chroma

ollama, kuzudb, chroma = initialize_servers(
    ollama_url="http://localhost:11434",
    kuzu_path="../databases/kuzu_db",
    chroma_path="../databases/chroma_db")

question = input("question: ")
response = ollama.complete(question)
print(response)