import chromadb

from src.embeddings import model
from src.llm import generate_answer

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("resume_collection")

question = input("Ask a question: ")

query_embedding = model.encode(question)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

context = "\n".join(results["documents"][0])

answer = generate_answer(
    context=context,
    question=question
)

print("\nAnswer:\n")
print(answer)