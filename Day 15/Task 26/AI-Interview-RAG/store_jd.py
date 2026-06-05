import chromadb

from src.text_loader import load_text_file
from src.chunker import chunk_text
from src.embeddings import generate_embeddings

jd_text = load_text_file("data/jd.txt")

chunks = chunk_text(jd_text)

embeddings = generate_embeddings(chunks)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="jd_collection"
)

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding.tolist()]
    )

print("JD Stored Successfully")