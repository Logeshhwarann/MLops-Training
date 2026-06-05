import chromadb

from src.llm import generate_answer

client = chromadb.PersistentClient(path="./chroma_db")

resume_collection = client.get_collection("resume_collection")
jd_collection = client.get_collection("jd_collection")

# Get all resume chunks
resume_docs = resume_collection.get()["documents"]

# Get all JD chunks
jd_docs = jd_collection.get()["documents"]

resume_text = "\n".join(resume_docs)
jd_text = "\n".join(jd_docs)

prompt = f"""
Analyze the resume against the job description.

Resume:
{resume_text}

Job Description:
{jd_text}

Provide:

1. Matching Skills
2. Missing Skills
3. Resume Match Score (0-100)
4. Suggestions to improve the resume
"""

answer = generate_answer(
    context=prompt,
    question="Analyze the resume"
)

print(answer)