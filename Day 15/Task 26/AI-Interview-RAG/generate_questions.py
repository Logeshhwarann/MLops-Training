import chromadb

from src.llm import generate_answer

client = chromadb.PersistentClient(path="./chroma_db")

resume_collection = client.get_collection("resume_collection")
jd_collection = client.get_collection("jd_collection")

# Get resume content
resume_docs = resume_collection.get()["documents"]

# Get JD content
jd_docs = jd_collection.get()["documents"]

resume_text = "\n".join(resume_docs)
jd_text = "\n".join(jd_docs)

prompt = f"""
Resume:
{resume_text}

Job Description:
{jd_text}

Generate interview questions based on both the resume and job description.

Requirements:

1. 10 Technical Questions
2. 5 Project-Based Questions
3. 5 HR Questions

The questions must be tailored specifically to the candidate's projects, skills, and the job requirements.
"""

answer = generate_answer(
    context=prompt,
    question="Generate interview questions"
)

print(answer)