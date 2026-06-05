import streamlit as st
import chromadb

from src.pdf_loader import extract_text_from_pdf
from src.text_loader import load_text_file
from src.chunker import chunk_text
from src.embeddings import generate_embeddings
from src.llm import generate_answer

st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Interview Assistant")

st.write("Upload your Resume and Job Description")

# Upload Files
resume_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

jd_file = st.file_uploader(
    "Upload Job Description (TXT)",
    type=["txt"]
)

# Save Uploaded Files
if resume_file:
    with open("data/resume.pdf", "wb") as f:
        f.write(resume_file.getbuffer())

if jd_file:
    with open("data/jd.txt", "wb") as f:
        f.write(jd_file.getbuffer())

# Process Button
if st.button("Process Documents"):

    client = chromadb.PersistentClient(path="./chroma_db")

    # Resume Processing
    resume_text = extract_text_from_pdf("data/resume.pdf")

    resume_chunks = chunk_text(resume_text)

    resume_embeddings = generate_embeddings(resume_chunks)

    resume_collection = client.get_or_create_collection(
        name="resume_collection"
    )

    try:
        client.delete_collection("resume_collection")
    except:
        pass

    resume_collection = client.get_or_create_collection(
        name="resume_collection"
    )

    for i, (chunk, embedding) in enumerate(
        zip(resume_chunks, resume_embeddings)
    ):
        resume_collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding.tolist()]
        )

    # JD Processing
    jd_text = load_text_file("data/jd.txt")

    jd_chunks = chunk_text(jd_text)

    jd_embeddings = generate_embeddings(jd_chunks)

    try:
        client.delete_collection("jd_collection")
    except:
        pass

    jd_collection = client.get_or_create_collection(
        name="jd_collection"
    )

    for i, (chunk, embedding) in enumerate(
        zip(jd_chunks, jd_embeddings)
    ):
        jd_collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding.tolist()]
        )

    st.success("Documents Processed Successfully!")

# Skill Gap Analysis
if st.button("Analyze Resume"):

    client = chromadb.PersistentClient(path="./chroma_db")

    resume_collection = client.get_collection(
        "resume_collection"
    )

    jd_collection = client.get_collection(
        "jd_collection"
    )

    resume_docs = resume_collection.get()["documents"]
    jd_docs = jd_collection.get()["documents"]

    resume_text = "\n".join(resume_docs)
    jd_text = "\n".join(jd_docs)

    prompt = f"""
Resume:
{resume_text}

Job Description:
{jd_text}

Provide:

1. Matching Skills
2. Missing Skills
3. Resume Match Score
4. Suggestions
"""

    result = generate_answer(
        context=prompt,
        question="Analyze Resume"
    )

    st.subheader("Resume Analysis")

    st.write(result)

# Generate Questions
if st.button("Generate Interview Questions"):

    client = chromadb.PersistentClient(path="./chroma_db")

    resume_collection = client.get_collection(
        "resume_collection"
    )

    jd_collection = client.get_collection(
        "jd_collection"
    )

    resume_docs = resume_collection.get()["documents"]
    jd_docs = jd_collection.get()["documents"]

    resume_text = "\n".join(resume_docs)
    jd_text = "\n".join(jd_docs)

    prompt = f"""
Resume:
{resume_text}

Job Description:
{jd_text}

Generate:

1. 10 Technical Questions
2. 5 Project Questions
3. 5 HR Questions
"""

    result = generate_answer(
        context=prompt,
        question="Generate Interview Questions"
    )

    st.subheader("Interview Questions")

    st.write(result)