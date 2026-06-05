from src.llm import generate_answer

print("=" * 50)
print("AI MOCK INTERVIEW")
print("=" * 50)

question = input("\nInterview Question: ")

answer = input("\nYour Answer: ")

prompt = f"""
You are an expert technical interviewer.

Interview Question:
{question}

Candidate Answer:
{answer}

Evaluate the answer and provide:

1. Score out of 10
2. Strengths
3. Weaknesses
4. Improved Answer
5. Interviewer's Feedback
"""

result = generate_answer(
    context=prompt,
    question="Evaluate candidate answer"
)

print("\n")
print("=" * 50)
print("INTERVIEW FEEDBACK")
print("=" * 50)
print(result)