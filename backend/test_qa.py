from backend.qa_pipeline import ask_question

question = "What precedent is cited for contract breach?"
answer = ask_question(question)

print(answer)
