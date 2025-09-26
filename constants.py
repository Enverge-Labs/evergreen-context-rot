OLLAMA_MODELS = ["deepseek-r1:14b", "gpt-oss:20b", "llama3.2", "mistral:7b"]

# Default judge prompt template
# WARNING: asked explicitly for a single word, otherwise Llama returns an answer for each part of the question.
DEFAULT_PROMPT = """
Given this question and the CORRECT answer, determine whether the response is correct (meaning it factually aligns with the correct answer). 
In some cases, 0 and "I do not have an answer" are considered to be both correct. 
If both responses say that there is no answer, this should be judged as true.
If the correct answer contains an answer, but the response abstains from answering, this should be judged as false.

Question: {question}

CORRECT answer: {correct_answer}

Response to judge: {output}

Instructions: Respond with only a single word: "true" if the response factually aligns with the correct answer, or "false" if it does not. Do not provide any explanation - just "true" or "false".
"""
