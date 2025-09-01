import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # LongMemeval Evaluation Notebook

    This notebook evaluates LongMemeval results using an LLM judge, similar to the `evaluate_longmemeval.py` script.
    """
    )
    return


@app.cell
def _():
    import sys
    import os

    # Add the experiments directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))
    OLLAMA_ENDPOINT = "http://localhost:11434/v1"
    os.environ["LLAMA_BASE_URL"] = OLLAMA_ENDPOINT
    os.environ["OPENAI_API_KEY"] = ""
    return


@app.cell
def _():
    from models.llm_judge import LLMJudge
    return (LLMJudge,)


@app.cell
def _():
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
    return (DEFAULT_PROMPT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Original command

    ```bash
    python evaluate/evaluate_longmemeval.py \
        --input-path ../../results/gpt_4_1_longmemeval_focused_results.csv \
        --output-path ../../results/gpt_4_1_longmemeval_focused_evaluated.csv \
        --model-name gpt-4.1-2025-04-14 \
        --output-column output \
        --question-column question \
        --correct-answer-column answer \
        --max-context-length 1_047_576  \
        --max-tokens-per-minute 2_000_000
    ```
    """
    )
    return


@app.cell
def _(DEFAULT_PROMPT):
    # Configuration inputs
    # input_path = "results/llama_3_2_longmemeval_focused_results.csv"
    # output_path = "results/llama_3_2_longmemeval_focused_evaluated.csv"
    # input_path = "results/llama_3_2_longmemeval_full_results.csv"
    # output_path = "results/llama_3_2_longmemeval_full_evaluated.csv"
    input_path = "results/gpt_oss_20b_longmemeval_focused_results.csv"
    output_path = "results/gpt_oss_20b_longmemeval_focused_evaluated.csv"

    # model_name = "llama3.2"
    model_name = "gpt-oss:20b"

    output_column = "output"

    question_column = "question"

    correct_answer_column = "answer"

    max_context_length = 1_047_576
    max_tokens_per_minute = 2_000_000

    custom_prompt = DEFAULT_PROMPT
    return (
        correct_answer_column,
        custom_prompt,
        input_path,
        max_context_length,
        max_tokens_per_minute,
        model_name,
        output_column,
        output_path,
        question_column,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Run in terminal before next cell

    ```bash
    sudo apt update && sudo apt install lshw -y
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve
    ```
    """
    )
    return


@app.cell
def _(model_name):
    # TODO: move this to a separate .py module
    # Taken from: https://github.com/ollama/ollama-python/blob/main/examples/pull.py
    from tqdm import tqdm
    from ollama import pull

    MODEL_NAME = model_name

    current_digest, bars = '', {}
    for progress in pull(MODEL_NAME, stream=True):
      digest = progress.get('digest', '')
      if digest != current_digest and current_digest in bars:
        bars[current_digest].close()

      if not digest:
        print(progress.get('status'))
        continue

      if digest not in bars and (total := progress.get('total')):
        bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

      if completed := progress.get('completed'):
        bars[digest].update(completed - bars[digest].n)

      current_digest = digest
    return


@app.cell
def _(
    LLMJudge,
    correct_answer_column,
    custom_prompt,
    model_name,
    output_column,
    question_column,
):
    # Create the LLM judge instance
    judge = LLMJudge(
        prompt=custom_prompt,
        model_name=model_name,
        output_column=output_column,
        question_column=question_column,
        correct_answer_column=correct_answer_column
    )
    return (judge,)


@app.cell
def _(
    input_path,
    judge,
    max_context_length,
    max_tokens_per_minute,
    output_path,
):
    try:
        judge.evaluate(
            input_path=input_path,
            output_path=output_path,
            max_context_length=max_context_length,
            max_tokens_per_minute=max_tokens_per_minute
        )
        print("✅ Judge done")
    except Exception as e:
        print(f"Error: {e}")
        print("❌ Judge failed")
    return


if __name__ == "__main__":
    app.run()
