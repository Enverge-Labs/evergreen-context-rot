import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    # Value taken from the terminal output
    OLLAMA_ENDPOINT = "http://localhost:11434/v1"
    os.environ["LLAMA_BASE_URL"] = OLLAMA_ENDPOINT
    return


@app.cell(hide_code=True)
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
def _():
    # MODEL_NAME = 'llama3.2'
    MODEL_NAME = 'gpt-oss:20b'
    return (MODEL_NAME,)


@app.cell
def _(MODEL_NAME):
    # TODO: move this to a separate .py module
    # Taken from: https://github.com/ollama/ollama-python/blob/main/examples/pull.py
    from tqdm import tqdm
    from ollama import pull

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
def _():
    from experiments.models.providers.llama import LlamaProvider
    return (LlamaProvider,)


@app.cell
def _(LlamaProvider):
    provider = LlamaProvider()
    return (provider,)


@app.cell
def _(MODEL_NAME, provider):
    provider.main(
        # input_path="data/cleaned_longmemeval_s_focused.csv",
        # output_path="results/llama_3_2_longmemeval_focused_results.csv",
        # input_column="focused_prompt",
        input_path="data/cleaned_longmemeval_s_full.csv",
        output_path="results/gpt_oss_20b_longmemeval_full_results.csv",
        input_column="full_prompt",

        output_column="output",
        model_name=MODEL_NAME,
        max_context_length=1_047_576,
        max_tokens_per_minute=2_000_000
    )
    return


if __name__ == "__main__":
    app.run()
