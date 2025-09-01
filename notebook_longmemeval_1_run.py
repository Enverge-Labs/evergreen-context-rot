import marimo

__generated_with = "0.14.7"
app = marimo.App(width="full")


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
    s## Run in terminal before next cell

    ```bash
    sudo apt update && sudo apt install lshw -y
    curl -fsSL https://ollama.com/install.sh | sh
    ollama start
    ```
    """
    )
    return


@app.cell
def _():
    OLLAMA_MODELS = ["llama3.2", "gpt-oss:20b", "deepseek-r1:32b"]
    return (OLLAMA_MODELS,)


@app.cell
def _(OLLAMA_MODELS, mo):
    selected_model_dropdown = mo.ui.dropdown(
        options=OLLAMA_MODELS, 
        value=OLLAMA_MODELS[0],
        label="Choose your model:",
        searchable=True,
    )
    selected_model_dropdown
    return (selected_model_dropdown,)


@app.cell
def _(selected_model_dropdown):
    MODEL_NAME = selected_model_dropdown.value
    return (MODEL_NAME,)


@app.cell
def _(mo):
    test_conditions_dropdown = mo.ui.dropdown(
        options=["focused", "full"],
        value="focused",
        label="Choose test conditions:"
    )
    test_conditions_dropdown
    return (test_conditions_dropdown,)


@app.cell
def _(test_conditions_dropdown):
    test_conditions = test_conditions_dropdown.value
    return (test_conditions,)


# This works, but block the sequence of cell execution.
# @app.cell
# def _():
#     # TODO: start the Ollama server in a subprocess controlled through Python
#     # TODO: constanly poll for the output of the subprocess and print it to the console
#     import subprocess
#     import io
#     import sys
#     import selectors
    
#     # Start subprocess
#     # bufsize = 1 means output is line buffered
#     # universal_newlines = True is required for line buffering

#     process = subprocess.Popen(["ollama", "start"],
#                                bufsize=1,
#                                stdout=subprocess.PIPE,
#                                stderr=subprocess.STDOUT,
#                                universal_newlines=True)

#     # Create callback function for process output
#     buf = io.StringIO()
#     def handle_output(stream, mask):
#         # Because the process' output is line buffered, there's only ever one
#         # line to read when this function is called
#         line = stream.readline()
#         buf.write(line)
#         sys.stdout.write(line)

#     # Register callback for an "available for read" event from subprocess' stdout stream
#     selector = selectors.DefaultSelector()
#     selector.register(process.stdout, selectors.EVENT_READ, handle_output)

#     # Loop until subprocess is terminated
#     while process.poll() is None:
#         # Wait for events and handle them with their registered callbacks
#         events = selector.select()
#         for key, mask in events:
#             callback = key.data
#             callback(key.fileobj, mask)
#     return


@app.cell
def _(MODEL_NAME, mo):
    # TODO: move this to a separate .py module
    # Taken from: https://github.com/ollama/ollama-python/blob/main/examples/pull.py
    from ollama import pull

    bars, completed_amounts = {}, {}

    for progress in pull(MODEL_NAME, stream=True):
      digest = progress.get('digest', '')

      if not digest:
        # print(progress.get('status'))
        mo.output.append(progress.get('status'))
        continue

      if digest not in bars and (total := progress.get('total')):
        bars[digest] = mo.status.progress_bar(total=total).__enter__()
        completed_amounts[digest] = 0

      if completed := progress.get('completed'):
        # Calculate the increment since last update
        last_completed = completed_amounts.get(digest, 0)
        increment = completed - last_completed
        if increment > 0:
            bars[digest].update(increment=increment)
            completed_amounts[digest] = completed

    # Clean up progress bars
    for bar in bars.values():
        try:
            bar.__exit__(None, None, None)
        except:
            pass
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
def _(test_conditions):
    input_path = f"data/cleaned_longmemeval_s_{test_conditions}.csv"
    return (input_path,)


@app.cell
def _():
    import re

    def replace_non_alphanumeric(text):
        return re.sub(r'[^a-zA-Z0-9]', '_', text)
    return (replace_non_alphanumeric,)


@app.cell
def _(MODEL_NAME, replace_non_alphanumeric):
    model_name_slug = replace_non_alphanumeric(MODEL_NAME)
    output_path = f"results/{model_name_slug}_longmemeval_focused_results.csv"
    return (output_path,)


@app.cell
def _(test_conditions):
    input_column = f"{test_conditions}_prompt"
    return (input_column,)


@app.cell
def _(MODEL_NAME, input_column, input_path, output_path, provider):
    provider.main(
        input_path=input_path,
        output_path=output_path,
        input_column=input_column,
        output_column="output",
        model_name=MODEL_NAME,
        max_context_length=1_047_576,
        max_tokens_per_minute=2_000_000
    )
    return


if __name__ == "__main__":
    app.run()
