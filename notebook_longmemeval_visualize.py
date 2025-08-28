import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Original instructions to generate plots

    ```bash

    python evaluate/visualize.py \
        --focused-path ../../results/gpt_4_1_longmemeval_focused_evaluated.csv \
        --full-path ../../results/gpt_4_1_longmemeval_full_evaluated.csv \
        --model-name "GPT-4.1" \
        --output-path ../../results/gpt_4_1_longmemeval.png
    ```
    """
    )
    return


@app.cell
def _():
    import argparse
    import sys
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    return pd, plt


@app.cell
def _(pd, plt):
    def visualize_longmemeval_results(focused_filepath: str, full_filepath: str, model_name: str, output_path: str):
        focused_df = pd.read_csv(focused_filepath)
        full_df = pd.read_csv(full_filepath)

        focused_mean = focused_df['llm_judge_output'].mean()
        full_mean = full_df['llm_judge_output'].mean()

        focused_color = "#EB4026"
        full_color = "#3A76E5"

        plt.figure(figsize=(8, 6))

        bars = plt.bar(['Focused', 'Full'], [focused_mean, full_mean], color=[focused_color, full_color])
        plt.ylim(0, 1)
        plt.ylabel('Average Score')
        plt.title(f'LongMemEval Overall Performance - {model_name}')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    return (visualize_longmemeval_results,)


@app.cell
def _():
    focused_path = "results/llama_3_2_longmemeval_focused_evaluated.csv"
    full_path = "results/llama_3_2_longmemeval_full_evaluated.csv"
    # focused_path = "results/gpt_4_1_longmemeval_focused_evaluated.csv"
    # full_path = "results/gpt_4_1_longmemeval_full_evaluated.csv"
    model_name = "llama3.2"
    output_path = "results/llama_3_2_longmemeval.png"
    return focused_path, full_path, model_name, output_path


@app.cell
def _(
    focused_path,
    full_path,
    model_name,
    output_path,
    visualize_longmemeval_results,
):
    visualize_longmemeval_results(
        focused_filepath=focused_path,
        full_filepath=full_path,
        model_name=model_name,
        output_path=output_path
    )
    print(f"Visualization saved to: {output_path}")
    return


if __name__ == "__main__":
    app.run()
