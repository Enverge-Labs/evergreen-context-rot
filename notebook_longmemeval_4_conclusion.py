import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.image("results/gpt_4_1_longmemeval.png"),
            mo.image("results/llama_3_2_longmemeval.png"),
            mo.image("results/gpt_oss_20b_longmemeval.png")
        ]
    )
    return


if __name__ == "__main__":
    app.run()
