import marimo

__generated_with = "0.14.7"
app = marimo.App(
    width="medium",
    app_title="Evergreen Context Rot",
    css_file="style.css",
)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Evergreen Context Rot
    ## Deep-dive playground

    The longest conversations reveal the deepest truths about memory. 
    [Chroma's recent context rot research](https://research.trychroma.com/context-rot) shows how language models falter as input length grows. One question stands apart: **how do newer, open-source models perform?**

    This is more than a needle-in-haystack test. 
    This is different. 

    Using **LongMemEval**, we test sustained dialogue. Context accumulates like sediment. Earlier exchanges shape later understanding. Models face what humans do daily: maintaining coherent conversation across thousands of turns. 

    The results reveal the fundamental limits of artificial memory in real conversations. Memory fades.

    This interactive report expands the original research. Test how newer, open-source models perform. Hence **evergreen**.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path
    import pandas as pd

    return Path, mo, os, pd


@app.cell(hide_code=True)
def _():
    import torch

    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    return (device_name,)


@app.cell
def _():
    from constants import OLLAMA_MODELS, DEFAULT_PROMPT
    from formatting import replace_non_alphanumeric

    return DEFAULT_PROMPT, OLLAMA_MODELS, replace_non_alphanumeric


@app.cell
def _(mo):
    mo.vstack([mo.md("<br /><br />"), mo.icon("lucide:arrow-down")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    <br /><br />
    ## Experiment

    **LongMemEval** is a long-context benchmark for conversational question-answering. [Full paper can be found here](https://github.com/xiaowu0162/LongMemEval).

    Ideally, models receive only relevant parts to focus on reasoning. Adding irrelevant context forces them to identify relevance first. The model performs two tasks simultaneously.

    Given a chat history between user and LLM, the task is answering questions about parts of that history. See the tabs below for some examples of types of retrieval and context reasoning tasks.

    <br />
    """
    )
    return


@app.cell
def _(mo):
    mo.ui.tabs(
        {
            "knowledge-update": mo.vstack(
                [
                    mo.md("<br />"),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md(
                                "I'm doing a century ride soon, do you think my **road bike** is ready for that distance, considering I've done 2,000 miles on it? I've been using it with my other two bikes, **a mountain bike** and a **commuter bike**"
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md("I just got a new **hybrid bike** recently."),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.md("<br />"),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:circle-question-mark", size=32)),
                            mo.md("How many bikes do I currently own?"),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [mo.right(mo.icon("lucide:lightbulb", size=32)), mo.md("4")],
                        justify="start",
                        widths=[1, 9],
                    ),
                ]
            ),
            "temporal-reasoning": mo.vstack(
                [
                    mo.md("<br />"),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md(
                                "I went to behind-the-scenes of the Science Museum today with **a friend who's a chemistry professor**. [22.01]"
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md(
                                "I attended a guided tour at the Natural History Museum yesterday with my dad. [11.03]"
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md(
                                "I just learned a lot in a lecture at the History Museum about ancient civilizations this month.[18.04]"
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.md("<br />"),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:circle-question-mark", size=32)),
                            mo.md(
                                "How many months have passed since my last museum visit with a friend? [25.06]"
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:lightbulb", size=32)),
                            mo.md("5 months"),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                ]
            ),
            "multi-session": mo.vstack(
                [
                    mo.md("<br />"),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md(
                                "I'm looking to find a piano technician to service my **Korg B1**."
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md(
                                "I've been playing my **black Fender Stratocaster** electric quitar a lot lately..."
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md(
                                "I've had my acoustic guitar, a **Yamaha FG800**, for about 8 years."
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:user", size=32)),
                            mo.md(
                                "I'm thinking of selling my old drum set, a **5-piece Pearl Export**."
                            ),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.md("<br />"),
                    mo.hstack(
                        [
                            mo.right(mo.icon("lucide:circle-question-mark", size=32)),
                            mo.md("How many musical instruments do I currently own?"),
                        ],
                        justify="start",
                        widths=[1, 9],
                    ),
                    mo.hstack(
                        [mo.right(mo.icon("lucide:lightbulb", size=32)), mo.md("5")],
                        justify="start",
                        widths=[1, 9],
                    ),
                ]
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""<br /><br />""")
    return


@app.cell(hide_code=True)
def _(OLLAMA_MODELS, Path, mo, replace_non_alphanumeric):
    def generate_hero_images():
        size = 500
        placeholder_path = "images/placeholder.png"
        # Order is dictated by the list of models defined in another file. Not a problem for now.
        image_paths = [
            f"results/{replace_non_alphanumeric(model)}_longmemeval.png"
            for model in OLLAMA_MODELS
        ]
        existing_paths = [path for path in image_paths if Path(path).exists()]
        min_legth = 4
        # Pad list of existing images with the placeholder path
        existing_paths += [placeholder_path] * (min_legth - len(existing_paths))
        image_components = [mo.image(path, width=size) for path in existing_paths]
        return image_components

    return (generate_hero_images,)


@app.cell(hide_code=True)
def _(generate_hero_images, mo):
    get_hero_images, set_hero_images = mo.state(generate_hero_images())
    return get_hero_images, set_hero_images


@app.cell
def hero_images(get_hero_images, mo):
    mo.hstack(get_hero_images(), wrap=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The steps, tools, and visualisations below run a full experimental cycle.
    <br /><br />
    """
    )
    return


@app.cell(hide_code=True)
def step_1_intro(mo):
    mo.md(
        r"""
    <br /><br />
    ## Step 1. Run
    Model and dataset selection
    """
    )
    return


@app.cell(hide_code=True)
def _(OLLAMA_MODELS, mo, os):
    # Value taken from the terminal output
    ollama_endpoint = "http://localhost:11434/v1"
    os.environ["LLAMA_BASE_URL"] = ollama_endpoint

    selected_model_dropdown = mo.ui.dropdown(
        options=OLLAMA_MODELS,
        value=OLLAMA_MODELS[0],
        label="Choose your model:",
        searchable=True,
    )
    test_conditions_dropdown = mo.ui.dropdown(
        options=["focused", "full"], value="focused", label="Choose dataset:"
    )
    mo.vstack(
        [
            selected_model_dropdown,
            test_conditions_dropdown,
        ],
        align="start",
        justify="start",
    )
    return selected_model_dropdown, test_conditions_dropdown


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(label="Run", kind="success")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def _(
    replace_non_alphanumeric,
    selected_model_dropdown,
    test_conditions_dropdown,
):
    model_name = selected_model_dropdown.value
    model_name_slug = replace_non_alphanumeric(model_name)
    test_conditions = test_conditions_dropdown.value

    #
    # Remove when ready.
    #
    run_input_path = f"data/cleaned_longmemeval_s_{test_conditions}.csv"
    run_output_path = (
        f"results/{model_name_slug}_longmemeval_{test_conditions}_results.csv"
    )
    run_input_column = f"{test_conditions}_prompt"
    return (
        model_name,
        model_name_slug,
        run_input_column,
        run_input_path,
        run_output_path,
        test_conditions,
    )


@app.cell(hide_code=True)
def run_output(
    Path,
    mo,
    model_name,
    run_button,
    run_input_column,
    run_input_path,
    run_output_path,
    test_conditions,
):
    from models.providers.ollama import OllamaProvider
    from model_management import pull_model
    from models.output_interface import marimo_output

    if run_button.value:
        if Path(run_output_path).exists():
            run_message = f" ✅ {model_name} already ran on {test_conditions} dataset"
            run_subtitle = "Next, let's evaluate these results."
        else:
            # TODO: update pull to check if model is present first.
            pull_model(model_name, mo)

            provider = OllamaProvider(output=marimo_output(mo))
            provider.main(
                input_path=run_input_path,
                output_path=run_output_path,
                input_column=run_input_column,
                output_column="output",
                model_name=model_name,
                max_context_length=1_047_576,
                max_tokens_per_minute=2_000_000,
            )
            run_message = f"🎬 Finished generating results for {model_name} and {test_conditions} dataset"
            run_subtitle = "Next, let's evaluate these results."
    else:
        run_message = ""
        run_subtitle = ""
    return marimo_output, pull_model, run_message, run_subtitle


@app.cell(hide_code=True)
def _(mo, run_message, run_subtitle):
    run_has_result = run_message and run_subtitle
    run_result = mo.vstack(
        [
            mo.md(f"### {run_message}"),
            mo.md(run_subtitle),
        ]
    )

    run_result if run_has_result else mo.md("_run results will appear here_")
    return


@app.cell(hide_code=True)
def step_2_intro(mo):
    mo.md(
        r"""
    _
    ## Step 2. Evaluate
    Evaluate outcomes for each model and dataset combo.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    evaluate_button = mo.ui.run_button(label="Evaluate", kind="success")
    evaluate_button
    return (evaluate_button,)


@app.cell(hide_code=True)
def _(model_name_slug, test_conditions):
    evaluate_input_path = (
        f"results/{model_name_slug}_longmemeval_{test_conditions}_results.csv"
    )
    evaluate_output_path = (
        f"results/{model_name_slug}_longmemeval_{test_conditions}_evaluated.csv"
    )

    output_column = "output"
    question_column = "question"
    correct_answer_column = "answer"
    max_context_length = 1_047_576
    max_tokens_per_minute = 2_000_000
    return (
        correct_answer_column,
        evaluate_input_path,
        evaluate_output_path,
        max_context_length,
        max_tokens_per_minute,
        output_column,
        question_column,
    )


@app.cell(hide_code=True)
def _(
    DEFAULT_PROMPT,
    correct_answer_column,
    marimo_output,
    mo,
    model_name,
    output_column,
    question_column,
):
    from models.llm_judge import LLMJudge

    judge = LLMJudge(
        prompt=DEFAULT_PROMPT,
        model_name=model_name,
        output_column=output_column,
        question_column=question_column,
        correct_answer_column=correct_answer_column,
        output=marimo_output(mo),
    )
    return (judge,)


@app.cell(hide_code=True)
def _(
    Path,
    evaluate_button,
    evaluate_input_path,
    evaluate_output_path,
    judge,
    max_context_length,
    max_tokens_per_minute,
    mo,
    model_name,
    test_conditions,
):
    if evaluate_button.value:
        mo.md(rf"""### Evaluating {evaluate_input_path}""")
        if Path(evaluate_input_path).exists():
            # TODO: make this function output to mo.output.
            judge.evaluate(
                input_path=evaluate_input_path,
                output_path=evaluate_output_path,
                max_context_length=max_context_length,
                max_tokens_per_minute=max_tokens_per_minute,
            )
            evaluate_message = (
                f"✅ Done evaluating **{model_name}** on **{test_conditions}** dataset"
            )
            evaluate_subtitle = "Next, let's visualise the findings. See below."
        else:
            evaluate_message = f"❌ Input file missing. Please run the previous step."
            evaluate_subtitle = ""
    else:
        evaluate_message = ""
        evaluate_subtitle = ""
    return evaluate_message, evaluate_subtitle


@app.cell(hide_code=True)
def _(evaluate_message, evaluate_subtitle, mo):
    evaluate_has_result = evaluate_message and evaluate_subtitle
    evaluate_result = mo.vstack(
        [
            mo.md(f"### {evaluate_message}"),
            mo.md(evaluate_subtitle),
        ]
    )

    (
        evaluate_result
        if evaluate_has_result
        else mo.md("_evaluation results will appear here_")
    )
    return


@app.cell(hide_code=True)
def step_3_intro(mo):
    mo.md(
        r"""
    _
    ## Step 3. Visualize
    See comparison charts of the outcomes.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    visualize_button = mo.ui.run_button(label="Visualise", kind="success")
    visualize_button
    return (visualize_button,)


@app.cell(hide_code=True)
def _(Path, mo, model_name, model_name_slug, visualize_button):
    from visualizing import visualize_longmemeval_results

    focused_path = f"results/{model_name_slug}_longmemeval_focused_evaluated.csv"
    full_path = f"results/{model_name_slug}_longmemeval_full_evaluated.csv"
    output_path = f"results/{model_name_slug}_longmemeval.png"

    missing_evaluations_message = None
    if not Path(focused_path).exists():
        missing_evaluations_message = f"The **focused** evaluation is missing. Please run the previous steps. Missing file: {focused_path}"
    if not Path(full_path).exists():
        missing_evaluations_message = f"The **full** evaluation is missing. Please run the previous steps. Missing file: {full_path}"

    visualisation_generated = False
    if visualize_button.value:
        if missing_evaluations_message is None:
            with mo.status.spinner(title="Loading...") as _spinner:
                visualize_longmemeval_results(
                    focused_filepath=focused_path,
                    full_filepath=full_path,
                    model_name=model_name,
                    output_path=output_path,
                )
                visualisation_outcome_message = f"📈 Chart for **{model_name_slug}** is done. [View it here](#experiment)"
                visualisation_generated = True
        else:
            visualisation_outcome_message = missing_evaluations_message
    else:
        visualisation_outcome_message = ""
    return visualisation_generated, visualisation_outcome_message


@app.cell
def _(mo, visualisation_outcome_message):
    visualisation_has_result = visualisation_outcome_message
    visualisation_result = mo.vstack(
        [
            mo.md(f"### {visualisation_outcome_message}"),
        ]
    )

    (
        visualisation_result
        if visualisation_has_result
        else mo.md("_visualisation results will appear here_")
    )
    return


@app.cell(hide_code=True)
def _(generate_hero_images, set_hero_images, visualisation_generated):
    if visualisation_generated:
        set_hero_images(generate_hero_images())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    <br /><br />
    ## FAQ
    """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "1. How can I run this on multiple models to compare results?": mo.md(
                "There's a constant named `OLLAMA_MODELS` in the `constants.py` file. You can add more models to the list and run the experiment again. This project uses Ollama to pull and run the models on local GPUs."
            ),
            "2. Are there any other techniques similar to LongMemEval to consider?": mo.md(
                "Several variations of Needle in a Haystack exist, including one with a [semantic match](https://aclanthology.org/2025.wraicogs-1.2/). The [Repeated Words](https://arxiv.org/html/2504.14218v1) task tests model performance on replicating repeated word sequences. The original report expands on these."
            ),
            "3. How can I add my own datasets?": mo.md(
                "You can add your own datasets to the `data/` folder and run the experiment again. The datasets should be in the same format as the `cleaned_longmemeval_s_focused.csv` and `cleaned_longmemeval_s_full.csv` files."
            ),
            "4. What's the best person to contact for more info in this area?": mo.md(
                "You can contact [Tudor](mailto:tudor@enverge.ai) for more info."
            ),
            "5. How can I contribute to this project?": mo.md(
                "You can contribute to this project by [forking the repository](https://github.com/Enverge-Labs/context-rot) and submitting a pull request."
            ),
            "6. How are the models run inside this notebook?": mo.md(
                "The notebook is GPU-native. It runs in an environment which requires no additional setup. A GPU is configured for both read-only and editable versions."
            ),
            "7. Can this run on closed-source models?": mo.md(
                "No. The original Context Rot paper and Chroma's report can. See their source code."
            ),
            "8. Is this open-source and can I see how this is computed?": mo.md(
                "Yes, this project is open-source and you can see the code [here](https://github.com/Enverge-Labs/context-rot). The code is organized into multiple notebooks and Python modules which store common components. The notebooks are built with [Marimo](https://marimo.io/)."
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.vstack([mo.md("<br /><br />"), mo.icon("lucide:arrow-down")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br /><br />
    ## Conclusion
    Our experiments show LLMs do not maintain consistent performance across input lengths. Even on simple tasks like non-lexical retrieval or text replication, performance becomes less uniform as input length grows.

    Our results highlight the need for rigorous long-context evaluation beyond current benchmarks. Context engineering matters. Whether relevant information is present in a model's context is not everything. How that information is presented matters more. Even the most capable models are sensitive to this. Effective context engineering is essential for reliable performance.

    See the source [code on GitHub](https://github.com/enverge-ai/context-rot).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br /><br />
    ## Annexes
    """
    )
    return


@app.cell(hide_code=True)
def _(OLLAMA_MODELS, mo):
    view_data_dropdown = mo.ui.dropdown(
        options=["focused", "full"], value="focused", label="Choose dataset:"
    )
    view_data_button = mo.ui.run_button(label="View", kind="success")

    reset_all_button = mo.ui.run_button(
        label="Reset all results and evaluations", kind="danger"
    )
    reset_model_dropdown = mo.ui.dropdown(
        options=OLLAMA_MODELS,
        value=OLLAMA_MODELS[0],
        label="Model results to reset:",
    )
    reset_test_conditions_dropdown = mo.ui.dropdown(
        options=["focused", "full"], value="focused", label="Dataset to reset:"
    )
    reset_selection_button = mo.ui.run_button(label="Reset", kind="danger")

    pull_models_button = mo.ui.run_button(label="Pull model", kind="success")
    pull_model_dropdown = mo.ui.dropdown(
        options=OLLAMA_MODELS,
        value=OLLAMA_MODELS[0],
        label="Model to pull:",
    )
    return (
        pull_model_dropdown,
        pull_models_button,
        reset_all_button,
        reset_model_dropdown,
        reset_selection_button,
        reset_test_conditions_dropdown,
        view_data_button,
        view_data_dropdown,
    )


@app.cell(hide_code=True)
def _(
    Path,
    generate_hero_images,
    mo,
    replace_non_alphanumeric,
    reset_model_dropdown,
    reset_test_conditions_dropdown,
    set_hero_images,
):
    def reset_selection():
        model_name_slug_to_reset = replace_non_alphanumeric(reset_model_dropdown.value)

        results_file_path = f"results/{model_name_slug_to_reset}_longmemeval_{reset_test_conditions_dropdown.value}_results.csv"
        evaluated_file_path = f"results/{model_name_slug_to_reset}_longmemeval_{reset_test_conditions_dropdown.value}_evaluated.csv"
        visualisation_path = f"results/{model_name_slug_to_reset}_longmemeval.png"
        Path(results_file_path).unlink(missing_ok=True)
        Path(evaluated_file_path).unlink(missing_ok=True)
        Path(visualisation_path).unlink(missing_ok=True)

        set_hero_images(generate_hero_images())

        return mo.md(
            f"""
        **{results_file_path}** deleted

        **{evaluated_file_path}** deleted

        **{visualisation_path}** deleted
        """
        )

    return (reset_selection,)


@app.cell(hide_code=True)
def _(mo, pull_model, pull_model_dropdown):
    def delayed_pull_model():
        pull_model(pull_model_dropdown.value, mo)
        return mo.md("")

    return


@app.cell
def _(
    mo,
    pd,
    pull_model_dropdown,
    pull_models_button,
    reset_all_button,
    reset_model_dropdown,
    reset_selection,
    reset_selection_button,
    reset_test_conditions_dropdown,
    view_data_button,
    view_data_dropdown,
):
    annex1_content = mo.vstack(
        [
            mo.hstack(
                [
                    view_data_dropdown,
                    view_data_button,
                ],
                justify="start",
            ),
            (
                mo.lazy(
                    pd.read_csv(
                        f"data/cleaned_longmemeval_s_{view_data_dropdown.value}.csv"
                    ),
                    show_loading_indicator=True,
                )
                if view_data_button.value
                else mo.md("")
            ),
        ],
    )

    danger_zone_output = mo.md("foo")

    annex2_content = mo.callout(
        mo.vstack(
            [
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                reset_model_dropdown,
                                reset_test_conditions_dropdown,
                                reset_selection_button,
                            ]
                        ),
                        mo.vstack(
                            [
                                mo.center(mo.md("**OR**")),
                            ]
                        ),
                        mo.vstack(
                            [
                                reset_all_button,
                            ]
                        ),
                    ]
                ),
                (
                    mo.lazy(reset_selection(), show_loading_indicator=True)
                    if reset_selection_button.value
                    else mo.md("")
                ),
            ]
        ),
        kind="danger",
    )

    annex3_content = mo.vstack(
        [
            mo.md("### Explicitly pull a model from Ollama"),
            mo.md("Only needed if **not** following the steps above."),
            mo.hstack(
                [pull_model_dropdown, pull_models_button],
                align="start",
                justify="start",
            ),
        ]
    )

    mo.accordion(
        {
            "Annex 1 - View datasets": annex1_content,
            "Annex 2 - Danger zone": annex2_content,
            "Annex 3 - Utils": annex3_content,
        }
    )
    return


@app.cell(hide_code=True)
def _(mo, pull_model, pull_model_dropdown, pull_models_button):
    if pull_models_button.value:
        pull_model(pull_model_dropdown.value, mo)
    else:
        mo.output.append(mo.md(""))
    return


@app.cell
def _(mo, reset_all_button):
    (
        mo.callout(
            mo.md(
                "⚠️⚠️⚠️ Reset all deletes all results files and charts. Enabled only in edit-mode."
            ),
            kind="warn",
        )
        if reset_all_button.value
        else mo.md("")
    )
    return


@app.cell
def _(mo):
    mo.vstack([mo.md("<br /><br />"), mo.icon("lucide:leaf")])

    return


@app.cell
def _(device_name, mo):
    mo.md(
        rf"""
    <br /><br />
    ## Notes

    1. Original LongMemEval paper, with more thorough explanations: https://github.com/xiaowu0162/LongMemEval

    2. The dataset uses natural language heavily. This could apply to other domains: programming, maths, etc.
    Know of a good programming dataset? Let me know! 

    3. All experiments run on a **{device_name}** on [Enverge Lab](https://enverge.ai/enverge-labs).

    Want to know more about any of this? [Let's chat](mailto:tudor@enverge.ai)
    """
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.icon("lucide:leaf"),
            mo.icon("lucide:leaf"),
            mo.icon("lucide:leaf", color="green"),
        ],
        justify="center",
    )

    return


if __name__ == "__main__":
    app.run()
