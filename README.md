# Evergreen Context Rot: 

This repository extends the results of Chroma's [Context Rot](https://research.trychroma.com/context-rot) to include newer, open-source models.
Based on Ollama, this project is structured as a editable notebook + read-only app, powered by Marimo.

## Motivation

Large Language Models are typically presumed to process context uniformly—that is, the model should handle the 10,000th token just as reliably as the 100th. However, in practice, this assumption does not hold. Model performance varies significantly as input length changes, even on simple tasks.

Increasing input tokens impacts LLM performance.

## Experiments

This project focuses on the [LongMemEval](https://arxiv.org/abs/2410.10813) task.

LLM-driven systems have integrated memory components to track user-assistant chat histories, enabling more accurate and personalized responses. However, their long-term memory capabilities in sustained interactions remain underexplored.

Designed to evaluate five core long-term memory abilities of chat assistants: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. 

Presents a significant challenge to existing long-term memory systems, with commercial chat assistants and long-context LLMs showing a 30% accuracy drop on memorizing information across sustained interactions.

## Data

Datasets can be found here [here](https://drive.google.com/drive/folders/1FuOysriSotnYasJUbZJzn31SWt85_3yf?usp=drive_link).

Any custom datasets should be added to the `data/` directory.

## Quick Start

### On local

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `marimo edit notebook.py`

### On Enverge Lab

1. Open this repo: [![Launch in Enverge.ai](https://lab.enverge.ai/assets/enverge-shield.svg)](http://lab.enverge.ai/open?repo=git@github.com:Enverge-Labs/evergreen-context-rot.git&notebook=notebook.py)


## Log

- 2025-09-24: Initial release.

## Notes

1. The dataset uses natural language heavily. This could apply to other domains: programming, maths, etc. Know of a good programming dataset? Let me know! 