# Models

We support Ollama models.

## File Structure

```
models/
├── README.md
├── base_provider.py         # Abstract base class for all providers
├── llm_judge.py             # LLM judge for evaluation
├── output_interface.py      # Output interface for all providers, for compatibility with Marimo
└── providers/
    ├── openai.py            # OpenAI provider implementation
    └── ollama.py            # Ollama provider implementation
```

## Environment Variables (optional)

- **OpenAI**: `OPENAI_API_KEY`

## Adding New Providers

To add a new provider, inherit from `BaseProvider` and implement:
- `process_single_prompt()`: Process a single prompt
- `get_client()`: Initialize API client