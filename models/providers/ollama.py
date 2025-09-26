import os
from openai import OpenAI
from typing import Any
from ..base_provider import BaseProvider


class OllamaProvider(BaseProvider):

    def process_single_prompt(
        self, prompt: str, model_name: str, max_output_tokens: int, index: int
    ) -> tuple[int, str]:
        # Some models are more verbose than others. Prompt-force them to be concise.
        prompt_ = (
            prompt + " Be concise. Return the shortest possible, meaningful answer."
        )
        response = self.client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_completion_tokens=max_output_tokens,
            messages=[{"role": "user", "content": prompt_}],
            stream=False,
        )

        if response.choices and len(response.choices) > 0:
            if response.choices[0].message.content == "":
                self.output(response)
            return index, response.choices[0].message.content
        else:
            return index, "ERROR_NO_CONTENT"

    def get_client(self) -> Any:
        return OpenAI(api_key="", base_url=os.getenv("LLAMA_BASE_URL"))
