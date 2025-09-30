import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from ..base_provider import BaseProvider


class OllamaResponse(BaseModel):
    answer: bool


class OllamaProvider(BaseProvider):

    def process_single_prompt(
        self, prompt: str, model_name: str, max_output_tokens: int, index: int, judge_mode: bool = False
    ) -> tuple[int, str]:

        # Introduce a stricter "judge mode" because the output must be binary.
        if judge_mode:
            # WARNING: here be dragons
            # In an attempt to make output more robust and consistent, I'm using JSON schema, but string interpolation
            # has some caveats. The code below is a hacky workaround to get the schema into the prompt.
            schema = OllamaResponse.model_json_schema()

            system_prompt = """
            This is a binary classification task. The possible answers are only "True" or "False". 

            The output JSON should have this schema: {schema}
            """.format(schema=schema)
            system_prompt += """
            EXAMPLE JSON OUTPUT:
            {"answer": true}
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            completion_kwargs = {
                "response_format": {
                    "type": "json_object"
                }
            }            
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
            completion_kwargs = {}

        response = self.client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_completion_tokens=max_output_tokens,
            messages=messages,
            stream=False,
            **completion_kwargs
        )

        if response.choices and len(response.choices) > 0:
            if response.choices[0].message.content == "":
                self.output(response)
            return index, response.choices[0].message.content
        else:
            return index, "ERROR_NO_CONTENT"

    def get_client(self) -> Any:
        return OpenAI(api_key="", base_url=os.getenv("LLAMA_BASE_URL"))
