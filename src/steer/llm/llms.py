"""LLMs config."""

import os

from litellm import Router

router = Router(
    model_list = [
        {
            "model_name": "gpt-4o",  # model alias
            "litellm_params": {
                "model": "openai/gpt-4o",  # actual model name
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-4o-mini",  # model alias
            "litellm_params": {
                "model": "openai/gpt-4o-mini",  # actual model name
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "claude-3-5-sonnet",  # model alias
            "litellm_params": {
                "model": "claude-3-5-sonnet-20241022",  # actual model name
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
        },
        {
            "model_name": "Llama-3.1-11B-vision",  # model alias
            "litellm_params": {
                "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",  # actual model name
                "api_key": "-",
                "api_base": "http://liacpc17.epfl.ch:8080",
            },
        },
    ],
    timeout=60,
    num_retries=1,
    retry_after=2,
    allowed_fails=1,
    # cooldown_time=60,
)

######### For llama, images are passed diferently ###########
# response = await acompletion(
#     api_base="http://liacpc17.epfl.ch:8080",
#     model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"![](data:image/png;base64,{b64img})"},
#     ],
#     max_tokens=1024,
# )
