"""LLMs config."""

import os

from dotenv import load_dotenv
from litellm import Router

load_dotenv()

router = Router(
    model_list=[
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
            "model_name": "gpt-4-turbo",  # model alias
            "litellm_params": {
                "model": "openai/gpt-4-turbo",  # actual model name
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "claude-3-5-sonnet",  # model alias
            "litellm_params": {
                "model": "claude-3-5-sonnet-20241022",  # actual model name
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                # "rpm": 40,
            },
        },
        {
            "model_name": "qwq-32b",  # model alias
            "litellm_params": {
                "model": "openrouter/qwen/qwq-32b-preview",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "api_base": "https://openrouter.ai/api/v1",
                # "rpm": 40,
            },
        },
        {
            "model_name": "deepseek-3",  # model alias
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-chat",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "api_base": "https://openrouter.ai/api/v1",
                # "rpm": 40,
            },
        },
        {
            "model_name": "deepseek-r1",  # model alias
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-r1",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "api_base": "https://openrouter.ai/api/v1",
                # "rpm": 40,
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
    # timeout=120,
    num_retries=3,
    retry_after=1,
    allowed_fails=1,
    cooldown_time=60,
    # cache_responses=True
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
