"""LLMs config."""

import os

from dotenv import load_dotenv
from litellm import Router
import litellm

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
            "model_name": "claude-3-7",  # model alias
            "litellm_params": {
                "model": "claude-3-7-sonnet-20250219",  # actual model name
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                # "rpm": 40,
            },
        },
    ],
    # timeout=120,
    num_retries=10,
    retry_after=30,
    allowed_fails=10,
    cooldown_time=60,
    # cache_responses=True
)
