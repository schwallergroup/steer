"""LLMs config."""

import os

import litellm
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
            "model_name": "claude-3-7",  # model alias
            "litellm_params": {
                "model": "claude-3-7-sonnet-20250219",  # actual model name
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                # "rpm": 40,
            },
        },
        {
            "model_name": "llama3.3-70b",  # model alias
            "litellm_params": {
                "model": "openrouter/meta-llama/llama-3.3-70b-instruct",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "llama3.3-8b",  # model alias
            "litellm_params": {
                "model": "openrouter/meta-llama/llama-3.3-8b-instruct:free",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "llama4-scout",  # model alias
            "litellm_params": {
                "model": "openrouter/meta-llama/llama-4-scout",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "llama4-maverick",  # model alias
            "litellm_params": {
                "model": "openrouter/meta-llama/llama-4-maverick",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "llama3-405b",  # model alias
            "litellm_params": {
                "model": "openrouter/meta-llama/llama-3.1-405b-instruct",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "qwen3-32b",  # model alias
            "litellm_params": {
                "model": "openrouter/qwen/qwen3-32b",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "deepseek-r1",  # model alias
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-r1-0528",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "deepseek-v3",  # model alias
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-chat-v3-0324",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "deepseek-r1-32b",  # model alias
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-r1-distill-qwen-32b",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "deepseek-r1-70b",  # model alias
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-r1-distill-llama-70b",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
            },
        },
        {
            "model_name": "deepseek-r1-8b",  # model alias
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-r1-0528-qwen3-8b",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "rpm": 2,
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
            "model_name": "qwq-32b",  # model alias
            "litellm_params": {
                "model": "openrouter/qwen/qwq-32b-preview",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "api_base": "https://openrouter.ai/api/v1",
                # "rpm": 40,
            },
        },
        {
            "model_name": "gemini-2-5-pro",  # model alias
            "litellm_params": {
                "model": "openrouter/google/gemini-2.5-pro-preview",  # actual model name
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "api_base": "https://openrouter.ai/api/v1",
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
