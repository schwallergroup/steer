"""Evaluate a given reaction."""

import base64
import os


import asyncio
import pandas as pd
from dotenv import load_dotenv
# from openai import AsyncOpenAI
from prompts import  *
from steer.utils.rxnimg import get_rxn_img
from litellm import acompletion
import asyncio



async def run(smiles, PREFIX, SUFFIX):

    load_dotenv()

    b64img = get_rxn_img(smiles)
    if b64img is None:
        raise ValueError("Failed to retrieve the image.")
    
    from litellm import Router
    router = Router(
        [
            {
                "model_name": "gpt-4o", # model alias 
                "litellm_params": { # params for litellm completion/embedding call 
                    "model": "openai/gpt-4o", # actual model name
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            },
            {
                "model_name": "gpt-4o-mini", # model alias 
                "litellm_params": { # params for litellm completion/embedding call 
                    "model": "openai/gpt-4o-mini", # actual model name
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            },
            {
                "model_name": "claude-3-5-sonnet", # model alias 
                "litellm_params": { # params for litellm completion/embedding call 
                    "model": "claude-3-5-sonnet-20241022", # actual model name
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                }
            }
        ],
        # enable_pre_call_checks=True, # enables router rate limits for concurrent calls
        default_max_parallel_requests=2,
    )
    print(b64img)
    response = await router.acompletion(
        # api_base="http://liacpc17.epfl.ch:8080",
        model="claude-3-5-sonnet",
        # model="gpt-4o-mini",
        # model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PREFIX},
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/png;base64,{b64img}"
                        }
                    },
                    {"type": "text", "text": SUFFIX},
                ],
            },
        ],
    )

    # response = await acompletion(
    #     api_base="http://liacpc17.epfl.ch:8080",
    #     # model="claude-3-5-sonnet-20241022",
    #     model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": f"![](data:image/png;base64,{b64img})"},
    #     ],
    #     max_tokens=1024,
    # )

    return response.choices[0].message.content



def parse_score(response):
    return response.split("<score>")[1].split("</score>")[0]

from prompts.molecules.toxicity import PREFIX, SUFFIX

async def run_row(row):
    smi = row[1]["smiles"].split(">>")[0]
    ans = await run(smi, PREFIX, SUFFIX)
    return ans

async def main():

    smis = pd.read_csv("src/steer/benchmark/smiles.csv", header=None)
    labls = pd.read_csv("src/steer/benchmark/labels.csv", sep=";;", header=None)

    df = pd.concat([smis, labls], axis=1)
    df.columns = ["smiles", "id", "label", "comment"]
    df['cat'] = df['label'].apply(lambda x: 0 if "Unlikely" in x else 1)

    from time import time
    t0 = time()
    answers = await asyncio.gather(*[run_row(row) for row in df.iterrows()])
    print(time()-t0, df.shape[0])
    scores = [parse_score(ans) for ans in answers]

    df["answer"] = answers
    df["llm_score"] = scores
    df.to_csv("src/steer/benchmark/answers_3.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())