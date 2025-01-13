"""Evaluate a given reaction."""

import asyncio
import base64
import importlib
import os
from typing import Dict, Optional

import pandas as pd
import weave
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

from steer.utils.rxnimg import get_rxn_img

from .llm_router import router
from .prompts import *


class Heuristic(BaseModel):
    """LLM Heuristic for scoring reactions."""

    model: str = "gpt-4o"
    prefix: str = ""
    suffix: str = ""
    prompt: Optional[str] = None  # Path to the prompt module
    cache: Dict = {}
    project_name: str = ""

    @weave.op()
    async def run(self, smiles: str):
        """Run the LLM."""
        load_dotenv()
        if smiles in self.cache:
            return self.cache[smiles]

        b64img = get_rxn_img(smiles)
        if b64img is None:
            raise ValueError("Failed to retrieve the image.")

        response = await router.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prefix},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64img}"
                            },
                        },
                        {"type": "text", "text": self.suffix},
                    ],
                },
            ],
        )

        self.cache[smiles] = response.choices[0].message.content
        return response.choices[0].message.content

    @model_validator(mode="after")
    def load_prompts(self):
        if self.project_name:
            weave.init(self.project_name)
        if self.prompt is not None:
            module = importlib.import_module(self.prompt)
            self.prefix = module.prefix
            self.suffix = module.suffix

        if isinstance(self.cache, str):
            self.cache = pd.read_csv(self.cache, header=None).to_dict()

        return self

    @staticmethod
    def _parse_score(response):
        try:
            return float(response.split("<score>")[1].split("</score>")[0])
        except:
            return 10.0  # Default score max.

    async def _run_row(self, row):
        smi = row[1]["smiles"].split(">>")[0]
        ans = await self.run(smi)
        return ans


async def main():
    from steer.llm.prompts import toxicity

    heur = Heuristic(
        model="gpt-4o", PREFIX=toxicity.prefix, SUFFIX=toxicity.suffix
    )

    smis = pd.read_csv("src/steer/benchmark/smiles.csv", header=None)
    labls = pd.read_csv(
        "src/steer/benchmark/labels.csv", sep=";;", header=None
    )

    df = pd.concat([smis, labls], axis=1)
    df.columns = ["smiles", "id", "label", "comment"]
    df["cat"] = df["label"].apply(lambda x: 0 if "Unlikely" in x else 1)

    df = df.iloc[:5]
    from time import time

    t0 = time()
    answers = await asyncio.gather(
        *[heur._run_row(row) for row in df.iterrows()]
    )
    print(time() - t0, df.shape[0])
    scores = [Heuristic._parse_score(ans) for ans in answers]

    df["answer"] = answers
    df["llm_score"] = scores
    df.to_csv("src/steer/benchmark/answers_3.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
