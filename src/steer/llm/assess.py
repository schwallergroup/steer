"""Evaluate a given reaction."""

import asyncio
import base64
import os

import pandas as pd
from dotenv import load_dotenv
from llms import router
from prompts import *
from prompts.molecules.toxicity import PREFIX, SUFFIX
from pydantic import BaseModel

from steer.utils.rxnimg import get_rxn_img


class Heuristic(BaseModel):
    """LLM Heuristic for scoring reactions."""

    model: str = "gpt-4o"
    PREFIX: str = ""
    SUFFIX: str = ""

    async def run(self, smiles: str):
        """Run the LLM."""
        load_dotenv()

        b64img = get_rxn_img(smiles)
        if b64img is None:
            raise ValueError("Failed to retrieve the image.")

        response = await router.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PREFIX},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64img}"
                            },
                        },
                        {"type": "text", "text": self.SUFFIX},
                    ],
                },
            ],
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_score(response):
        return response.split("<score>")[1].split("</score>")[0]

    async def _run_row(self, row):
        smi = row[1]["smiles"].split(">>")[0]
        ans = await self.run(smi)
        return ans


async def main():
    heur = Heuristic(model="gpt-4o", PREFIX=PREFIX, SUFFIX=SUFFIX)

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
