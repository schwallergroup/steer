"""Evaluate a full route against a query."""

import weave
import asyncio
import base64
import importlib
import os
from typing import Optional, Dict, List, Any
from io import BytesIO
from PIL.Image import Image

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

from steer.utils.rxnimg import get_rxn_img

from .llms import router
from .prompts import *
from synthegy.chem import FixedRetroReaction
from synthegy.reactiontree import ReactionTree



class LM(BaseModel):
    """LLM Heuristic for scoring reactions."""

    model: str = "gpt-4o"
    prefix: str = ""
    suffix: str = ""
    prompt: Optional[str] = None  # Path to the prompt module
    # cache: Dict | str = {}
    project_name: str = ""

    async def run(self, tree: ReactionTree, query: str):
        # First get list of smiles
        smiles = self.get_smiles(tree)

        imgs = [get_rxn_img(s) for s in smiles]
        response = await self.lmcall(imgs, query)
        # score = self._parse_score(response)
        return response
    
    @weave.op()
    async def lmcall(self, imgs: List[Image], query: str):
        """Run the LLM."""
        load_dotenv()

        # Create sequence of image prompts
        img_msgs = []
        for i, s in enumerate(imgs):

            buffered = BytesIO()
            s.save(buffered, format="PNG")
            b64img = base64.b64encode(buffered.getvalue()).decode()
            if b64img is None:
                raise ValueError("Failed to retrieve the image.")
            
            msg = [
                {"type": "text", "text": f"Reaction #{i+1}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64img}"
                    },
                },
            ]
            img_msgs.extend(msg)

        response = await router.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prefix.format(query=query)},
                        *img_msgs,
                        # {
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": f"data:image/png;base64,{b64img}"
                        #     },
                        # },
                        {"type": "text", "text": self.suffix},
                    ],
                },
            ],
        )
    
        # self.cache[smiles] = response.choices[0].message.content
        return response.choices[0].message.content

    @model_validator(mode="after")
    def load_prompts(self):
        if self.project_name:
            weave.init(self.project_name)
        if self.prompt is not None:
            module = importlib.import_module(self.prompt)
            self.prefix = module.prefix
            self.suffix = module.suffix

        # if isinstance(self.cache, str):
        #     self.cache = pd.read_csv(self.cache, header=None).to_dict()

        return self

    @staticmethod
    def _parse_score(response):
        try:
            return float(response.split("<score>")[1].split("</score>")[0])
        except:
            return 0 # Default score (min)

    # async def _run_row(self, row):
    #     smi = row[1]["smiles"].split(">>")[0]
    #     ans = await self.run(smi)
    #     return ans

    def get_smiles(self, tree: ReactionTree):
        """Get all smiles from a tree."""
        smiles = []
        for m in tree.graph.nodes():
            if isinstance(m, FixedRetroReaction):
                rsmi = m.metadata['mapped_reaction_smiles'].split('>>')
                rvsmi = f"{rsmi[1]}>>{rsmi[0]}"
                smiles.append(rvsmi)
        return smiles



async def main():
    # from steer.llm.prompts import toxicity

    # heur = Evaluator(
    #     model="gpt-4o", PREFIX=toxicity.prefix, SUFFIX=toxicity.suffix
    # )

    # smis = pd.read_csv("src/steer/benchmark/smiles.csv", header=None)
    # labls = pd.read_csv(
    #     "src/steer/benchmark/labels.csv", sep=";;", header=None
    # )

    # df = pd.concat([smis, labls], axis=1)
    # df.columns = ["smiles", "id", "label", "comment"]
    # df["cat"] = df["label"].apply(lambda x: 0 if "Unlikely" in x else 1)

    # df = df.iloc[:5]
    # from time import time

    # t0 = time()
    # answers = await asyncio.gather(
    #     *[heur._run_row(row) for row in df.iterrows()]
    # )
    # print(time() - t0, df.shape[0])
    # scores = [Evaluator._parse_score(ans) for ans in answers]

    # df["answer"] = answers
    # df["llm_score"] = scores
    # df.to_csv("src/steer/benchmark/answers_3.csv", index=False)
    return 0


if __name__ == "__main__":
    asyncio.run(main())
