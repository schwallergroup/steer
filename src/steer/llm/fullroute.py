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
from litellm import Timeout
from synthegy.chem import FixedRetroReaction
from synthegy.reactiontree import ReactionTree
from weave.trace.context.call_context import get_current_call
from steer.logger import setup_logger

logger = setup_logger(__name__)


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

        try:
            response = await router.acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prefix.format(query=query)},
                            *img_msgs,
                            {"type": "text", "text": self.suffix},
                        ],
                    },
                ],
            )
        except Timeout as e:
            logger.error(f"API Timeout: {e}")
            return "<score>0</score>"
    
        current_call = get_current_call()
        # self.cache[smiles] = response.choices[0].message.content
        return dict(
            response=response.choices[0].message.content,
            url=current_call.ui_url,
        )

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
    import json

    lm = LM(
        prompt="steer.llm.prompts.fullroute",
        model="gpt-4o",
        project_name="steer-test",
    )

    with open("data/aizynth_output.json", "r") as f:
        data = json.load(f)[0]
    
    result = await lm.run(ReactionTree.from_dict(data), "Metal free synthesis")
    return result


if __name__ == "__main__":
    asyncio.run(main())
