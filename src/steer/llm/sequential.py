"""Evaluate a full route against a query."""

import asyncio
import base64
import importlib
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import weave  # type: ignore
from aizynthfinder.chem import FixedRetroReaction  # type: ignore
from aizynthfinder.reactiontree import ReactionTree  # type: ignore
from dotenv import load_dotenv  # type: ignore
from PIL.Image import Image
from pydantic import BaseModel, model_validator  # type: ignore
from weave.trace.context.call_context import get_current_call  # type: ignore

from steer.logger import setup_logger
from steer.utils.rxnimg import get_rxn_img

from .llm_router import router
from .prompts import *

logger = setup_logger(__name__)


class LM(BaseModel):
    """LLM Heuristic for scoring reactions."""

    model: str = "gpt-4o"
    vision: bool = False
    prefix: str = ""
    suffix: str = ""
    prompt: Optional[str] = None  # Path to the prompt module
    project_name: str = ""

    async def run(self, tree: ReactionTree, query: str, task: Any):
        """Get smiles and run LLM."""

        if self.model == "random":
            response = dict(
                response=f"<score>{np.random.choice(np.arange(1,11))}</score>",
                url="",
            )
        else:
            rxn_msgs = self.make_msg_sequence(tree)
            response = await self._run_llm(rxn_msgs, query, taskid=task.id)
        return response

    @weave.op()
    async def _run_llm(self, msgs, query, taskid=""):
        try:
            response = await router.acompletion(
                model=self.model,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prefix.format(query=query),
                            },
                            *msgs,
                            {"type": "text", "text": self.suffix},
                        ],
                    },
                ],
            )
            response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"{e}")
            response = "<score>-1</score>"

        current_call = get_current_call()
        return dict(
            response=response,
            url=current_call.ui_url or "-",
        )

    def make_msg_sequence(self, tree: ReactionTree):
        rxns = self.get_smiles_with_depth(tree)

        msgs = []
        for i, s in enumerate(rxns):
            depth, smi = s
            if self.vision:
                inp = self._get_img_msg(smi)
            else:
                inp = self._get_txt_msg(smi)

            msg = [
                {"type": "text", "text": f"Reaction #{i+1}. Depth: {depth}"},
                inp,
            ]
            msgs.extend(msg)
        return msgs

    def _get_txt_msg(self, smi):
        """Get text message."""
        return {"type": "text", "text": f"{smi}"}

    def _get_img_msg(self, smi):
        """Get image message."""

        img = get_rxn_img(smi)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64img = base64.b64encode(buffered.getvalue()).decode()
        if b64img is None:
            raise ValueError("Failed to retrieve the image.")

        msg = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64img}"},
        }
        return msg

    async def run_single_route(self, task, d):
        result = await self.run(ReactionTree.from_dict(d), task.prompt, task)
        d["lmdata"] = dict(
            query=task.prompt,
            response=result["response"],
            weave_url=result["url"],
            routescore=self._parse_score(result["response"]),
        )
        return d

    async def run_single_task(self, task, data, nroutes=10):
        result = await asyncio.gather(
            *[self.run_single_route(task, d) for d in data[:nroutes]]
        )
        return result

    @model_validator(mode="after")
    def load_prompts(self):
        load_dotenv()
        if self.project_name:
            weave.init(self.project_name)
        if self.prompt is not None:
            module = importlib.import_module(self.prompt)
            self.prefix = module.prefix
            self.suffix = module.suffix
        return self

    @staticmethod
    def _parse_score(response):
        try:
            return float(response.split("<score>")[1].split("</score>")[0])
        except:
            return -1  # Default score (min)

    def get_smiles(self, tree: ReactionTree):
        """Get all smiles from a tree."""
        smiles = []
        for m in tree.graph.nodes():
            if isinstance(m, FixedRetroReaction):
                rsmi = m.metadata["mapped_reaction_smiles"].split(">>")
                rvsmi = f"{rsmi[1]}>>{rsmi[0]}"
                smiles.append(rvsmi)
        return smiles

    def get_smiles_with_depth(self, tree: ReactionTree):
        """Get all smiles from a tree, with depth in tree."""
        smiles = []
        for m in tree.graph.nodes():
            if isinstance(m, FixedRetroReaction):
                rsmi = m.metadata["mapped_reaction_smiles"].split(">>")
                rvsmi = f"{rsmi[1]}>>{rsmi[0]}"

                # Get distance of node m from root
                depth = nx.shortest_path_length(
                    tree.graph, source=tree.root, target=m
                )
                depth = int(
                    (depth - 1) / 2
                )  # Correct for molecule nodes in between
                smiles.append((depth, rvsmi))
        return smiles


async def main():
    import json

    lm = LM(
        prompt="steer.llm.prompts.fullroute",
        model="claude-3-5-sonnet",
        project_name="steer-test",
    )

    with open("data/aizynth_output.json", "r") as f:
        data = json.load(f)[0]

    result = await lm.run(ReactionTree.from_dict(data), "Metal free synthesis")
    return result


if __name__ == "__main__":
    asyncio.run(main())
