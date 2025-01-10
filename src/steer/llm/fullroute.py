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

from steer.llm.llms import router
from steer.llm.prompts import *
from litellm import Timeout
from weave.trace.context.call_context import get_current_call
from steer.logger import setup_logger

logger = setup_logger(__name__)


class LM(BaseModel):
    """LLM Heuristic for scoring reactions."""

    model: str = "gpt-4o"
    prefix: str = ""
    suffix: str = ""
    intermed: str = ""
    prompt: Optional[str] = None  # Path to the prompt module
    # cache: Dict | str = {}
    project_name: str = ""

    async def run(self, rxn: str, list_rxns: List[str], query: Optional[str] = None, return_score: bool = False):
        # First get list of smiles
        rxnimg = get_rxn_img(rxn)

        smiles = list_rxns

        imgs = [get_rxn_img(s) for s in smiles]
        response = await self.lmcall(rxnimg, imgs, smiles[0])
        if return_score:
            score = self._parse_score(response)
            return score
        else:
            return response
    
    # @weave.op()
    async def lmcall(self, rxnimg: Image, imgs: List[Image], query: Optional[str]):
        """Run the LLM."""
        load_dotenv()


        buffered = BytesIO()
        rxnimg.save(buffered, format="PNG")
        b64img = base64.b64encode(buffered.getvalue()).decode()
        rxn_msg = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64img}"
                },
            },
            {"type": "text", "text": self.intermed},
        ]


        # Create sequence of image prompts
        img_msgs = []
        for i, s in enumerate(imgs):

            buffered = BytesIO()
            s.save(buffered, format="PNG")
            b64img = base64.b64encode(buffered.getvalue()).decode()
            if b64img is None:
                raise ValueError("Failed to retrieve the image.")
            
            msg = [
                {"type": "text", "text": f"Step #{i+1}"},
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
                            {"type": "text", "text": self.prefix},
                            *rxn_msg,
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
            # url=current_call.ui_url,
            score=self._parse_score({"response":response.choices[0].message.content}),
            query=query,
        )

    @model_validator(mode="after")
    def load_prompts(self):
        if self.project_name:
            weave.init(self.project_name)
        if self.prompt is not None:
            module = importlib.import_module(self.prompt)
            self.prefix = module.prefix
            self.suffix = module.suffix
            try:
                self.intermed = module.intermed
            except:
                self.intermed = ""

        # if isinstance(self.cache, str):
        #     self.cache = pd.read_csv(self.cache, header=None).to_dict()

        return self

    @staticmethod
    def _parse_score(response):
        try:
            return float(response['response'].split("<score>")[1].split("</score>")[0])
        except Exception as e:
            logger.error(f"Failed to parse score: {e}")
            return 0 # Default score (min)

    # async def _run_row(self, row):
    #     smi = row[1]["smiles"].split(">>")[0]
    #     ans = await self.run(smi)
    #     return ans



async def main():
    lm = LM(
        prompt="steer.llm.prompts.alphamol",
        model="claude-3-5-sonnet",
        project_name="steer-mechanism-test",
    )

    start_smiles = "C1CCCCC1=O.F"

    list_smiles = [ # All legal ionization + attacks from cyclohexanone + HF
        '[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C](=[O])[C-]([H])[H]',
        '[H][F].[H][C+]([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C-]([H])[H]',
        '[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C-]([H])[H]',
        '[H][F].[H][C+]([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C-]([H])[H]',
        '[H][F].[H][C-]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C+]=[O]',
        '[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C-]=[O]',
        '[H][C]1([H])[C-]([O+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][F]',
        '[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][F]',
        '[H+].[H][F].[H][C-]1[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H-].[H][F].[H][C+]1[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H+].[H][F].[H][C-]1[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H-].[H][F].[H][C+]1[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H+].[H][F].[H][C-]1[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]1([H])[H]',
        '[H-].[H][F].[H][C+]1[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]1([H])[H]',
        '[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[F+].[H-].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
    ]

    correct_path = [
        "C1CCCCC1=O.F",
        '[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[F-].[H+].[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H+].[H][C]1([H])[C]([F])([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
    ]

    result = await lm.run(
        [f"{correct_path[i]}>>{correct_path[i+1]}" for i in range(len(correct_path)-1)],
        query="Please give me a mechanism awoiding too strained cycles, or atoms of charges higher than 1 or -1.",
        return_score=True,
    )
    print(result)
    return result

if __name__ == "__main__":
    asyncio.run(main())
