"""Evaluate a full route against a query."""

import asyncio
import base64
import importlib
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
import weave
from dotenv import load_dotenv
from litellm import Timeout
from PIL.Image import Image
from pydantic import BaseModel, model_validator
from weave.trace.context.call_context import get_current_call

from steer.llm.llm_router import router
from steer.llm.prompts import *
from steer.logger import setup_logger
from steer.utils.rxnimg import get_manual_rxn_img, get_rxn_img

logger = setup_logger(__name__)


class LM(BaseModel):
    """LLM Heuristic for scoring reactions."""

    model: str = "gpt-4o"
    prefix: str = ""
    suffix: str = ""
    intermed: str = ""
    prompt: Optional[str] = None  # Path to the prompt module
    project_name: str = ""

    async def run(
        self,
        rxn: str,
        list_rxns: List[str],
        query: Optional[str] = None,
        return_score: bool = False,
    ):
        # First get list of smiles
        imgf = get_manual_rxn_img
        # imgf = get_rxn_img

        rxnimg = imgf(rxn)

        smiles = list_rxns

        imgs = [imgf(s) for s in smiles]
        response = await self.lmcall(rxnimg, imgs, smiles[0])
        if return_score:
            score = self._parse_score(response)
            return score
        else:
            return response

    # @weave.op()
    async def lmcall(
        self, rxnimg: Image, imgs: List[Image], query: Optional[str]
    ):
        """Run the LLM."""
        load_dotenv()

        buffered = BytesIO()
        rxnimg.save(buffered, format="PNG")
        b64img = base64.b64encode(buffered.getvalue()).decode()
        rxn_msg = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64img}"},
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
                    "image_url": {"url": f"data:image/png;base64,{b64img}"},
                },
            ]
            img_msgs.extend(msg)

        try:
            responses = []
            for i in range(5):
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
                responses.append(response.choices[0].message.content)
        except Timeout as e:
            logger.error(f"API Timeout: {e}")
            return "<score>0</score>"

        current_call = get_current_call()
        scores = [self._parse_score({"response": r}) for r in responses]
        print(query, scores)
        return dict(
            response=response.choices[0].message.content,
            score=sum(scores) / len(scores),
            query=query,
            # url=current_call.ui_url,
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

        return self

    @staticmethod
    def _parse_score(response):
        try:
            return float(
                response["response"].split("<score>")[1].split("</score>")[0]
            )
        except Exception as e:
            logger.error(f"Failed to parse score: {e}")
            return 0  # Default score (min)


async def main():
    lm = LM(
        prompt="steer.llm.prompts.alphamol",
        model="claude-3-5-sonnet",
        project_name="steer-mechanism-test",
    )

    start_smiles = "C1CCCCC1=O.F"

    list_smiles = [  # All legal ionization + attacks from cyclohexanone + HF
        "[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C](=[O])[C-]([H])[H]",
        "[H][F].[H][C+]([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C-]([H])[H]",
        "[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C-]([H])[H]",
        "[H][F].[H][C+]([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C-]([H])[H]",
        "[H][F].[H][C-]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C+]=[O]",
        "[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C-]=[O]",
        "[H][C]1([H])[C-]([O+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][F]",
        "[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][F]",
        "[H+].[H][F].[H][C-]1[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H-].[H][F].[H][C+]1[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H+].[H][F].[H][C-]1[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H-].[H][F].[H][C+]1[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H+].[H][F].[H][C-]1[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]1([H])[H]",
        "[H-].[H][F].[H][C+]1[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]1([H])[H]",
        "[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[F+].[H-].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    ]

    correct_path = [
        "C1CCCCC1=O.F",
        "[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[F-].[H+].[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H+].[H][C]1([H])[C]([F])([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    ]

    result = await lm.run(
        [
            f"{correct_path[i]}>>{correct_path[i+1]}"
            for i in range(len(correct_path) - 1)
        ],
        query="Please give me a mechanism awoiding too strained cycles, or atoms of charges higher than 1 or -1.",
        return_score=True,
    )
    print(result)
    return result


if __name__ == "__main__":
    asyncio.run(main())
