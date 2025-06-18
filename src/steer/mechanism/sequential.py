"""Score the next step in a mechanism proposal."""

import asyncio
import base64
import importlib
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
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
    vision: bool = False
    prefix: str = ""
    intermed: str = ""
    suffix: str = ""
    prompt: Optional[str] = None  # Path to the prompt module
    prompt_needs_expert_description: bool = False
    expert_description: Optional[str] = None
    project_name: str = ""

    assert (not prompt_needs_expert_description) or ((task is not None and task.expert_description is not None) or (task is None and expert_description != None))

    async def run(
        self,
        rxn: str,
        step: str,
        expert_description: str,
        history: Optional[List[str]] = None,
        task: Any = None,
    ):
        """Get smiles and run LLM."""

        if self.model == "random":
            response = dict(
                response=f"<score>{np.random.choice(np.arange(1,11))}</score>",
                url="",
            )
        else:
            msgs = self.make_msg_sequence(rxn, history)
            response = await self._run_llm(
                msgs, step, taskid=task.id if task else "", expert_description=expert_description
            )
        return response

    @weave.op()
    async def _run_llm(self, msgs, step, taskid="", expert_description=""):
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
                                "text": self.prefix if not self.prompt_needs_expert_description else self.prefix.format(expert_description=expert_description),
                            },
                            *msgs,
                            {
                                "type": "text",
                                "text": self.suffix.format(step=step),
                            },
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
            url=current_call.ui_url if current_call is not None else "-",
        )

    def make_msg_sequence(self, rxn: str, history: Optional[List[str]]):
        msgs = [self._get_msg(rxn)]
        if history is not None:
            msgs.append(
                {"type": "text", "text": self.intermed},
            )
            for i, s in enumerate(history):
                msg = [
                    {"type": "text", "text": f"Step #{i+1}:"},
                    self._get_msg(s),
                ]
                msgs.extend(msg)
        return msgs

    def _get_msg(self, smi):
        if self.vision:
            inp = self._get_img_msg(smi)
        else:
            inp = self._get_txt_msg(smi)
        return inp

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
        prompt="steer.mechanism.prompts.preprint_prompt_last_step_plus_game",
        model="claude-3-5-sonnet",
        project_name="steer-mechanism-test",
    )

    start_smiles = "C1CCCCC1=O.F"
    correct_path = [
        "C1CCCCC1=O.F",
        "[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[F-].[H+].[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H+].[H][C]1([H])[C]([F])([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    ]

    rxns = [
        f"{correct_path[i]}>>{correct_path[i+1]}"
        for i in range(len(correct_path) - 1)
    ]

    runs = []
    for i, step in enumerate(correct_path[1:]):
        result = lm.run(
            rxn=f"{start_smiles}>>{correct_path[-1]}",
            history=rxns[:i],
            step=step,
        )
        runs.append(result)

    for result in await asyncio.gather(*runs):
        logger.info(lm._parse_score(result))
    return result


if __name__ == "__main__":
    asyncio.run(main())
