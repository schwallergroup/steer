# Here define the classes for the tasks

from typing import Callable, Literal, Optional

from eval_types import MultiRxnCond, RingBreakDepth, SpecificBondBreak
from pydantic import BaseModel, model_validator

EVAL_CLASSES = {
    "RingBreakDepth": RingBreakDepth,
    "SpecificBondBreak": SpecificBondBreak,
    "MultiRxnCond": MultiRxnCond,
}


class Task(BaseModel):
    id: str
    smiles: str
    prompt: str
    eval_type: Literal["RingBreakDepth", "SpecificBondBreak", "MultiRxnCond"]
    eval_config: dict
    evaluate: Callable

    @model_validator(mode="after")
    def setup_eval_class(self):
        self.evaluate = EVAL_CLASSES[self.eval_type](self.eval_config)
        return self

    @classmethod
    def load_from_json(cls, data):
        if data.get("eval_type") is not None:
            return cls(**data)
        return None


def load_default_tasks(dir):
    import json

    tasks = []
    with open(f"{dir}/prompt_specs.json", "r") as f:
        data = json.load(f)
        for task in data:
            t = Task.load_from_json(task)
            if t is not None:
                tasks.append(t)
    return tasks


if __name__ == "__main__":
    import os

    path = os.path.dirname(os.path.abspath(__file__))
    RESULTS_PATH = os.path.join(path, "../../../../synthegy/steer/")
    PROMPT_TYPE = "fullroute"

    import json

    with open(
        f"{RESULTS_PATH}/fullroute_no_feasibility/output_2024-18-12_160055_ca06156bee8f14dcf0bd7e14f68eddcc.json",
        "r",
    ) as f:
        data = json.load(f)

    task = Task(
        id="Form_piperidine_and_oxoisoindolinone_rings_in_the_synthesis._Get_the_piperidine-2,6-dione_from_comme",
        smiles="CC1CCCCN1C(=O)C2=CC=CC=C2",
        prompt="Form piperidine and oxoisoindolinone rings in the synthesis. Get the piperidine-2,6-dione from commercial sources.",
        eval_type="MultiRxnCond",
        config={
            "allow_piperidine": True,
            "allow_oxoisoindolinone": True,
            "allow_piperidine26diox": False,
        },
    )

    gt_score, lmscore = task.evaluate(data)
    print(gt_score, lmscore)
