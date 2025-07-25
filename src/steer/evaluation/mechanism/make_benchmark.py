"""Make benchmark for the mechanism prediction task."""

tasks = [
    # [
    #     "C1CCCCC1=O.N",
    #     "[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][N]([H])[H]",
    #     "C1CCCCC1([O-])[NH3+]",
    #     "[H+].[H][N]([H])[C]1([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    #     "C1CCCCC1(O)N",
    # ],
    # [
    #     "C=O.C=[PH3]",
    #     "[H][C]([H])=[P]([H])([H])[H].[H][C+]([H])[O-]",
    #     "[H][C+]([H])[O-].[H][C-]([H])[P+]([H])([H])[H]",
    #     "[H][C]([H])([O-])[C]([H])([H])[P+]([H])([H])[H]",
    #     "O1CC[PH3]1",
    #     "[H][C+]([H])[C]([H])([H])[P]([H])([H])([H])[O-]",
    #     "[H][C+]([H])[C-]([H])[H].[H][P+]([H])([H])[O-]",
    #     "[H][C]([H])=[C]([H])[H].[H][P+]([H])([H])[O-]",
    #     "O=[PH3].C=C",
    # ],
    # [
    #     "O1CC[PH3]1",
    #     "[H][C+]([H])[C]([H])([H])[P]([H])([H])([H])[O-]",
    #     "[H][C+]([H])[C-]([H])[H].[H][P+]([H])([H])[O-]",
    #     "[H][C]([H])=[C]([H])[H].[H][P+]([H])([H])[O-]",
    #     "O=[PH3].C=C",
    # ],
    # [
    #     "CC([H])=O.CO.[H+]",
    #     "[H+].[H+].[H][C](=[O])[C]([H])([H])[H].[H][C]([H])([H])[O-]",
    #     "[H+].[H+].[H][C]([H])([H])[O-].[H][C+]([O-])[C]([H])([H])[H]",
    #     "[H+].[H+].[H][C]([H])([H])[O][C]([H])([O-])[C]([H])([H])[H]",
    #     "CC(OC)O.[H+]",
    # ],
    # [
    #     "OC(C)OC.CO.[H+]",
    #     "[OH2+]C(C)OC.CO",
    #     "[H][O][C]([H])([H])[H].[H][O][H].[H][C+]([O][C]([H])([H])[H])[C]([H])([H])[H]",
    #     "[H][O][H].[H][O+]([C]([H])([H])[H])[C]([H])([O][C]([H])([H])[H])[C]([H])([H])[H]",
    #     "[H+].[H][C]([H])([H])[O][C]([H])([O][C]([H])([H])[H])[C]([H])([H])[H].[H][O][H]",
    #     "CC(OC)(OC)[H].[OH3+]",
    # ],
    # [
    #     "C=C.[O-][O+]=O",
    #     "[H][C]([H])=[C]([H])[H].[O-][O][O+]",
    #     "[H][C+]([H])[C-]([H])[H].[O+][O][O-]",
    #     "[H][C+]([H])[C]([H])([H])[O][O][O-]",
    #     "C1COOO1",
    # ],
    # [
    #     "C1COOO1",
    #     "[H][C+]([H])[O][O][O][C-]([H])[H]",
    #     "[H][C]([H])=[O+][O][O][C-]([H])[H]",
    #     "[H][C]([H])=[O+][O-].[H][C-]([H])[O+]",
    #     "C=[O+][O-].C=O",
    #     "[H][C]([H])=[O].[H][C+]([H])[O][O-]",
    #     "[H][C]([H])=[O+][C]([H])([H])[O][O-]",
    #     "[H][C+]([H])[O][C]([H])([H])[O][O-]",
    #     "C1OCOO1",
    # ],
    # [
    #     "C=O.CP(C)(C)=C",
    #     "C=O.C[P+](C)(C)[CH2-]",
    #     "[H][C+]([H])[O-].[H][C-]([H])[P+]([C]([H])([H])[H])([C]([H])([H])[H])[C]([H])([H])[H]",
    #     "[H][C]([H])([O-])[C]([H])([H])[P+]([C]([H])([H])[H])([C]([H])([H])[H])[C]([H])([H])[H]",
    #     "CP1(C)(CCO1)C",
    #     "[H][C-]([H])[C]([H])([H])[O][P+]([C]([H])([H])[H])([C]([H])([H])[H])[C]([H])([H])[H]",
    #     "[H][C]([H])([H])[P+]([O-])([C]([H])([H])[H])[C]([H])([H])[H].[H][C+]([H])[C-]([H])[H]",
    #     "[H][C]([H])([H])[P+]([O-])([C]([H])([H])[H])[C]([H])([H])[H].[H][C]([H])=[C]([H])[H]",
    #     "C=C.CP(C)(C)=O",
    # ],
    # [
    #     "CC(=O)C.[BH4-]",
    #     "[H][C]([H])([H])[C+]([O-])[C]([H])([H])[H].[H][B-]([H])([H])[H]",
    #     "[H-].[H][B]([H])[H].[H][C]([H])([H])[C+]([O-])[C]([H])([H])[H]",
    #     "CC([O-])C.B",
    # ],
    # [
    #     "CC(=O)C.[BH4-]",
    #     "[H][C]([H])([H])[C+]([O-])[C]([H])([H])[H].[H][B-]([H])([H])[H]",
    #     "[H-].[H][B]([H])[H].[H][C]([H])([H])[C+]([O-])[C]([H])([H])[H]",
    #     "CC([O-])C.B",
    #     "CC([O][BH3-])C",
    # ],
    # [
    #     "O=C1CCCCC1.[BH4-]",
    #     "[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][B-]([H])([H])[H]",
    #     "[H-].[H][B]([H])[H].[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    #     "[H][B]([H])[H].[H][C]1([H])[C]([H])([H])[C]([H])([H])[C]([H])([O-])[C]([H])([H])[C]1([H])[H]",
    #     "[BH3-]OC1CCCCC1",
    # ],
    # [
    #     "CC(=O)C.CC(=O)C.C[Mg]C",
    #     "[H][C]([H])([H])[C](=[O])[C]([H])([H])[H].[H][C]([H])([H])[C](=[O])[C]([H])([H])[H].[H][C]([H])([H])[Mg+].[H][C-]([H])[H]",
    #     "[H][C]([H])([H])[C](=[O])[C]([H])([H])[H].[H][C]([H])([H])[C+]([O-])[C]([H])([H])[H].[H][C]([H])([H])[Mg+].[H][C-]([H])[H]",
    #     "[H][C]([H])([H])[C](=[O])[C]([H])([H])[H].[H][C]([H])([H])[C]([O-])([C]([H])([H])[H])[C]([H])([H])[H].[H][C]([H])([H])[Mg+]",
    #     "CC(=O)C.C[Mg]OC(C)(C)C",
    #     "[H][C]([H])([H])[C](=[O])[C]([H])([H])[H].[H][C]([H])([H])[C]([O][Mg+])([C]([H])([H])[H])[C]([H])([H])[H].[H][C-]([H])[H]",
    #     "[H][C]([H])([H])[C]([O][Mg+])([C]([H])([H])[H])[C]([H])([H])[H].[H][C]([H])([H])[C+]([O-])[C]([H])([H])[H].[H][C-]([H])[H]",
    #     "[H][C]([H])([H])[C]([O][Mg+])([C]([H])([H])[H])[C]([H])([H])[H].[H][C]([H])([H])[C]([O-])([C]([H])([H])[H])[C]([H])([H])[H]",
    #     "CC([O][Mg]OC(C)(C)C)(C)C",
    # ],
    # [
    #     "C(=O)O.ClP(Cl)(Cl)(Cl)Cl",
    #     "[H][C](=[O])[O+]([H])[P-]([Cl])([Cl])([Cl])([Cl])[Cl]",
    #     "[H+].[H][C](=[O])[O][P-]([Cl])([Cl])([Cl])([Cl])[Cl]",
    #     "[H][O+]=[C]([H])[O][P-]([Cl])([Cl])([Cl])([Cl])[Cl]",
    #     "C(OP(Cl)(Cl)(Cl)Cl)=[OH+].[Cl-]",
    #     "[Cl-].[H][O+]=[C]([H])[O+]=[P-]([Cl])([Cl])([Cl])[Cl]",
    #     "[O]=[P-]([Cl])([Cl])([Cl])[Cl].[Cl-].[H][C+]=[O+][H]",
    #     "[O]=[P-]([Cl])([Cl])([Cl])[Cl].[H][O+]=[C]([H])[Cl]",
    #     "C(Cl)=[OH+].O=P(Cl)(Cl)Cl.[Cl-]",
    #     "[O]=[P]([Cl])([Cl])[Cl].[Cl-].[H+].[H][C](=[O])[Cl]",
    #     "C(=O)Cl.Cl.ClP(Cl)(Cl)=O",
    # ],
    # [
    #     "CC=O.CO.[H+]",
    #     "CC=[OH+].CO",
    #     "[H+].[H][C]([H])([H])[O-].[H][O+]=[C]([H])[C]([H])([H])[H]",
    #     "[H+].[H][C]([H])([H])[O-].[H][O][C+]([H])[C]([H])([H])[H]",
    #     "CC(O)(OC).[H+]",
    # ],
    # [
    #     "C=CCBr.C[O-]",
    #     "[Br-].[H][C]([H])([H])[O-].[H][C]([H])=[C]([H])[C+]([H])[H]",
    #     "C=CCOC.[Br-]",
    # ],
    [
        "[H+].[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]2=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]2[N]=[C]1[C]([H])([H])[H].[H][O][H]",
        "[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C]([C]([H])([H])[H])=[N+]2[H].[H][O][H]",
        "[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C]([H])([H])[H])[N]2[H].[H][O][H]",
        "[H+].[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C-]([H])[H])[N]2[H].[H][O][H]",
        "[H+].[H][C]1=[C]([H])[C]([C+]([H])[C-]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C-]([H])[H])[N]2[H].[H][O][H]",
        "[H][C]1=[C]([H])[C]([C+]([H])[C]([H])([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C-]([H])[H])[N]2[H].[H][O][H]",
        "[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C]([H])([H])[C]([H])([C]1=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]1[H])[C]([H])([H])[N+](=[O])[O-])[N]2[H].[H][O][H]",
        "[H+].[H][C]1=[C]([H])[C]([H])=[C]2[C]([H])=[C]([H])[C+]([C]([H])([H])[C]([H])([C]3=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]3[H])[C]([H])([H])[N+](=[O])[O-])[N-][C]2=[C]1[H].[H][O][H]",
        "[H+].[H][C]1=[C]([H])[C]2=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]2[N]=[C]1[C]([H])([H])[C]([H])([C]1=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]1[H])[C]([H])([H])[N+](=[O])[O-].[H][O][H]",
    ],
    [
        "[H][C]([H])([H])[O][C](=[O])[C]([H])([H])[C](=[O])[O][C]([H])([H])[H].[H][C]1=[C]([H])[C]([H])([H])[C]([H])([H])[C]1=[O].[H][O-]",
        "[H+].[H][C]1=[C]([H])[C]([H])([H])[C]([H])([H])[C]1=[O].[H][C-]([C](=[O])[O][C]([H])([H])[H])[C](=[O])[O][C]([H])([H])[H].[H][O-]",
        "[H][C]1=[C]([H])[C]([H])([H])[C]([H])([H])[C]1=[O].[H][O][H].[H][C-]([C](=[O])[O][C]([H])([H])[H])[C](=[O])[O][C]([H])([H])[H]",
        "[H][O][H].[H][C+]1[C-]([H])[C](=[O])[C]([H])([H])[C]1([H])[H].[H][C-]([C](=[O])[O][C]([H])([H])[H])[C](=[O])[O][C]([H])([H])[H]",
        "[H][O][H].[H][C-]1[C](=[O])[C]([H])([H])[C]([H])([H])[C]1([H])[C]([H])([C](=[O])[O][C]([H])([H])[H])[C](=[O])[O][C]([H])([H])[H]",
        "[H+].[H][C-]1[C](=[O])[C]([H])([H])[C]([H])([H])[C]1([H])[C]([H])([C](=[O])[O][C]([H])([H])[H])[C](=[O])[O][C]([H])([H])[H].[H][O-]",
        "[H][C]([H])([H])[O][C](=[O])[C]([H])([C](=[O])[O][C]([H])([H])[H])[C]1([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]1([H])[H].[H][O-]",
    ],
    [
        "[O]=[S]([Cl])[Cl].[H][O][C](=[O])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[H][O][C](=[O])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H].[O-][S+]([Cl])[Cl]",
        "[H][O][C+]([O-])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H].[O-][S+]([Cl])[Cl]",
        "[H][O][C+]([O][S]([O-])([Cl])[Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[H][O+]=[C]([O][S]([O-])([Cl])[Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[Cl-].[H][O+]=[C]([O][S+]([O-])[Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[Cl-].[H][O+]=[C]([O][S](=[O])[Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[Cl-].[H][O][C+]([O][S](=[O])[Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[H][O][C]([Cl])([O][S](=[O])[Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[Cl-].[H][O][C]([Cl])([O][S+]=[O])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[O]=[S+][O-].[Cl-].[H][O][C+]([Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[O]=[S]=[O].[Cl-].[H][O][C+]([Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[O]=[S]=[O].[Cl-].[H][O+]=[C]([Cl])[C]([H])([H])[C]([H])([H])[C]1=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]1[H]",
        "[O]=[S]=[O].[Cl-].[H+].[H][C]1=[C]([H])[C]([H])=[C]([C]([H])([H])[C]([H])([H])[C](=[O])[Cl])[C]([H])=[C]1[H]",
        "[O]=[S]=[O].[H][C]1=[C]([H])[C]([H])=[C]([C]([H])([H])[C]([H])([H])[C](=[O])[Cl])[C]([H])=[C]1[H].[H][Cl]",
    ],
]


import json

import numpy as np
import requests

from steer.mechanism.molecule_set import legal_moves_from_smiles


def get_moves(state, _next, n=10):
    response = legal_moves_from_smiles(state, highlight_reactive_center=False)[
        "smiles_list"
    ]

    if (
        _next in response
    ):  # We want n wrong moves here, so we remove the ground truth
        response.remove(_next)

    try:
        return np.random.choice(
            response, min(n, len(response)), replace=False
        ).tolist()
    except:
        return response


def get_task(task, n=10):
    return {
        "id": task[0] + ">>" + task[-1],
        "rxn": task[0] + ">>" + task[-1],
        "steps": task,
        "step_options": [
            get_moves(task[i], task[i + 1], n=n)
            for i in range(0, len(task) - 1)
        ],
    }


ftasks = [get_task(task, n=5) for task in tasks]
with open("benchmark.json", "r") as f:
    olds = json.load(f)

with open("benchmark.json", "w") as f:
    olds.extend(ftasks)
    json.dump(olds, f, indent=2)
