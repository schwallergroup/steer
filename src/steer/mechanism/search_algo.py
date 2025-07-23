import asyncio
from typing import List, Optional

import numpy as np
from colorama import Fore
from rdkit.Chem import MolFromSmiles, MolToSmiles

from steer.mechanism.molecule_set import MoleculeSet, MoleculeSetTooCrazyError
from steer.mechanism.sequential import LM


def rdkit_canonicalize(smi):
    return MolToSmiles(MolFromSmiles(smi))


async def score_one_step(
    scorer: LM,
    start_ms: MoleculeSet,
    goal_ms: MoleculeSet,
    curr_ms: MoleculeSet,
    all_next_ms: Optional[
        List[MoleculeSet]
    ] = None,  # If None, will be calculated
    history_ms: List[MoleculeSet] = [],
):

    if all_next_ms is None:
        all_next_ms = []
        for move in curr_ms.all_legal_moves:
            try:
                next_ms = curr_ms.make_move(move)
                all_next_ms.append(next_ms)

            except MoleculeSetTooCrazyError:
                print(f"Problem with move: {move} on mol_set {curr_ms}")
                continue

    response = await asyncio.gather(
        *[
            scorer.run(
                rxn=f"{start_ms.canonical_smiles}>>{goal_ms.canonical_smiles}",
                history=[
                    hist.canonical_smiles for hist in history_ms + [curr_ms]
                ],
                step=f"{curr_ms.canonical_smiles}>>{nms.canonical_smiles}",
                expert_description=(
                    scorer.expert_description
                    if scorer.prompt_needs_expert_description
                    else ""
                ),
            )
            for nms in all_next_ms
        ]
    )

    scores = [scorer._parse_score(result) for result in response]

    return all_next_ms, scores


async def beam_search(
    scorer: LM,
    start_smiles: str,
    goal_smiles: str,
    beam_width: int,
    tie_breaker: str = "take_all",
    num_llm_calls_before_rescoring: int = 2,
    min_score_to_keep: float = 5,
):
    global last_cost_checkpoint, check_every

    start_ms = MoleculeSet(start_smiles)
    start_can_smi = start_ms.canonical_smiles
    goal_ms = MoleculeSet(goal_smiles)
    goal_can_smi = goal_ms.canonical_smiles

    nodes_seen = {
        0: {
            "ms": start_ms,
            "smiles": start_can_smi,
            "score": 0,
            "parent_idx": None,
            "depth": 0,
        }
    }

    last_seen_idx = 0

    nodes_expanded = dict()

    # select idx with lowest score
    lowest_score_idx = min(nodes_seen, key=lambda x: nodes_seen[x]["score"])
    selected_node = nodes_seen.pop(lowest_score_idx)
    nodes_expanded[lowest_score_idx] = selected_node

    print(
        f"{Fore.YELLOW}Careful: For now only the exact goal is considered as a goal, no substructure detection{Fore.RESET}"
    )

    lowest_score_goal = None
    list_of_valid_goal_nodes = []

    n_nodes_scored = 0
    n_nodes_expanded = 0
    llm_calls = 0

    while (
        lowest_score_goal is None
        or selected_node["score"] <= lowest_score_goal
    ):

        if llm_calls - last_cost_checkpoint > check_every:
            print(
                f"{Fore.LIGHTBLACK_EX}Checkpoint: {llm_calls} LLM calls{Fore.RESET}"
            )
            last_cost_checkpoint += check_every
            breakpoint()
            print()

        n_nodes_expanded += 1

        # If the node is the goal, no need to expand it further
        if selected_node["smiles"] == goal_can_smi:
            print(
                f"{Fore.LIGHTBLACK_EX}Selected node is the goal, passing to next node{Fore.RESET}"
            )
            history_selected_node = get_history(selected_node, nodes_expanded)
            history_str = ">>".join(
                [
                    h.canonical_smiles
                    for h in [start_ms]
                    + history_selected_node
                    + (
                        [selected_node["ms"]]
                        if selected_node["parent_idx"] is not None
                        else []
                    )
                ]
            )
            print(
                f"{Fore.LIGHTBLACK_EX}History:\n{history_str} score: {selected_node['score']:5.3f}{Fore.RESET}"
            )

        else:
            print(
                f"{Fore.LIGHTBLACK_EX}Selected node: {selected_node}{Fore.RESET}"
            )
            history_selected_node = get_history(selected_node, nodes_expanded)
            history_str = ">>".join(
                [
                    h.canonical_smiles
                    for h in [start_ms]
                    + history_selected_node
                    + (
                        [selected_node["ms"]]
                        if selected_node["parent_idx"] is not None
                        else []
                    )
                ]
            )
            print(
                f"{Fore.LIGHTBLACK_EX}History:\n{history_str} score: {selected_node['score']:5.3f}{Fore.RESET}"
            )

            # breakpoint()
            # print()

            nms_list, scores = await score_one_step(
                scorer=scorer,
                start_ms=start_ms,
                goal_ms=goal_ms,
                curr_ms=selected_node["ms"],
                history_ms=history_selected_node,
            )

            reranked_nms_scores = sorted(
                [(n_ms, score) for n_ms, score in zip(nms_list, scores)],
                key=lambda x: x[1],
                reverse=True,  # Here since we didn't calculate the opposite score, we sort in descending order
            )

            for n_ms, score in reranked_nms_scores:
                print(
                    f"{Fore.LIGHTBLUE_EX}LLM Score: {score:04.1f}/10 for Molecule set: {n_ms.rdkit_canonical_smiles} {'(<- chosen)' if MoleculeSet.default_canonicalization == 'RDKit' else '(chosen ->)'} (Explicit canonicalization: {n_ms.explicit_canonical_smiles} )"
                )
                llm_calls += 1
                n_nodes_scored += 1

            if reranked_nms_scores[-1][1] < min_score_to_keep:
                print(
                    f"Getting rid of all scores below {min_score_to_keep} (min score is {reranked_nms_scores[-1][1]})"
                )

            rescored_ms = {
                n_ms.canonical_smiles: {
                    "ms": n_ms,
                    "score_list": [score],
                    "score": np.mean(score),
                    "op_score": opposite_score(np.mean(score)),
                }
                for n_ms, score in zip(nms_list, scores)
                if score >= min_score_to_keep
            }

            if num_llm_calls_before_rescoring > 1:
                print(
                    f"Refining scores of the remaining nodes with {num_llm_calls_before_rescoring-1} additional LLM calls"
                )
                for _ in range(num_llm_calls_before_rescoring - 1):
                    if len(rescored_ms) == 0:
                        print("No nodes to rescore")
                        break
                    next_nms = [val["ms"] for val in rescored_ms.values()]
                    nms_list, scores = await score_one_step(
                        scorer=scorer,
                        start_ms=start_ms,
                        goal_ms=goal_ms,
                        curr_ms=selected_node["ms"],
                        all_next_ms=next_nms,
                        history_ms=history_selected_node,
                    )

                    for nms, score in zip(nms_list, scores):
                        rescored_ms[nms.canonical_smiles]["score_list"].append(
                            score
                        )
                        rescored_ms[nms.canonical_smiles]["score"] = np.mean(
                            rescored_ms[nms.canonical_smiles]["score_list"]
                        )
                        rescored_ms[nms.canonical_smiles]["op_score"] = (
                            opposite_score(
                                rescored_ms[nms.canonical_smiles]["score"]
                            )
                        )

                    sorted_next_nodes_scores = sorted(
                        [
                            (val["ms"], val["op_score"], val["score"])
                            for val in rescored_ms.values()
                        ],
                        key=lambda x: x[1],
                    )

                    for nms, op_score, score in sorted_next_nodes_scores:
                        print(
                            f"{Fore.LIGHTBLUE_EX}LLM Score: {score:04.1f}/10 Search score: {op_score:05.3f} for Molecule set: {nms.rdkit_canonical_smiles} {'(<- chosen)' if MoleculeSet.default_canonicalization == 'RDKit' else '(chosen ->)'} (Explicit canonicalization: {nms.explicit_canonical_smiles} )"
                        )
                        llm_calls += 1

                    if sorted_next_nodes_scores[-1][2] < min_score_to_keep:
                        print(
                            f"Getting rid of all scores below {min_score_to_keep} (min score is {sorted_next_nodes_scores[-1][2]})"
                        )

                        rescored_ms = {
                            smiles: val
                            for smiles, val in rescored_ms.items()
                            if val["score"] >= min_score_to_keep
                        }

            if tie_breaker == "take_all":
                if len(rescored_ms) == 0:
                    kept_next_nodes_scores = []
                elif len(sorted_next_nodes_scores) < beam_width:
                    kept_next_nodes_scores = sorted_next_nodes_scores
                else:
                    score_cutoff = sorted_next_nodes_scores[beam_width - 1][1]
                    kept_next_nodes_scores = [
                        s
                        for s in sorted_next_nodes_scores
                        if s[1] <= score_cutoff
                    ]

            elif tie_breaker == "llm_max_1_then_take_all":
                score_cutoff = sorted_next_nodes_scores[beam_width - 1][1]
                kept_next_nodes_scores = [
                    s for s in sorted_next_nodes_scores if s[1] <= score_cutoff
                ]

                # If there are more kept nodes than the beam width
                if len(kept_next_nodes_scores) > beam_width:
                    print("Re-scoring the kept nodes")
                    # We will reassign scores of all the kept nodes and caluclate their average score
                    dict_of_observed_scores = {
                        nms.canonical_smiles: [ori_score]
                        for nms, _, ori_score in kept_next_nodes_scores
                    }

                    nms, new_scores = await score_one_step(
                        scorer=scorer,
                        start_ms=start_ms,
                        goal_ms=goal_ms,
                        curr_ms=selected_node["ms"],
                        all_next_ms=[
                            nms for nms, _, _ in kept_next_nodes_scores
                        ],
                        history_ms=history_selected_node,
                    )

                    for n_ms, new_score in zip(nms, new_scores):
                        dict_of_observed_scores[n_ms.canonical_smiles].append(
                            new_score
                        )

                    avg_scores = {
                        nms_smiles: sum(scores) / len(scores)
                        for nms_smiles, scores in dict_of_observed_scores.items()
                    }
                    op_scores = {
                        nms_smiles: opposite_score(score)
                        for nms_smiles, score in avg_scores.items()
                    }

                    sorted_avg_scores = sorted(
                        [
                            (
                                n_ms,
                                op_scores[n_ms.canonical_smiles],
                                avg_scores[n_ms.canonical_smiles],
                            )
                            for n_ms in nms
                        ],
                        key=lambda x: x[1],
                    )

                    score_cutoff = sorted_avg_scores[beam_width - 1][1]
                    kept_next_nodes_scores = [
                        s for s in sorted_avg_scores if s[1] <= score_cutoff
                    ]

                    for nms, op_score, original_score in sorted_avg_scores:
                        print(
                            f"{Fore.LIGHTBLUE_EX}LLM Score: {original_score:04.1f}/10 Search score: {op_score:05.3f} for Molecule set: {nms.rdkit_canonical_smiles} {'(<- chosen)' if MoleculeSet.default_canonicalization == 'RDKit' else '(chosen ->)'} (Explicit canonicalization: {nms.explicit_canonical_smiles} )"
                        )
                        llm_calls += 1

            else:
                raise NotImplementedError(
                    f"tie_breaker {tie_breaker} not implemented"
                )

            for nms, score, original_score in kept_next_nodes_scores:
                nodes_seen[last_seen_idx + 1] = {
                    "ms": nms,
                    "smiles": nms.canonical_smiles,
                    "score": score + selected_node["score"],
                    "parent_idx": lowest_score_idx,
                    "depth": selected_node["depth"] + 1,
                }

                last_seen_idx += 1

        if len(nodes_seen) == 0:
            print(f"{Fore.RED}No more nodes to expand{Fore.RESET}")
            break

        lowest_score_idx = min(
            nodes_seen, key=lambda x: nodes_seen[x]["score"]
        )
        selected_node = nodes_seen.pop(lowest_score_idx)
        nodes_expanded[lowest_score_idx] = selected_node

        if selected_node["smiles"] == goal_can_smi:
            list_of_valid_goal_nodes.append(selected_node)
            if lowest_score_goal is None:
                lowest_score_goal = selected_node["score"]
            else:
                lowest_score_goal = min(
                    lowest_score_goal, selected_node["score"]
                )

    if len(list_of_valid_goal_nodes) == 0:
        print()
        print(f"{Fore.RED}No goal reached{Fore.RESET}")
    else:
        print()
        print(
            f"{len(list_of_valid_goal_nodes)} {'goal' if len(list_of_valid_goal_nodes) == 1 else 'equivalent goals'} reached"
        )
        print(f"{n_nodes_scored} nodes scored ({llm_calls} LLM calls)")
        print(f"{n_nodes_expanded} nodes expanded")
        for goal_nbr, goal_node in enumerate(list_of_valid_goal_nodes):
            print(
                f"{Fore.GREEN}------------------ GOAL #{goal_nbr+1:03.0f} ------------------{Fore.RESET}"
            )
            print(f"{Fore.GREEN}Goal reached!{Fore.RESET}")
            print(f"Goal node: {goal_node}")
            history_str = "\n".join(
                [
                    h.canonical_smiles
                    for h in [start_ms]
                    + get_history(goal_node, nodes_expanded)
                    + [goal_ms]
                ]
            )
            print(f"History:\n{history_str} score: {goal_node['score']:5.3f}")


def opposite_score(score):

    minimal_score = 0.001
    return minimal_score + (1 - (score / 10)) ** 2


def get_history(node, expanded_nodes):
    reverse_history = []
    while node["parent_idx"] is not None:
        reverse_history.append(node["smiles"])
        node = expanded_nodes[node["parent_idx"]]

    return [MoleculeSet(smi) for smi in reverse_history[:0:-1]]


async def main(
    scorer: LM,
    start_smiles: str,
    goal_smiles: str,
    beam_width: int,
    tie_breaker: str = "take_all",
    num_llm_calls_before_rescoring: int = 3,
    min_score_to_keep: float = 5,
):

    await beam_search(
        scorer=scorer,
        start_smiles=start_smiles,
        goal_smiles=goal_smiles,
        beam_width=beam_width,
        tie_breaker=tie_breaker,
        num_llm_calls_before_rescoring=num_llm_calls_before_rescoring,
        min_score_to_keep=min_score_to_keep,
    )


if __name__ == "__main__":
    # Nu-attack on cyclohexanone
    expert_description = "1. First, the carbonyl will resonate to create an empty orbital on the carbon, which can be attacked by ammonia\n2. Then, a proton transfer can happen between the positively charged Nitrogen and the negatively charged Oxygen"
    correct_path = [
        "[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][N]([H])[H]",
        "[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][N]([H])[H]",
        "[H][C]1([H])[C]([H])([H])[C]([H])([H])[C]([O-])([N+]([H])([H])[H])[C]([H])([H])[C]1([H])[H]",
        "[H+].[H][N]([H])[C]1([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H][O][C]1([N]([H])[H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    ]

    # carboxylic acid to acyl chloride

    expert_description = "1. First, the S=O bond will resonate, as well as the carbonyl, preparing the molecules in such a way that the oxygen of the carbonyl can attack the sulfur. This attack being pushed by the alcohol, resonating to recreate a carbonyl C=O bond.\n2. Then, the negatively charged oxygen on the sulfur will attack the sulfur after a chlorid ion has been eliminated.\n3. The carbonyl can then resonate to let the released chloride ion attack the carbonyl carbon.\n4. After that, the sulfur will release a second chloride ion, allowing to the singly bonded oxygen to doubly bond with the sulfur, pushed by the resonance of the alcohol on the carbon.\n5. Finally, the doubly bonded oxygen can release a proton that can be picked up by the chloride ion, regenerating the acidic conditions in the form of hydrochloride."
    correct_path = [
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
    ]

    # NaBH4 reduction of cyclohexanone

    correct_path = [
        "[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][B-]([H])([H])[H].[Na+]",
        "[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][B-]([H])([H])[H].[Na+]",
        "[H-].[H][B]([H])[H].[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[Na+]",
        "[H][B]([H])[H].[H][C]1([H])[C]([H])([H])[C]([H])([H])[C]([H])([O-])[C]([H])([H])[C]1([H])[H].[Na+]",
        "[H][B-]([H])([H])[O][C]1([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[Na+]",
    ]

    # Butanone acetalyzation with ethylen glycol
    expert_description = "1. Because of the acidic conditions, the carbonyl oxygen will be protonated\n2. Then the carbonyl can resonate to create an empty orbital on the carbon, which can be attacked by the oxygen of diol\n3. In the two following steps, a proton transfer will occur by deprotonating the charged oxygen and then protonating the tertiary alcohol\n4. At this point, we can eliminate water so that the terminal alcohol can attack the carbocation\n5. Finally, the catalytic proton can be released from the charged oxygen to regenerate the acidic conditions"
    correct_path = [
        "OCCO.CC(CC)=O.[H+]",
        "OCCO.CC(CC)=[OH+]",
        "OCCO.C[C+](CC)O",
        "CC([OH+]CCO)(CC)O",
        "CC(OCCO)(CC)O.[H+]",
        "CC(OCCO)(CC)[OH2+]",
        "C[C+](OCCO)(CC).O",
        "CC1(OCC[OH+]1)(CC).O",
        "CC1(OCCO1)(CC).O.[H+]",
    ]

    # Acetone acetalyzation with methanol

    correct_path = [
        "C=O.OC.OC.[H+]",
        "C=[OH+].OC.OC",
        "[CH2+]O.OC.OC",
        "C[OH+]CO.OC",
        "COCO.OC.[H+]",
        "COC[OH2+].OC",
        "CO[CH2+].OC.O",
        "COC[OH+]C.O",
        "COCOC.O.[H+]",
    ]

    # Selective Nu-attack 1

    expert_description = "1. First, the ethyl substited carbonyl will resonate to create an empty orbital on the carbon, which can be attacked by the nitrogen of the ammonia\n2. Then, a proton transfer will occur between the positively charged nitrogen and the negatively charged oxygen"
    correct_path = [
        "O=C(CC)CC(=O)C.N",
        "[O-][C+](CC)CC(=O)C.N",
        "[O-]C(CC)([NH3+])CC(=O)C",
        "[O-]C(CC)(N)CC(=O)C.[H+]",
        "OC(CC)(N)CC(=O)C",
    ]

    # Selective Nu-attack 2

    correct_path = [
        "O=C(CC)CC(=O)C.N",
        "O=C(CC)C[C+]([O-])C.N",
        "O=C(CC)CC([NH3+])([O-])C",
        "O=C(CC)CC(N)([O-])C.[H+]",
        "O=C(CC)CC(N)(O)C",
    ]

    # Wittig with phosphine

    # Same expert description for triphenylphosphine
    expert_description = "1. First of all, the P=C bond will resonate to create an empty orbital on the phosphorus, which can be attacked by the carbonyl oxygen, creating a P-O bond\n2. Then, the negatively charged carbon can attack the other carbon after the carbonyl has resonated to create a carbocation, as this step will form the cyclic intermediate we want.\n3. After that, the C-O bond can ionize onto the oxygen, allowing it to attack one more time the phosphorus and creating the P=O double bond\n4. Finally, the negatively charged phosphorus can ionize its bond onto the carbon, allowing this carbon to create the ethene double bond"
    correct_path = [
        "[H][C]([H])=[O].[H][C]([H])=[P]([H])([H])[H]",
        "[H][C]([H])=[P]([H])([H])[H].[H][C+]([H])[O-]",
        "[H][C+]([H])[O-].[H][C-]([H])[P+]([H])([H])[H]",
        "[H][C]([H])([O-])[C]([H])([H])[P+]([H])([H])[H]",
        "[H][C]1([H])[O][P]([H])([H])([H])[C]1([H])[H]",
        "[H][C+]([H])[C]([H])([H])[P]([H])([H])([H])[O-]",
        "[H][C+]([H])[C-]([H])[H].[H][P+]([H])([H])[O-]",
        "[H][C]([H])=[C]([H])[H].[H][P+]([H])([H])[O-]",
        "[H][C]([H])=[C]([H])[H].[H][P]([H])([H])=[O]",
    ]

    # Hemiacetalization of formaldehyde with methanol
    expert_description = "1. First, the carbonyl will be protonated, due to the acidic conditions\n2. Then, the methanol will attack the carbocation created by the resonance of the carbonyl\n3. Finally, the oxygen positively charged will release a proton, regenerating the acidic conditions"
    correct_path = [
        "CC=O.CO.[H+]",
        "CC=[OH+].CO",
        "C[CH+]O.CO",
        "CC([OH+]C)O",
        "CC(OC)O.[H+]",
    ]

    # Second half of formaldehyde acetalization with methanol
    expert_description = "1. First, the alcohol will be protonated, due to the acidic conditions\n2. Then, the methanol can attack the carbocation created by the water leaving group\n3. Finally, the oxygen positively charged will release a proton, regenerating the acidic conditions"
    correct_path = [
        "CC(OC)O.CO.[H+]",
        "CC(OC)[OH2+].CO",
        "C[CH+](OC).CO.O",
        "CC(OC)([OH+]C).O",
        "CC(OC)(OC).O.[H+]",
    ]

    # Molozonide to ozonide
    expert_description = "1. First, a bond will break between two oxygen atoms, as well as in between the two carbon atoms, creating a carbocation with the negatively charged oxygen, and inversely a carbanion with the positively charged oxygen.\n2. Then, these two structures will stabilize. The first one by simply resonating, creating a formaldehyde molecule, and the second, creating a C=O double bond.\n3. At this point, we reached an important intermediate, and the double bond can again resonate creating two oxygen atoms with opposite charge next to each other.\n4. This last step created an empty orbital on the carbon, that can be attacked by the oxygen of the formaldehyde.\n5. After that, we can resonate the carbonyl to create a carbocation, that can be attacked by the negatively charged oxygen and close the cycle in its final configuration."
    correct_path = [
        "C1COOO1",
        "[O+]CCO[O-]",
        "[O+][CH2-].[CH2+]O[O-]",
        "C=O.[CH2+]O[O-]",
        "C=O.C=[O+][O-]",
        "C=O.[CH2+]O[O-]",
        "C=[O+]CO[O-]",
        "[CH2+]OCO[O-]",
        "O1COOC1",
    ]

    # Michael addition in acidic conditions

    expert_description = "1. First, the Nitrogen atom of the michael donor will be protonated, and resonances will happen to form the enamine intermediate\n2. By the time the enamine intermediate is formed, the water can attack the proton in solution, keeping the acidic conditions but not letting the free proton in solution\n3. Then, a concerted move of the michael acceptor being ionized, and the enamine C=C bond being ionized and attacking the carbon positively charged will happen\n4. After that, the negatively charged carbon can attack the proton in solution that the water has to release to the solution\n5. Finally, the nitrogen can release a proton, and stabilize the molecule by attacking its neighboring carbon."

    correct_path = [
        "[H+].[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]2=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]2[N]=[C]1[C]([H])([H])[H].[H][O][H]",
        "[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C]([C]([H])([H])[H])=[N+]2[H].[H][O][H]",
        "[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C]([H])([H])[H])[N]2[H].[H][O][H]",
        "[H+].[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C-]([H])[H])[N]2[H].[H][O][H]",
        "[H+].[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C](=[C]([H])[H])[N]2[H].[H][O][H]",
        "[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C](=[C]([H])[H])[N]2[H].[H][O+]([H])[H]",
        "[H][C]1=[C]([H])[C]([C]([H])=[C]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C-]([H])[H])[N]2[H].[H][O+]([H])[H]",
        "[H][C]1=[C]([H])[C]([C+]([H])[C-]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C-]([H])[H])[N]2[H].[H][O+]([H])[H]",
        "[H][C]1=[C]([H])[C]([C]4([H])[C-]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C]4([H])[H])[N]2[H].[H][O+]([H])[H]",
        "[H][C]1=[C]([H])[C]([C]4([H])[C-]([H])[N+](=[O])[O-])=[C]([H])[C]([H])=[C]1[Br].[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C]4([H])[H])[N]2[H].[H][O][H].[H+]",
        "[H][C]1=[C]([H])[C]([H])=[C]2[C](=[C]1[H])[C]([H])=[C]([H])[C+]([C]([H])([H])[C]([H])([C]1=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]1[H])[C]([H])([H])[N+](=[O])[O-])[N]2[H].[H][O][H]",
        "[H+].[H][C]1=[C]([H])[C]([H])=[C]2[C]([H])=[C]([H])[C+]([C]([H])([H])[C]([H])([C]3=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]3[H])[C]([H])([H])[N+](=[O])[O-])[N-][C]2=[C]1[H].[H][O][H]",
        "[H+].[H][C]1=[C]([H])[C]2=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]2[N]=[C]1[C]([H])([H])[C]([H])([C]1=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]1[H])[C]([H])([H])[N+](=[O])[O-].[H][O][H]",
    ]

    last_cost_checkpoint = 0
    check_every = 500
    # model = "random"
    model = "claude-3-5-sonnet"
    # prompt = "steer.mechanism.prompts.preprint_prompt_last_step_plus_game"
    prompt = (
        "steer.mechanism.prompts.preprint_prompt_last_step_plus_game_expert"
    )
    needs_expert_description = True

    MoleculeSet.default_canonicalization = "RDKit"

    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print()

    scorer = LM(
        prompt=prompt,
        prompt_needs_expert_description=needs_expert_description,
        expert_description=(
            expert_description if needs_expert_description else ""
        ),
        model=model,
        project_name="steer-mechanism-search-test",
    )

    i = 0

    try:
        res = asyncio.run(
            main(
                scorer=scorer,
                start_smiles=correct_path[0],
                goal_smiles=correct_path[-1],
                num_llm_calls_before_rescoring=3,
                min_score_to_keep=5.01,
                beam_width=3,
                tie_breaker="take_all",
            )
        )
    except KeyboardInterrupt:
        print("Manually interrupted")
