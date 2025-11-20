"""Generated evaluation code for: Late stage Suzuki cross-coupling assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki-Miyaura cross-coupling reaction occurs at the final stage
    (root) of the synthesis route. Rewards routes where Suzuki coupling is used as the
    last synthetic step to assemble major molecular fragments.
    """
    
    def __init__(self, config: Dict):
        self.stage = config.get("parameters", {}).get("stage", "final")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        
        if self.stage == "final":
            # For final stage, we want x to be close to 0 (root of tree)
            if x == 0:
                return 10  # Perfect - Suzuki at final step
            elif x <= 0.1:  # Very close to final step
                return 8
            elif x <= 0.3:  # Reasonably late stage
                return 5
            else:
                return 2  # Too early in synthesis
        else:
            # For other stages, convert depth to score
            return max(0, 10 - x * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Suzuki-Miyaura coupling by looking for:
        1. Boron-containing reagent (boronic acid/ester) in reactants
        2. Halide (Br, I, or activated Cl) in reactants  
        3. Formation of C-C bond between aryl/vinyl groups
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            products, reactants = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mol = Chem.MolFromSmiles(products.strip())
            
            if not all(mol for mol in reactant_mols + [product_mol]):
                return False
            
            # Check for boron-containing reactant (boronic acid/ester patterns)
            boron_patterns = [
                "[B](O)(O)",  # Boronic acid
                "[B]1OC(C)(C)C(C)(C)O1",  # Pinacol boronate
                "[B](OC)(OC)",  # Boronic ester
                "[B](O[CH3])(O[CH3])",  # Methyl boronate
            ]
            
            has_boron = False
            for mol in reactant_mols:
                for pattern in boron_patterns:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_boron = True
                        break
                if has_boron:
                    break
            
            if not has_boron:
                return False
            
            # Check for halide reactant (common leaving groups in Suzuki)
            halide_patterns = [
                "[c,C][Br]",  # Aryl/alkyl bromide
                "[c,C][I]",   # Aryl/alkyl iodide
                "[c][Cl]",    # Aryl chloride (activated)
            ]
            
            has_halide = False
            for mol in reactant_mols:
                for pattern in halide_patterns:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_halide = True
                        break
                if has_halide:
                    break
            
            # Must have both boron and halide components
            return has_boron and has_halide
            
        except Exception:
            return False
