"""Generated evaluation code for: Late stage Suzuki cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki cross-coupling reaction occurs as the final step
    in the synthesis route. Returns higher scores when Suzuki coupling is used
    in the last reaction step.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            return 1 if x == 0 else 0  # Only final step (depth 0) gets full score
        else:
            if x < 0:
                return 0  # Condition not met
            return max(0, 1 - x)  # Earlier steps get lower scores
    
    def hit_condition(self, d):
        """
        Detects Suzuki cross-coupling by identifying characteristic patterns:
        - Formation of C-C bond between aryl/vinyl groups
        - Presence of boronic acid/ester patterns in reactants
        - Loss of boron-containing leaving groups
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            prod = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Check for boronic acid or boronic ester patterns in reactants
            boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(-O)-O")
            boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1-O-C-C-O-1")  # Pinacol ester
            boronic_simple_pattern = Chem.MolFromSmarts("[#6]-B(-[OH,O])-[OH,O]")
            
            # Check for halide patterns (coupling partner)
            aryl_halide_pattern = Chem.MolFromSmarts("c-[Br,I,Cl]")
            vinyl_halide_pattern = Chem.MolFromSmarts("C=C-[Br,I,Cl]")
            
            has_boronic_reactant = False
            has_halide_reactant = False
            
            for reactant in reactants:
                if reactant is None:
                    continue
                    
                # Check for boronic compounds
                if (reactant.HasSubstructMatch(boronic_acid_pattern) or 
                    reactant.HasSubstructMatch(boronic_ester_pattern) or
                    reactant.HasSubstructMatch(boronic_simple_pattern)):
                    has_boronic_reactant = True
                    
                # Check for halides
                if (reactant.HasSubstructMatch(aryl_halide_pattern) or
                    reactant.HasSubstructMatch(vinyl_halide_pattern)):
                    has_halide_reactant = True
            
            # Check that product has new C-C bond and no boron
            if prod is None:
                return False
                
            # Product should not contain boron
            has_boron_in_product = any(atom.GetAtomicNum() == 5 for atom in prod.GetAtoms())
            
            # Suzuki coupling signature: boronic compound + halide -> C-C coupled product
            return (has_boronic_reactant and has_halide_reactant and not has_boron_in_product)
            
        except Exception:
            return False
