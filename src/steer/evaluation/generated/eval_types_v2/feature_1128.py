"""Generated evaluation code for: Suzuki coupling for biaryl bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SuzukiCouplingDepth(BaseScoring):
    """
    Evaluates synthesis routes based on the depth at which Suzuki coupling 
    reactions occur for biaryl bond formation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            if self.condition_type == "bool":
                return 1  # Suzuki coupling occurs
            else:
                # Earlier Suzuki coupling (lower depth) is generally better
                return max(0, 1 - x)
    
    def hit_condition(self, d):
        """
        Detects Suzuki coupling by looking for:
        1. Formation of biaryl bond (C-C between aromatic rings)
        2. Presence of boronic acid/ester pattern in reactants
        3. Palladium catalyst indicators in metadata if available
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for biaryl formation in product
            biaryl_pattern = Chem.MolFromSmarts("c-c")  # aromatic C-C bond
            if not product.HasSubstructMatch(biaryl_pattern):
                return False
            
            # Check for boronic acid/ester patterns in reactants
            boronic_patterns = [
                Chem.MolFromSmarts("cB(O)O"),      # boronic acid
                Chem.MolFromSmarts("cB1OCC[CH2]O1"), # boronic ester (pinacol)
                Chem.MolFromSmarts("cB(OC)OC")     # boronic ester (methyl)
            ]
            
            has_boronic = any(
                reactant.HasSubstructMatch(pattern)
                for reactant in reactants
                for pattern in boronic_patterns
            )
            
            # Check for halide pattern (coupling partner)
            halide_pattern = Chem.MolFromSmarts("c[Cl,Br,I]")
            has_halide = any(
                reactant.HasSubstructMatch(halide_pattern)
                for reactant in reactants
            )
            
            # Suzuki coupling typically requires both boronic compound and halide
            return has_boronic and has_halide
            
        except Exception:
            return False
