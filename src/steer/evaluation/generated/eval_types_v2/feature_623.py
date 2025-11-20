"""Generated evaluation code for: Late stage Suzuki cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzuki(BaseScoring):
    """
    Evaluates whether a Suzuki cross-coupling reaction occurs in the late stage
    of a synthesis route (within the specified depth threshold from the target).
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"].get("depth_threshold", 2)
        # Suzuki coupling involves organoborane and organohalide reactants
        self.borane_pattern = Chem.MolFromSmarts("[B]")
        self.halide_pattern = Chem.MolFromSmarts("[c,C][F,Cl,Br,I]")
        # Typical Suzuki product: C-C bond formation between aromatic/sp2 carbons
        self.biaryl_pattern = Chem.MolFromSmarts("c-c")
        self.vinyl_aryl_pattern = Chem.MolFromSmarts("C=C-c")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        else:
            # Late stage is better, normalize depth to 0-1 scale
            depth_score = max(0, 1 - x)
            return depth_score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step represents a Suzuki cross-coupling.
        Suzuki coupling typically involves:
        1. One reactant containing boron (boronic acid/ester)
        2. One reactant containing halide (aryl/vinyl halide)
        3. Product shows C-C bond formation
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for presence of borane and halide reactants
            has_borane = False
            has_halide = False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.borane_pattern):
                    has_borane = True
                if reactant.HasSubstructMatch(self.halide_pattern):
                    has_halide = True
            
            # Must have both borane and halide reactants
            if not (has_borane and has_halide):
                return False
            
            # Check if product contains typical Suzuki coupling patterns
            has_coupling_product = (product.HasSubstructMatch(self.biaryl_pattern) or 
                                  product.HasSubstructMatch(self.vinyl_aryl_pattern))
            
            return has_coupling_product
            
        except Exception:
            return False
