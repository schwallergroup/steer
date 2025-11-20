"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentIsoxazoleStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are joined
    via isoxazole ring formation. Checks for the presence of isoxazole formation
    reaction and measures the convergence depth in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_reaction = config["coupling_reaction"]
        # Isoxazole SMARTS pattern for detection
        self.isoxazole_pattern = "[#6]1=[#7][#8][#6]=[#6]1"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent isoxazole formation found
        else:
            # Earlier convergence (lower depth fraction) is better
            # Scale to 0-10 with preference for mid-to-late stage convergence
            if x < 0.3:
                return 5 + (0.3 - x) * 10  # Bonus for early convergence
            elif x > 0.8:
                return max(0, 5 - (x - 0.8) * 15)  # Penalty for very late convergence
            else:
                return 8  # Optimal range
                
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent isoxazole formation
        by detecting isoxazole ring formation from two separate fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or len(reactants) < 2:
                return False
                
            # Check if product contains isoxazole ring
            isoxazole_smarts = Chem.MolFromSmarts(self.isoxazole_pattern)
            if not product.HasSubstructMatch(isoxazole_smarts):
                return False
                
            # Check if isoxazole is being formed (not present in reactants)
            reactants_have_isoxazole = any(
                reactant.HasSubstructMatch(isoxazole_smarts) 
                for reactant in reactants if reactant
            )
            
            if reactants_have_isoxazole:
                return False  # Isoxazole already exists, not formation
                
            # Verify convergent strategy: at least 2 substantial fragments
            substantial_reactants = [
                r for r in reactants 
                if r and r.GetNumAtoms() > 3  # Filter out small reagents
            ]
            
            return len(substantial_reactants) >= self.fragment_count
            
        except Exception:
            return False
