"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting coupling reactions
    that join two significant molecular fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_step_position = config.get("coupling_step_position", "middle")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        
        # For convergent synthesis, middle-stage coupling is preferred
        if self.coupling_step_position == "middle":
            # Optimal convergent coupling occurs around 0.3-0.7 depth
            if 0.3 <= x <= 0.7:
                return 1.0
            elif x < 0.3:
                return x / 0.3  # Early coupling gets partial credit
            else:
                return max(0, (1.0 - x) / 0.3)  # Late coupling gets less credit
        elif self.coupling_step_position == "early":
            return 1.0 - x  # Earlier is better
        else:  # late
            return x  # Later is better
    
    def hit_condition(self, d) -> bool:
        """
        Detects convergent coupling by identifying reactions where:
        1. Multiple reactants combine to form one product
        2. Reactants are of similar complexity (indicating fragment coupling)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1].split(".")
        
        # Must have at least the specified number of fragments
        if len(reactants_smiles) < self.fragment_count:
            return False
            
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles]
            
            if not product or not all(reactants):
                return False
            
            # Calculate molecular complexity (heavy atom count as proxy)
            product_complexity = product.GetNumHeavyAtoms()
            reactant_complexities = [mol.GetNumHeavyAtoms() for mol in reactants]
            
            # Filter out small reagents (< 20% of product complexity)
            min_fragment_size = max(3, product_complexity * 0.2)
            significant_fragments = [c for c in reactant_complexities if c >= min_fragment_size]
            
            # Check if we have the required number of significant fragments
            if len(significant_fragments) < self.fragment_count:
                return False
            
            # For convergent synthesis, fragments should be of similar size
            # (none should dominate - largest shouldn't be >70% of total)
            total_fragment_atoms = sum(significant_fragments)
            max_fragment = max(significant_fragments)
            
            if total_fragment_atoms > 0 and max_fragment / total_fragment_atoms <= 0.7:
                return True
                
        except Exception:
            return False
            
        return False
