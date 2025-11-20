"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if the route divides into 
    the specified number of main fragments that are coupled at a specific position.
    
    For convergent synthesis, we want to identify when multiple substantial fragments
    (not just simple reagents) are combined in a coupling reaction.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_position = config["coupling_step_position"]
        
    def route_scoring(self, x) -> float:
        """
        Score based on coupling position:
        - For 'final' position: earlier coupling gets lower score
        - x is the depth fraction where coupling occurs
        """
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_position == "final":
            # Reward coupling that happens late in the synthesis (high x value)
            return x * 10
        elif self.coupling_position == "early":
            # Reward coupling that happens early in the synthesis (low x value)  
            return (1 - x) * 10
        else:
            # For any position, just reward finding the pattern
            return 8.0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step.
        A convergent step combines multiple substantial fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            # Need at least the specified number of fragments
            if len(reactants_smiles) < self.fragment_count:
                return False
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if Chem.MolFromSmiles(r)]
            
            if not product or len(reactants) < self.fragment_count:
                return False
            
            # Filter out small reagents (less than 4 heavy atoms) to focus on substantial fragments
            substantial_fragments = []
            for mol in reactants:
                heavy_atom_count = mol.GetNumHeavyAtoms()
                if heavy_atom_count >= 4:  # Minimum size for a substantial fragment
                    substantial_fragments.append(mol)
            
            # Check if we have the required number of substantial fragments
            if len(substantial_fragments) < self.fragment_count:
                return False
            
            # Verify this is actually a coupling by checking if the substantial fragments
            # contribute significantly to the product
            product_heavy_atoms = product.GetNumHeavyAtoms()
            fragment_atoms_sum = sum(mol.GetNumHeavyAtoms() for mol in substantial_fragments[:self.fragment_count])
            
            # The fragments should account for a significant portion of the product
            # (allowing for some atom count difference due to coupling chemistry)
            atom_efficiency = fragment_atoms_sum / product_heavy_atoms if product_heavy_atoms > 0 else 0
            
            return atom_efficiency >= 0.7  # At least 70% of product atoms come from main fragments
            
        except Exception:
            return False
