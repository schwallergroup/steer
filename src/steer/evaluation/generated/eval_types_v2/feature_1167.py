"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if the route assembles 
    the target from multiple major fragments via a coupling reaction.
    
    This class identifies convergent routes where two or more substantial fragments 
    are combined in a single coupling step, typically indicating an efficient 
    synthetic strategy.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.target_coupling_step = config.get("coupling_step", 1)
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score. Earlier convergent steps are better.
        
        Args:
            x: Depth fraction where convergent coupling occurs (-1 if not found)
            
        Returns:
            Score from 0-1 (higher is better)
        """
        if x < 0:
            return 0  # No convergent coupling found
        else:
            return 1 - x  # Earlier coupling is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step.
        
        Args:
            d: Reaction node dictionary containing metadata
            
        Returns:
            True if this is a convergent coupling reaction
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactant_smiles_list = reactants_smiles.split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles_list]
            
            if not product or not all(reactants):
                return False
            
            # Check if we have the expected number of fragments
            if len(reactants) < self.fragment_count:
                return False
            
            # Filter out small molecules (likely reagents/catalysts)
            major_fragments = []
            for reactant in reactants:
                if reactant.GetNumHeavyAtoms() >= 5:  # Threshold for "major" fragment
                    major_fragments.append(reactant)
            
            # Check if we have enough major fragments for convergent synthesis
            if len(major_fragments) < self.fragment_count:
                return False
            
            # Verify this is a true coupling by checking molecular complexity
            product_heavy_atoms = product.GetNumHeavyAtoms()
            total_reactant_heavy_atoms = sum(r.GetNumHeavyAtoms() for r in major_fragments)
            
            # Product should contain most atoms from the major fragments
            # (allowing for small loss due to coupling reaction)
            if product_heavy_atoms < 0.8 * total_reactant_heavy_atoms:
                return False
            
            # Additional check: ensure fragments are of reasonable size relative to product
            min_fragment_size = max(r.GetNumHeavyAtoms() for r in major_fragments)
            if min_fragment_size < 0.2 * product_heavy_atoms:
                return False
            
            return True
            
        except Exception:
            return False
