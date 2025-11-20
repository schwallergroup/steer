"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates whether a synthesis route follows a convergent strategy by coupling
    major fragments at a specific step. Checks if the route builds separate complex
    fragments before combining them in a coupling reaction.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.target_coupling_step = config["coupling_step"]
        self.min_fragment_complexity = config.get("min_fragment_complexity", 5)  # minimum heavy atoms
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Reward coupling that happens close to target step
            # Earlier coupling (lower step number) is generally better for convergent synthesis
            step_difference = abs(x * 10 - self.target_coupling_step)  # x is normalized depth
            return max(0, 10 - step_difference)
    
    def hit_condition(self, d):
        """
        Check if this reaction represents a convergent coupling of major fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            if "." not in reactants_smiles:
                return False  # Need multiple reactants for convergent step
                
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            if len(reactants) < self.fragment_count:
                return False
                
            # Check if we have the right number of substantial fragments
            major_fragments = []
            for reactant in reactants:
                heavy_atom_count = reactant.GetNumHeavyAtoms()
                if heavy_atom_count >= self.min_fragment_complexity:
                    major_fragments.append(reactant)
            
            if len(major_fragments) < self.fragment_count:
                return False
                
            # Check if this looks like a coupling reaction by examining bond formation
            return self._is_coupling_reaction(product, major_fragments)
            
        except Exception:
            return False
    
    def _is_coupling_reaction(self, product, fragments):
        """
        Determine if the reaction represents coupling of fragments by checking
        if the product contains substructures from multiple fragments.
        """
        if not product or len(fragments) < 2:
            return False
            
        # Check if product contains significant portions of each fragment
        matches = 0
        for fragment in fragments[:self.fragment_count]:
            # Create a more flexible pattern by removing some specificity
            pattern = self._create_flexible_pattern(fragment)
            if pattern and product.HasSubstructMatch(pattern):
                matches += 1
                
        # Require that we can match at least the specified number of fragments
        return matches >= self.fragment_count
    
    def _create_flexible_pattern(self, mol):
        """
        Create a flexible SMARTS pattern from a molecule for substructure matching.
        """
        try:
            # Use the molecule itself as pattern, but could be made more flexible
            # by converting to SMARTS and generalizing some features
            return mol
        except Exception:
            return None
