"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategies where two substantial fragments
    are joined to form the target molecule. Checks for reactions that combine
    two reactants of similar complexity at an appropriate depth.
    """
    
    def __init__(self, config: Dict):
        self.max_depth_difference = config.get("max_depth_difference", 10)
        self.min_fragment_complexity = config.get("min_fragment_complexity", 5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent step found
        else:
            # Earlier convergent steps (smaller x) are better
            return max(0, 10 - (x * 10))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent synthesis step."""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            reactant_list = reactants_smiles.split(".")
            
            # Need exactly 2 reactants for convergent synthesis
            if len(reactant_list) != 2:
                return False
                
            reactant_mols = []
            complexities = []
            
            for r_smiles in reactant_list:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is None:
                    return False
                reactant_mols.append(mol)
                
                # Calculate fragment complexity (heavy atom count + ring count)
                complexity = mol.GetNumHeavyAtoms() + Chem.GetSSSR(mol).Count()
                complexities.append(complexity)
            
            # Both fragments must meet minimum complexity
            if min(complexities) < self.min_fragment_complexity:
                return False
                
            # Fragments should have similar complexity (convergent strategy)
            complexity_ratio = max(complexities) / max(min(complexities), 1)
            if complexity_ratio > 3.0:  # Not too dissimilar in size
                return False
                
            # Check that both reactants contribute significant structure to product
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
                
            product_atoms = product_mol.GetNumHeavyAtoms()
            
            # Each fragment should contribute at least 20% of final structure
            min_contribution = product_atoms * 0.2
            if any(complexity < min_contribution for complexity in complexities):
                return False
                
            return True
            
        except (ValueError, AttributeError):
            return False
