"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two major fragments are coupled.
    Checks for epoxide opening reactions that join fragments of similar complexity.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "epoxide_opening")
        self.epoxide_pattern = Chem.MolFromSmarts("[O1][C][C]1")  # Three-membered epoxide ring
        
    def route_scoring(self, x) -> float:
        """
        Score based on depth of convergent coupling.
        Earlier convergent steps (lower depth) are preferred.
        """
        if x < 0:
            return 0  # No convergent coupling found
        else:
            return 1 - x  # Earlier coupling is better (higher score)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling via epoxide opening.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
            
            # Check if we have the expected number of major fragments
            if len(reactants) != self.fragment_count:
                return False
            
            # Check for epoxide opening: epoxide in reactants but not in product
            has_epoxide_reactant = any(mol.HasSubstructMatch(self.epoxide_pattern) for mol in reactants)
            has_epoxide_product = product.HasSubstructMatch(self.epoxide_pattern)
            
            if not (has_epoxide_reactant and not has_epoxide_product):
                return False
            
            # Check that fragments are of similar complexity (atom count within 50% of each other)
            atom_counts = [mol.GetNumAtoms() for mol in reactants]
            if len(atom_counts) >= 2:
                min_atoms = min(atom_counts)
                max_atoms = max(atom_counts)
                if min_atoms > 0 and (max_atoms / min_atoms) > 2.0:
                    return False  # Fragments too different in size
            
            # Check that each fragment contributes significant complexity (>5 atoms)
            if any(count < 5 for count in atom_counts):
                return False
                
            return True
            
        except Exception:
            return False
