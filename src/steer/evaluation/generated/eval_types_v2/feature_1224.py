"""Generated evaluation code for: Early penicillin to cephalosporin ring expansion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PenicillinCephalosporinExpansion(BaseScoring):
    """
    Evaluates the timing of penicillin-to-cephalosporin ring expansion reactions.
    Checks when a penicillin beta-lactam ring is broken/expanded, typically
    occurring early in synthesis routes before side chain modifications.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early" or "late"
        self.direction = config["parameters"]["direction"]  # "break" or "form"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring expansion doesn't occur
        
        if self.timing == "early":
            return 1 - x  # Earlier is better, score decreases with depth
        else:  # late timing
            return x  # Later is better, score increases with depth
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves penicillin ring expansion."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        # Parse molecules
        try:
            products = [Chem.MolFromSmiles(products_smiles)]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            if not all(products + reactants):
                return False
                
        except Exception:
            return False
            
        # Create SMARTS pattern for penicillin core
        penicillin_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if not penicillin_pattern:
            return False
            
        # Check for ring expansion pattern
        if self.direction == "break":
            # Penicillin present in products, modified/absent in reactants
            product_has_penicillin = any(mol.HasSubstructMatch(penicillin_pattern) for mol in products if mol)
            reactant_has_penicillin = any(mol.HasSubstructMatch(penicillin_pattern) for mol in reactants if mol)
            
            # Ring expansion: penicillin core in product but modified in reactants
            return product_has_penicillin and not reactant_has_penicillin
            
        else:  # direction == "form"
            # Penicillin formed from precursors
            product_has_penicillin = any(mol.HasSubstructMatch(penicillin_pattern) for mol in products if mol)
            reactant_has_penicillin = any(mol.HasSubstructMatch(penicillin_pattern) for mol in reactants if mol)
            
            return product_has_penicillin and not reactant_has_penicillin
