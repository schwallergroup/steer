"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """Evaluates if a specific ring is formed late in the synthesis route."""
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation is better (x close to 0 gives high score)
        elif self.timing == "early":
            return x  # Earlier formation is better (x close to 1 gives high score)
        else:
            return 1.0 if x >= 0 else 0.0  # Just presence/absence
    
    def hit_condition(self, d):
        """Check if the specified ring formation occurs in this reaction step."""
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        # Parse molecules
        try:
            prod_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            # Remove None molecules (parsing failures)
            prod_mols = [m for m in prod_mols if m is not None]
            react_mols = [m for m in react_mols if m is not None]
            
        except:
            return False
        
        if self.direction == "formation":
            # Check if ring is present in products but not in reactants
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in prod_mols)
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in react_mols)
            
            return ring_in_products and not ring_in_reactants
            
        elif self.direction == "breaking":
            # Check if ring is present in reactants but not in products
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in prod_mols)
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in react_mols)
            
            return ring_in_reactants and not ring_in_products
            
        return False
