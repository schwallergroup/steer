"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring structure is formed at a late stage in the synthesis.
    
    Checks if the target ring (defined by SMARTS pattern) is present in the product
    but absent in at least one reactant, indicating ring formation occurred in that step.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        
        # Compile the SMARTS pattern for efficiency
        from rdkit import Chem
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Later formation is better (closer to 1.0 is better)
            return 10 * (1 - x)  # Scale to 0-10
        elif self.timing == "early":
            # Earlier formation is better (closer to 0.0 is better)  
            return 10 * x  # Scale to 0-10
        else:
            # If no timing preference, just reward presence
            return 10
    
    def hit_condition(self, d):
        """
        Check if the specified ring formation occurs in this reaction step.
        """
        from rdkit import Chem
        
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        # Parse molecules
        product = Chem.MolFromSmiles(product_smiles)
        if product is None:
            return False
            
        reactants = []
        for r_smiles in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol is not None:
                reactants.append(mol)
        
        if not reactants:
            return False
        
        # Check if ring formation occurred
        if self.direction == "formation":
            # Product should contain the ring
            product_has_ring = product.HasSubstructMatch(self.ring_pattern)
            if not product_has_ring:
                return False
                
            # At least one reactant should NOT contain the ring
            reactant_missing_ring = any(not r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            return reactant_missing_ring
            
        elif self.direction == "breaking":
            # At least one reactant should contain the ring
            reactant_has_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            if not reactant_has_ring:
                return False
                
            # Product should NOT contain the ring
            product_missing_ring = not product.HasSubstructMatch(self.ring_pattern)
            return product_missing_ring
            
        return False
