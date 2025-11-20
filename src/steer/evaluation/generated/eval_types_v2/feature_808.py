"""Generated evaluation code for: Late stage cyclopropanation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation reactions.
    Detects when a specific ring structure is formed and rewards routes
    where this formation occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage preference: higher depth fraction = higher score
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return x * 10  # Later formation gets higher score
            elif self.timing == "early":
                return (1 - x) * 10  # Earlier formation gets higher score
            else:
                return 5  # Neutral if no timing preference
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node involves the formation of the target ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if ring is present in product
            product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if self.direction == "formation":
                # Ring formation: ring in product but not in any single reactant
                if not product_has_ring:
                    return False
                
                # Check if ring is already present in reactants
                for reactant in reactant_mols:
                    if reactant.HasSubstructMatch(self.ring_pattern):
                        return False  # Ring already exists, not a formation
                
                return True  # Ring in product but not in reactants = formation
                
            elif self.direction == "breaking":
                # Ring breaking: ring in reactants but not in product
                if product_has_ring:
                    return False
                
                # Check if ring is present in any reactant
                for reactant in reactant_mols:
                    if reactant.HasSubstructMatch(self.ring_pattern):
                        return True  # Ring in reactant but not product = breaking
                
                return False
                
        except Exception:
            return False
        
        return False
