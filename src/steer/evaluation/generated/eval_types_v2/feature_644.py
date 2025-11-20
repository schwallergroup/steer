"""Generated evaluation code for: Late stage piperazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates late-stage ring formation events in synthesis routes.
    Detects when a specific ring pattern is formed and scores based on timing.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage formation, higher depth fractions are better.
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Higher score for later formation (closer to target)
            return x * 10
        elif self.timing == "early":
            # Higher score for earlier formation (closer to starting materials)
            return (1 - x) * 10
        else:
            # Neutral - any formation gets partial credit
            return 5.0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms the target ring structure.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if self.direction == "formation":
                # Check if ring is absent in reactants but present in products
                reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                products_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                
                return not reactants_have_ring and products_have_ring
                
            elif self.direction == "breaking":
                # Check if ring is present in reactants but absent in products
                reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                products_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                
                return reactants_have_ring and not products_have_ring
                
        except Exception:
            return False
            
        return False
