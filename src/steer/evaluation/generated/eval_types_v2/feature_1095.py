"""Generated evaluation code for: Late cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclopropaneFormationTiming(BaseScoring):
    """
    Evaluates the timing of cyclopropane ring formation in synthesis routes.
    Rewards late-stage cyclopropane formation as a key strategic choice.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation gets higher score
            elif self.timing == "early":
                return x  # Earlier formation gets higher score
            else:
                return 0.5  # Neutral scoring if timing not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves cyclopropane ring formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if self.direction == "formation":
                # Check if cyclopropane is formed (absent in reactants, present in products)
                reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                product_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                
                return not reactant_has_ring and product_has_ring
                
            elif self.direction == "breaking":
                # Check if cyclopropane is broken (present in reactants, absent in products)
                reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                product_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                
                return reactant_has_ring and not product_has_ring
                
        except Exception:
            return False
            
        return False
