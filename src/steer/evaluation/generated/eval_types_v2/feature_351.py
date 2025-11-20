"""Generated evaluation code for: Late imidazopyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ImidazopyridineRingFormation(BaseScoring):
    """
    Evaluates when imidazopyridine ring formation occurs in the synthesis route.
    Rewards late-stage formation of the imidazopyridine fused ring system.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Late-stage formation is better (closer to 1.0)
            else:  # early
                return x  # Early-stage formation is better (closer to 0.0)
    
    def hit_condition(self, d) -> bool:
        """
        Checks if imidazopyridine ring formation occurs in this reaction step.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if self.direction == "formation":
                # Check if imidazopyridine ring is absent in reactants but present in products
                reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                product_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                
                return not reactant_has_ring and product_has_ring
            
            else:  # breaking
                # Check if imidazopyridine ring is present in reactants but absent in products
                reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                product_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                
                return reactant_has_ring and not product_has_ring
                
        except Exception:
            return False
