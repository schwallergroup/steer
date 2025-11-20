"""Generated evaluation code for: Late piperazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperazineRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on when piperazine ring formation occurs.
    Rewards routes where the piperazine ring (C1CNCCN1) is formed late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CNCCN1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late stage formation is better (higher score for higher depth fraction)
            if self.timing == "late":
                return x * 10  # Scale to 0-10, where 1.0 (latest) = 10 points
            else:  # early
                return (1 - x) * 10  # Scale to 0-10, where 0.0 (earliest) = 10 points
                
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves piperazine ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not products or not all(reactant_mols):
                return False
                
            # Create pattern for piperazine ring
            piperazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not piperazine_pattern:
                return False
                
            # Check if piperazine is present in products
            has_piperazine_in_products = products.HasSubstructMatch(piperazine_pattern)
            
            # Check if piperazine is present in any reactant
            has_piperazine_in_reactants = any(mol.HasSubstructMatch(piperazine_pattern) 
                                            for mol in reactant_mols if mol is not None)
            
            if self.direction == "formation":
                # Ring formation: not in reactants but present in products
                return has_piperazine_in_products and not has_piperazine_in_reactants
            else:  # "breaking"
                # Ring breaking: present in reactants but not in products
                return has_piperazine_in_reactants and not has_piperazine_in_products
                
        except Exception:
            return False
