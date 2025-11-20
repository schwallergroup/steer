"""Generated evaluation code for: Late piperazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperazineFormation(BaseScoring):
    """
    Evaluates synthesis routes based on when piperazine ring formation occurs.
    Rewards late-stage piperazine ring formation in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CNCCN1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is better, so return 1 - depth_fraction
            # This gives higher scores for reactions that occur later in the route
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents piperazine ring formation.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        # Products (left side of reaction)
        products = rxn_parts[0]
        # Reactants (right side of reaction)  
        reactants = rxn_parts[1]
        
        try:
            # Check if product contains piperazine ring
            product_mol = Chem.MolFromSmiles(products)
            if product_mol is None:
                return False
                
            piperazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if piperazine_pattern is None:
                return False
                
            product_has_piperazine = product_mol.HasSubstructMatch(piperazine_pattern)
            
            # Check if any reactant lacks the piperazine ring
            reactant_smiles_list = reactants.split(".")
            reactant_has_piperazine = False
            
            for reactant_smiles in reactant_smiles_list:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol is not None and reactant_mol.HasSubstructMatch(piperazine_pattern):
                    reactant_has_piperazine = True
                    break
            
            # Ring formation occurs if product has piperazine but reactants don't
            return product_has_piperazine and not reactant_has_piperazine
            
        except Exception:
            return False
