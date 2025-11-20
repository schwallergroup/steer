"""Generated evaluation code for: Late stage piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation.
    Rewards routes where the specified ring structure is formed closer to the end of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late", "early", or specific depth
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        elif self.timing == "early":
            return x  # Earlier formation gets higher score
        else:
            # Specific timing target
            return 1 - abs(x - float(self.timing))
    
    def hit_condition(self, d) -> bool:
        """
        Detects if the target ring is formed (or broken) in this reaction step.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".") if smi]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".") if smi]
            
            if not all(reactants) or not all(products):
                return False
            
            # Create pattern matcher
            ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if ring_pattern is None:
                return False
            
            # Count ring occurrences in reactants and products
            reactant_matches = sum(len(mol.GetSubstructMatches(ring_pattern)) for mol in reactants)
            product_matches = sum(len(mol.GetSubstructMatches(ring_pattern)) for mol in products)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_matches > reactant_matches
            else:  # direction == "break"
                # Ring breaking: fewer rings in products than reactants
                return reactant_matches > product_matches
                
        except (KeyError, AttributeError, ValueError):
            return False
