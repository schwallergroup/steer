"""Generated evaluation code for: Late indole ring formation via Fischer synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIndoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage indole ring formation via Fischer synthesis.
    Checks if the indole ring system is formed in the later stages of the synthesis,
    preferring routes where complex fragments are assembled in the final steps.
    """
    
    def __init__(self, config):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x):
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation gets higher score (closer to 1)
            else:
                return x  # Earlier formation gets higher score
    
    def hit_condition(self, d):
        """
        Check if this reaction involves indole ring formation.
        Returns True if indole is formed in this step.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        if len(rxn) != 2:
            return False
            
        reactants_smiles = rxn[0]
        products_smiles = rxn[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Create indole pattern
            indole_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if indole_pattern is None:
                return False
            
            # Check if indole is present in products but not in reactants
            indole_in_reactants = any(mol.HasSubstructMatch(indole_pattern) for mol in reactants)
            indole_in_products = any(mol.HasSubstructMatch(indole_pattern) for mol in products)
            
            if self.direction == "formation":
                # Ring formation: indole present in products but not in reactants
                return indole_in_products and not indole_in_reactants
            else:
                # Ring breaking: indole present in reactants but not in products
                return indole_in_reactants and not indole_in_products
                
        except Exception:
            return False
