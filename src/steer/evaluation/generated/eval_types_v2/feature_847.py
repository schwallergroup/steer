"""Generated evaluation code for: Late stage thiazole assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageThiazoleAssembly(BaseScoring):
    """
    Evaluates synthesis routes based on late-stage thiazole ring formation.
    Detects when a thiazole ring (c1scnc1) is formed and scores based on 
    how late in the synthesis this occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.formation_stage = config["parameters"]["formation_stage"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole formation doesn't happen
        else:
            # Late-stage formation is better - higher depth gives higher score
            return x * 10  # Convert depth fraction to 0-10 scale, favoring late stage
    
    def hit_condition(self, d) -> bool:
        """
        Checks if thiazole ring formation occurs in this reaction step.
        Returns True if the product contains thiazole but reactants don't.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            prod_smiles = rxn[0]
            react_smiles = rxn[1].split(".")
            
            # Parse molecules
            product = Chem.MolFromSmiles(prod_smiles)
            if product is None:
                return False
                
            reactants = []
            for r_smiles in react_smiles:
                reactant = Chem.MolFromSmiles(r_smiles)
                if reactant is not None:
                    reactants.append(reactant)
            
            if not reactants:
                return False
            
            # Check if product has thiazole ring
            product_has_thiazole = product.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_thiazole:
                return False
            
            # Check if any reactant already has the complete thiazole ring
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Thiazole already present, not a formation
            
            # Thiazole is in product but not in reactants - this is formation
            return True
            
        except (KeyError, IndexError, AttributeError):
            return False
