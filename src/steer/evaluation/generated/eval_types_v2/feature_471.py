"""Generated evaluation code for: Weinreb amide for ketone synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WeinrebAmideKetone(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Weinreb amide ketone formation reactions.
    
    Weinreb amides (N-methoxy-N-methylamides) are useful intermediates that react with
    organometallic reagents to form ketones without over-addition issues that plague
    other ketone-forming reactions.
    """
    
    def __init__(self, config: Dict):
        self.weinreb_amide_smarts = config["parameters"]["functional_group_smarts"]
        self.weinreb_pattern = Chem.MolFromSmarts(self.weinreb_amide_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Weinreb amide reaction not found
        else:
            return 1 - x  # Earlier use is better (more strategic)
    
    def hit_condition(self, d):
        """
        Check if a reaction involves Weinreb amide to ketone conversion.
        This requires detecting the Weinreb amide pattern in reactants and 
        a ketone formation in the product.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Check if any reactant contains Weinreb amide pattern
            has_weinreb_amide = any(
                reactant.HasSubstructMatch(self.weinreb_pattern) 
                for reactant in reactants if reactant is not None
            )
            
            if not has_weinreb_amide:
                return False
            
            # Check if product has ketone and reactant Weinreb amide is consumed
            ketone_pattern = Chem.MolFromSmarts("C(=O)C")
            has_ketone_product = product is not None and product.HasSubstructMatch(ketone_pattern)
            
            # Verify Weinreb amide is consumed (not present in product)
            weinreb_consumed = product is None or not product.HasSubstructMatch(self.weinreb_pattern)
            
            return has_weinreb_amide and has_ketone_product and weinreb_consumed
            
        except (KeyError, AttributeError, ValueError):
            return False
