"""Generated evaluation code for: Early pyrrolopyridine core construction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrrolopyridineConstruction(BaseScoring):
    """
    Evaluates whether the pyrrolopyridine core (c1ccn2ccnc2c1) is formed early in the synthesis route.
    Rewards routes where this bicyclic core is constructed in early stages before functional group modifications.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        For early timing: lower depth (earlier) gets higher score.
        """
        if x < 0:
            return 0  # Core formation doesn't happen
        
        # Early formation is rewarded - invert the depth fraction
        # x=0 (very early) -> score=10, x=1 (very late) -> score=0
        return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms the pyrrolopyridine core.
        Returns True if the core is absent in reactants but present in product.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            
            # Parse product
            product = Chem.MolFromSmiles(product_smiles)
            if product is None:
                return False
            
            # Check if product contains the pyrrolopyridine core
            product_has_core = product.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_core:
                return False
            
            # Parse reactants
            reactant_smiles_list = reactants_smiles.split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles_list]
            reactants = [r for r in reactants if r is not None]
            
            # Check if any reactant already contains the complete core
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Core already exists, not a formation reaction
            
            # Core is formed in this step (present in product, absent in all reactants)
            return True
            
        except (KeyError, ValueError, AttributeError):
            return False
