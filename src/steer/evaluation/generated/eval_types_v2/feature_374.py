"""Generated evaluation code for: Dual cyclopropanation strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualCyclopropanationStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on the presence of at least two cyclopropanation reactions.
    Returns a score from 0-10 where 10 indicates the target number of cyclopropanation 
    reactions are present in the route.
    """
    
    def __init__(self, config):
        self.min_count = config["parameters"].get("min_count", 2)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        cyclopropanation_count = sum(1 for r in reactions if self.detect_cyclopropanation(r))
        
        condition = cyclopropanation_count >= self.min_count
        return condition, cyclopropanation_count
    
    def detect_cyclopropanation(self, rxn):
        """
        Detects cyclopropanation reactions by identifying formation of cyclopropane rings.
        Checks if cyclopropane is present in products but not in reactants.
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol is not None:
                    reactants.append(mol)
            
            # Parse products  
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol is not None:
                    products.append(mol)
            
            # Cyclopropane pattern
            cyclopropane_pattern = Chem.MolFromSmarts("C1CC1")
            if cyclopropane_pattern is None:
                return False
            
            # Count cyclopropane rings in reactants and products
            reactant_cyclopropanes = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                                       for mol in reactants)
            product_cyclopropanes = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                                      for mol in products)
            
            # Cyclopropanation occurs if more cyclopropanes in products than reactants
            return product_cyclopropanes > reactant_cyclopropanes
            
        except Exception:
            return False
    
    def route_scoring(self, x):
        """
        Converts cyclopropanation count to 0-10 score.
        x is the number of cyclopropanation reactions found.
        """
        if x >= self.min_count:
            return 10.0  # Perfect score if minimum count met
        elif x == 0:
            return 0.0   # No score if no cyclopropanations found
        else:
            # Partial credit for some cyclopropanations
            return (x / self.min_count) * 10.0
