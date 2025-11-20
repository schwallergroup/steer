"""Generated evaluation code for: Early furan to pyridine ring transformation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyFuranToPyridineTransformation(BaseScoring):
    """
    Evaluates whether a furan to pyridine ring transformation occurs early in the synthesis route.
    Returns higher scores when the transformation happens within the early stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"]["stage_threshold"]
        self.furan_pattern = Chem.MolFromSmarts("c1ccoc1")  # Furan ring
        self.pyridine_pattern = Chem.MolFromSmarts("c1ccncc1")  # Pyridine ring
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Transformation doesn't occur
        
        # Score based on how early the transformation occurs
        # x is the depth fraction where transformation happens
        if x <= self.stage_threshold:
            return 10  # Maximum score for early transformation
        else:
            # Linearly decrease score for later transformations
            return max(0, 10 * (1 - x) / (1 - self.stage_threshold))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves a furan to pyridine transformation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = Chem.MolFromSmiles(rxn_parts[0])
            reactants_smiles = rxn_parts[1].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not products or not reactants:
                return False
            
            # Check if product contains pyridine ring
            has_pyridine_in_product = products.HasSubstructMatch(self.pyridine_pattern)
            
            # Check if any reactant contains furan ring
            has_furan_in_reactants = any(
                reactant.HasSubstructMatch(self.furan_pattern) 
                for reactant in reactants if reactant
            )
            
            # Check if reactants lack pyridine (to ensure formation, not just presence)
            lacks_pyridine_in_reactants = not any(
                reactant.HasSubstructMatch(self.pyridine_pattern)
                for reactant in reactants if reactant
            )
            
            # Transformation occurs if: furan in reactants + pyridine formed in product
            return (has_furan_in_reactants and 
                   has_pyridine_in_product and 
                   lacks_pyridine_in_reactants)
                   
        except Exception:
            return False
