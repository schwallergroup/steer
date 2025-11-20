"""Generated evaluation code for: Late cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateCyclopropaneFormation(BaseScoring):
    """
    Evaluates routes based on late-stage cyclopropane ring formation.
    Rewards routes where cyclopropane rings are formed closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't happen
        else:
            # Late-stage formation is better (lower depth fraction = higher score)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves cyclopropane ring formation.
        """
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Create substructure pattern for cyclopropane
            cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if cyclopropane_pattern is None:
                return False
            
            # Count cyclopropane rings in reactants and products
            reactant_cyclopropanes = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                                       for mol in reactants if mol is not None)
            product_cyclopropanes = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                                      for mol in products if mol is not None)
            
            # Check if cyclopropane ring was formed (more in products than reactants)
            return product_cyclopropanes > reactant_cyclopropanes
            
        except Exception:
            return False
