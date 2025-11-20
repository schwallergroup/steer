"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclopropaneFormation(BaseScoring):
    """
    Evaluates routes based on late-stage cyclopropane ring formation.
    Looks for reactions that form cyclopropane rings (C1CC1) and scores
    based on how late in the synthesis this occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No cyclopropane formation found
        
        if self.timing == "late":
            # Late stage is better, so higher depth fraction gets higher score
            return x * 10
        elif self.timing == "early":
            # Early stage is better, so lower depth fraction gets higher score
            return (1 - x) * 10
        else:
            # Any stage is acceptable
            return 10 if x >= 0 else 0
    
    def hit_condition(self, d):
        """
        Check if this reaction involves cyclopropane ring formation.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        if len(rxn) != 2:
            return False
            
        # Parse reactants and products
        reactants_smiles = rxn[0]
        products_smiles = rxn[1]
        
        try:
            # Count cyclopropane rings in reactants
            reactant_cyclopropanes = 0
            for reactant_smiles in reactants_smiles.split("."):
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol:
                    reactant_cyclopropanes += len(reactant_mol.GetSubstructMatches(self.ring_pattern))
            
            # Count cyclopropane rings in products
            product_cyclopropanes = 0
            for product_smiles in products_smiles.split("."):
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol:
                    product_cyclopropanes += len(product_mol.GetSubstructMatches(self.ring_pattern))
            
            # Check if cyclopropane rings were formed (more in products than reactants)
            if self.direction == "formation":
                return product_cyclopropanes > reactant_cyclopropanes
            elif self.direction == "breaking":
                return reactant_cyclopropanes > product_cyclopropanes
            else:
                return product_cyclopropanes != reactant_cyclopropanes
                
        except Exception:
            return False
