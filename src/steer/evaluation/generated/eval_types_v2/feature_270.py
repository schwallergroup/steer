"""Generated evaluation code for: Late oxadiazolone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxadiazoloneFormation(BaseScoring):
    """
    Evaluates whether oxadiazolone ring formation occurs late in the synthesis route.
    
    Detects the formation of 1,2,4-oxadiazol-5-one rings and scores based on 
    how late in the route the ring formation occurs, with later formation 
    being preferred.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Score based on depth fraction where ring formation occurs.
        Later formation (higher depth fraction) gets better score.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late timing, higher depth fraction is better
            if self.timing == "late":
                return x * 10  # Scale to 0-10 range
            else:
                return (1 - x) * 10  # For early timing preference
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves oxadiazolone ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None values from failed parsing
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Count oxadiazolone rings in reactants and products
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactants)
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in products)
            
            # Check for ring formation (more rings in products than reactants)
            if self.direction == "formation":
                return product_rings > reactant_rings
            elif self.direction == "break":
                return reactant_rings > product_rings
            else:
                return product_rings != reactant_rings
                
        except (KeyError, ValueError, AttributeError):
            return False
