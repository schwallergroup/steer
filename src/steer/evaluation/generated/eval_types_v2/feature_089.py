"""Generated evaluation code for: Late pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrazoleFormation(BaseScoring):
    """
    Evaluates routes based on the timing of pyrazole ring formation.
    Rewards routes where pyrazole rings are formed late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Late-stage formation is better (closer to 1.0)
            else:  # early timing
                return x  # Early-stage formation is better (closer to 0.0)
    
    def hit_condition(self, d) -> bool:
        """Check if pyrazole ring formation occurs in this reaction step."""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        
        # Parse molecules
        product = Chem.MolFromSmiles(product_smiles)
        if product is None:
            return False
        
        reactants = []
        for r_smiles in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol is not None:
                reactants.append(mol)
        
        if not reactants:
            return False
        
        # Count pyrazole rings in product and reactants
        product_rings = len(product.GetSubstructMatches(self.ring_pattern))
        reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactants)
        
        if self.direction == "formation":
            # Ring formation: more rings in product than reactants
            return product_rings > reactant_rings
        else:  # "breaking"
            # Ring breaking: fewer rings in product than reactants
            return product_rings < reactant_rings
