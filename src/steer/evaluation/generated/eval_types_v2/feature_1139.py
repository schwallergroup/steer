"""Generated evaluation code for: Early complex multicomponent pyrazole formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for early pyrazole ring formation.
    Checks if a pyrazole ring (c1ccnn1) is formed early in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ccnn1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrazole formation doesn't happen
        else:
            # Early formation is better, so return higher score for smaller depth fractions
            if self.timing == "early":
                return 1 - x  # x is depth fraction, so 1-x rewards early occurrence
            else:
                return x  # Late formation would be rewarded with x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents pyrazole ring formation.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        # Parse molecules
        try:
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
                
        except:
            return False
        
        # Check if product contains pyrazole ring
        product_has_pyrazole = prod_mol.HasSubstructMatch(self.ring_pattern)
        
        # Check if any reactant contains pyrazole ring
        reactants_have_pyrazole = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
        
        # Ring formation: product has pyrazole but reactants don't
        if self.direction == "formation":
            return product_has_pyrazole and not reactants_have_pyrazole
        # Ring breaking: reactants have pyrazole but product doesn't  
        elif self.direction == "breaking":
            return reactants_have_pyrazole and not product_has_pyrazole
        
        return False
