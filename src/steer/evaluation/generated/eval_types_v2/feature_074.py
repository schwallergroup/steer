"""Generated evaluation code for: Early pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazoleRingFormation(BaseScoring):
    """
    Evaluates early pyrazole ring formation in synthesis routes.
    
    This class detects when pyrazole rings (c1ccnn1) are formed during synthesis
    and scores routes based on early formation timing. Higher scores are given
    when pyrazole formation occurs early in the synthesis sequence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)

    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For early formation: lower depth fraction = higher score
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "early":
            # Early formation preferred: score decreases with depth
            return max(0, 10 * (1 - x))
        else:
            # Late formation preferred: score increases with depth
            return max(0, 10 * x)

    def hit_condition(self, d) -> bool:
        """
        Check if pyrazole ring formation occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        # Parse reactants and products
        try:
            product_mol = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) 
                           for r in reactants_smiles.split(".") if r.strip()]
            
            if not product_mol or not reactant_mols:
                return False
                
        except Exception:
            return False
        
        if self.direction == "formation":
            # Check if pyrazole ring is formed (present in product but not in reactants)
            product_has_pyrazole = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_pyrazole:
                return False
                
            # Check if any reactant already has the pyrazole ring
            reactants_have_pyrazole = any(mol.HasSubstructMatch(self.ring_pattern) 
                                        for mol in reactant_mols if mol)
            
            # Ring formation occurs if product has pyrazole but reactants don't
            return product_has_pyrazole and not reactants_have_pyrazole
            
        elif self.direction == "break":
            # Check if pyrazole ring is broken (present in reactants but not in product)
            product_has_pyrazole = product_mol.HasSubstructMatch(self.ring_pattern)
            reactants_have_pyrazole = any(mol.HasSubstructMatch(self.ring_pattern) 
                                        for mol in reactant_mols if mol)
            
            # Ring breaking occurs if reactants have pyrazole but product doesn't
            return reactants_have_pyrazole and not product_has_pyrazole
            
        return False
