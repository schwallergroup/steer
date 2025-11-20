"""Generated evaluation code for: Late stage N-aryl bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NArylBondFormation(BaseScoring):
    """
    Evaluates the timing of N-aryl bond formation in synthesis routes.
    Detects when an N-aryl bond (defined by SMARTS pattern) is formed
    and scores based on how late in the synthesis this occurs.
    Late-stage N-aryl bond formation is preferred as it avoids carrying
    the N-aryl group through multiple synthetic steps.
    """
    
    def __init__(self, config):
        self.bond_smarts = config.get("bond_smarts", "[N;R]-[c;R]")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        self.bond_pattern = Chem.MolFromSmarts(self.bond_smarts)
    
    def route_scoring(self, x):
        if x < 0:
            return 0  # N-aryl bond formation doesn't occur
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (higher score)
            elif self.timing == "early":
                return x  # Earlier formation is better
            else:
                return 0.5  # Neutral scoring if timing not specified
    
    def hit_condition(self, d):
        """
        Check if N-aryl bond formation occurs in this reaction step.
        Returns True if the bond is absent in reactants but present in product.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            if product is None:
                return False
            
            # Check if product contains the N-aryl bond pattern
            product_has_bond = product.HasSubstructMatch(self.bond_pattern)
            
            if not product_has_bond:
                return False
            
            # Check if any reactant already has this bond (if so, it's not formation)
            reactant_smiles_list = reactants_smiles.split(".")
            for reactant_smiles in reactant_smiles_list:
                reactant = Chem.MolFromSmiles(reactant_smiles)
                if reactant is not None and reactant.HasSubstructMatch(self.bond_pattern):
                    return False  # Bond already exists in reactant
            
            # Bond is present in product but absent in all reactants -> formation occurred
            return True
            
        except (KeyError, ValueError, AttributeError):
            return False
