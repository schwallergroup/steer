"""Generated evaluation code for: Late stage coumarin ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCoumarin(BaseScoring):
    """
    Evaluates whether coumarin ring formation occurs late in the synthesis route.
    Looks for the formation of the coumarin core structure (benzopyran-2-one) 
    and rewards when this key ring formation happens in later stages.
    """
    
    def __init__(self, config: Dict):
        self.coumarin_smarts = "c1ccc2ccc(=O)oc2c1"  # coumarin core pattern
        self.coumarin_mol = Chem.MolFromSmarts(self.coumarin_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Score based on depth fraction where coumarin formation occurs.
        Late stage formation (higher x) gets better score.
        """
        if x < 0:
            return 0  # Coumarin formation doesn't happen
        else:
            return x * 10  # Later formation gets higher score (0-10 range)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves coumarin ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse product and reactants
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains coumarin core
            product_has_coumarin = product.HasSubstructMatch(self.coumarin_mol)
            
            if not product_has_coumarin:
                return False
            
            # Check if any reactant already has the complete coumarin core
            reactants_have_coumarin = any(r.HasSubstructMatch(self.coumarin_mol) for r in reactants)
            
            # Ring formation occurs if product has coumarin but reactants don't
            return not reactants_have_coumarin
            
        except (KeyError, ValueError, AttributeError):
            return False
