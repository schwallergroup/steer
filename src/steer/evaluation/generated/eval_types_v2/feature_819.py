"""Generated evaluation code for: Late cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateCyclopropaneFormation(BaseScoring):
    """
    Evaluates whether cyclopropane ring formation occurs late in the synthesis route.
    Detects cyclopropane formation by checking if a cyclopropane ring is present in 
    the product but absent in all reactants.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CC1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        For late formation: later is better, so higher depth gives higher score.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late formation is better - depth closer to 1 gives higher score
            return x * 10
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves cyclopropane ring formation.
        Returns True if cyclopropane is present in product but absent in all reactants.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            if product is None:
                return False
                
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    reactant_mols.append(mol)
            
            if not reactant_mols:
                return False
                
            # Create cyclopropane pattern
            cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if cyclopropane_pattern is None:
                return False
                
            # Check if product has cyclopropane
            product_has_cyclopropane = product.HasSubstructMatch(cyclopropane_pattern)
            
            if not product_has_cyclopropane:
                return False
                
            # Check if any reactant has cyclopropane
            reactants_have_cyclopropane = any(
                mol.HasSubstructMatch(cyclopropane_pattern) for mol in reactant_mols
            )
            
            # Ring formation occurs if product has it but no reactant has it
            return not reactants_have_cyclopropane
            
        except (KeyError, ValueError, AttributeError):
            return False
