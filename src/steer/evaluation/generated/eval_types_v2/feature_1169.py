"""Generated evaluation code for: Late indole ring formation via Fischer synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIndoleFischerSynthesis(BaseScoring):
    """
    Evaluates synthesis routes for late-stage indole ring formation via Fischer indole synthesis.
    Checks for the formation of indole rings (c1ccc2[nH]ccc2c1) in the later stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.indole_smarts = config["parameters"]["ring_smarts"]  # "c1ccc2[nH]ccc2c1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.method = config["parameters"]["method"]  # "Fischer indole synthesis"
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        For late-stage formation, lower depth fractions (closer to target) are better.
        """
        if x < 0:
            return 0  # Indole formation doesn't occur
        else:
            # Late-stage formation is rewarded (closer to 1.0 is better)
            return 10 * x  # Linear scoring where x=1.0 gives max score of 10
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents Fischer indole synthesis forming an indole ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse molecules
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
                
            # Check if indole ring is formed (present in products but not in reactants)
            indole_pattern = Chem.MolFromSmarts(self.indole_smarts)
            if not indole_pattern:
                return False
                
            # Count indole rings in products and reactants
            indole_in_products = sum(1 for mol in products if mol.HasSubstructMatch(indole_pattern))
            indole_in_reactants = sum(1 for mol in reactants if mol.HasSubstructMatch(indole_pattern))
            
            # Check if indole ring is newly formed (more in products than reactants)
            if indole_in_products > indole_in_reactants:
                # Additional check for Fischer indole synthesis pattern
                return self._is_fischer_indole_synthesis(reactants, products)
                
        except Exception:
            return False
            
        return False
        
    def _is_fischer_indole_synthesis(self, reactants, products) -> bool:
        """
        Check if the reaction pattern matches Fischer indole synthesis.
        Fischer synthesis typically involves phenylhydrazine + ketone/aldehyde -> indole + NH3 + H2O
        """
        # Look for phenylhydrazine pattern in reactants
        phenylhydrazine_pattern = Chem.MolFromSmarts("c1ccccc1NN")
        if not phenylhydrazine_pattern:
            return False
            
        # Look for carbonyl pattern (ketone/aldehyde) in reactants
        carbonyl_pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
        if not carbonyl_pattern:
            return False
            
        has_phenylhydrazine = any(mol.HasSubstructMatch(phenylhydrazine_pattern) for mol in reactants)
        has_carbonyl = any(mol.HasSubstructMatch(carbonyl_pattern) for mol in reactants)
        
        return has_phenylhydrazine and has_carbonyl
