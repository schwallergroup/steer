"""Generated evaluation code for: Early quinoline ring formation via Gould-Jacobs"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyQuinolineGJ(BaseScoring):
    """
    Evaluates early quinoline ring formation via Gould-Jacobs reaction.
    Checks if quinoline ring formation occurs at or before the specified step depth.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_step = config["parameters"]["formation_step"]
        self.quinoline_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Quinoline formation not detected
        
        if self.timing == "early":
            # Reward formation at or before target step
            if x <= self.formation_step / 10.0:  # Convert step to depth fraction
                return 10  # Maximum score for early formation
            else:
                # Penalize later formation
                penalty = (x - self.formation_step / 10.0) * 20
                return max(0, 10 - penalty)
        
        return 10 - (x * 10)  # General case: earlier is better
    
    def hit_condition(self, d):
        """
        Detects quinoline ring formation by checking if quinoline appears in 
        reactants but not in all individual product fragments (indicating cyclization).
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check if any product contains quinoline
            quinoline_in_products = any(mol.HasSubstructMatch(self.quinoline_pattern) for mol in products)
            
            # Check if any reactant contains quinoline
            quinoline_in_reactants = any(mol.HasSubstructMatch(self.quinoline_pattern) for mol in reactants)
            
            # Quinoline formation detected if it appears in products but not in reactants
            # OR if this looks like a Gould-Jacobs type reaction (cyclization)
            if quinoline_in_products and not quinoline_in_reactants:
                return True
            
            # Additional check for Gould-Jacobs signature: 
            # Look for intramolecular cyclization patterns
            if quinoline_in_products:
                return self._is_gould_jacobs_pattern(reactants, products)
                
        except Exception:
            return False
        
        return False
    
    def _is_gould_jacobs_pattern(self, reactants, products):
        """
        Additional heuristic to detect Gould-Jacobs cyclization pattern.
        Looks for typical precursors and cyclization signatures.
        """
        # Common Gould-Jacobs precursor patterns
        aniline_pattern = Chem.MolFromSmarts("c1ccc(N)cc1")  # Aniline derivative
        ester_pattern = Chem.MolFromSmarts("C(=O)OC")  # Ester group
        
        # Check if reactants contain typical GJ precursors
        has_aniline = any(mol.HasSubstructMatch(aniline_pattern) for mol in reactants)
        has_ester = any(mol.HasSubstructMatch(ester_pattern) for mol in reactants)
        
        # If we have typical precursors and quinoline formation, likely GJ reaction
        return has_aniline and has_ester
