"""Generated evaluation code for: Early quinolone core assembly via Gould-Jacobs reaction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class QuinoloneCoreAssembly(BaseScoring):
    """
    Checks for early quinolone core assembly via Gould-Jacobs reaction.
    Detects formation of quinolone heterocycle through condensation reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.quinolone_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "early":
            # Early formation is better - penalize later depths
            return max(0, 10 * (1 - x))
        else:
            # Standard depth-based scoring
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a quinolone core via condensation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains quinolone core
            if not product.HasSubstructMatch(self.quinolone_pattern):
                return False
            
            # Check if quinolone core is absent in all reactants (formation reaction)
            quinolone_in_reactants = any(r.HasSubstructMatch(self.quinolone_pattern) for r in reactants)
            if quinolone_in_reactants:
                return False
            
            # Check for condensation pattern (formation of N-C and C=O bonds)
            if self.formation_method == "condensation":
                return self._is_condensation_reaction(reactants, product)
            
            return True
            
        except Exception:
            return False
    
    def _is_condensation_reaction(self, reactants, product):
        """Check if reaction involves condensation typical of Gould-Jacobs"""
        # Look for typical Gould-Jacobs reactants: aniline derivative + β-dicarbonyl
        has_aniline = False
        has_carbonyl_compound = False
        
        aniline_pattern = Chem.MolFromSmarts("c1ccc(N)cc1")  # Aniline core
        beta_dicarbonyl_pattern = Chem.MolFromSmarts("C(=O)CC(=O)")  # β-dicarbonyl
        ester_pattern = Chem.MolFromSmarts("C(=O)O[CH3,CH2]")  # Ester group
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(aniline_pattern):
                has_aniline = True
            if (reactant.HasSubstructMatch(beta_dicarbonyl_pattern) or 
                reactant.HasSubstructMatch(ester_pattern)):
                has_carbonyl_compound = True
        
        return has_aniline and has_carbonyl_compound
