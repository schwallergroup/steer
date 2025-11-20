"""Generated evaluation code for: Late stage thiazole formation via Hantzsch synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageThiazoleFormation(BaseScoring):
    """
    Evaluates routes for late-stage thiazole formation via Hantzsch synthesis.
    
    Checks if a thiazole ring (c1scnc1) is formed late in the synthesis route
    through Hantzsch synthesis involving thioamide and chloroacetaldehyde reactants.
    Returns higher scores for later formation of the thiazole ring.
    """
    
    def __init__(self, config: Dict):
        self.thiazole_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.reaction_type = config["parameters"]["reaction_type"]  # "hantzsch_thiazole"
        self.stage = config["parameters"]["stage"]  # "late"
        
        # Compile the thiazole pattern
        self.thiazole_pattern = Chem.MolFromSmarts(self.thiazole_smarts)
        
        # SMARTS patterns for Hantzsch thiazole synthesis reactants
        self.thioamide_pattern = Chem.MolFromSmarts("[#6][C](=[S])[N]")  # thioamide
        self.chloroacetaldehyde_pattern = Chem.MolFromSmarts("[Cl][CH2][CH]=O")  # chloroacetaldehyde
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Late stage formation (higher x) gets higher scores.
        """
        if x < 0:
            return 0  # Thiazole formation doesn't happen
        else:
            # Late stage formation is better, so higher depth fraction = higher score
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a thiazole ring via Hantzsch synthesis.
        
        Args:
            d: Dictionary containing reaction metadata
            
        Returns:
            bool: True if this is a thiazole-forming Hantzsch reaction
        """
        try:
            # Get the mapped reaction SMILES
            mapped_rxn = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if thiazole ring is formed (present in product but not in reactants)
            product_has_thiazole = product.HasSubstructMatch(self.thiazole_pattern)
            
            if not product_has_thiazole:
                return False
            
            # Check if thiazole is absent in all reactants (indicating formation)
            reactants_have_thiazole = any(r.HasSubstructMatch(self.thiazole_pattern) for r in reactants)
            
            if reactants_have_thiazole:
                return False  # Thiazole already present, not forming
            
            # Check for Hantzsch synthesis pattern: thioamide + chloroacetaldehyde
            has_thioamide = any(r.HasSubstructMatch(self.thioamide_pattern) for r in reactants)
            has_chloroacetaldehyde = any(r.HasSubstructMatch(self.chloroacetaldehyde_pattern) for r in reactants)
            
            # Alternative check: look for any carbonyl + thioamide pattern
            carbonyl_pattern = Chem.MolFromSmarts("[CH2][C]=O")
            has_carbonyl = any(r.HasSubstructMatch(carbonyl_pattern) for r in reactants)
            
            # Return True if we have thiazole formation with appropriate Hantzsch reactants
            return has_thioamide and (has_chloroacetaldehyde or has_carbonyl)
            
        except Exception:
            return False
