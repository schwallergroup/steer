"""Generated evaluation code for: Early Claisen condensation for fragment assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyClaisen(BaseScoring):
    """
    Evaluates routes for early-stage Claisen condensation reactions.
    Checks for the presence of Claisen condensation (formation of 1,3-dicarbonyl compounds)
    and rewards when it occurs early in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "early")
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Claisen condensation doesn't happen
        else:
            # Early-stage condensation is better (lower depth gets higher score)
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Claisen condensation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for formation of 1,3-dicarbonyl pattern (key feature of Claisen condensation)
            dicarbonyl_patterns = [
                "[C:1](=[O:2])[CH2:3][C:4](=[O:5])",  # 1,3-diketone
                "[C:1](=[O:2])[CH2:3][C:4](=[O:5])[OH]",  # beta-ketocarboxylic acid
                "[C:1](=[O:2])[CH2:3][C:4](=[O:5])[O]",   # beta-ketoester
            ]
            
            # Check if product contains 1,3-dicarbonyl and reactants don't
            product_has_dicarbonyl = any(
                product.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                for pattern in dicarbonyl_patterns
            )
            
            reactants_have_dicarbonyl = any(
                any(reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                    for pattern in dicarbonyl_patterns)
                for reactant in reactants
            )
            
            # Also check for typical Claisen reactant patterns
            ester_pattern = "[C:1](=[O:2])[O:3][C:4]"  # Ester
            ketone_pattern = "[C:1](=[O:2])[CH2,CH3:3]"  # Ketone/methyl ketone
            
            has_ester_reactant = any(
                reactant.HasSubstructMatch(Chem.MolFromSmarts(ester_pattern))
                for reactant in reactants
            )
            
            has_carbonyl_reactant = any(
                reactant.HasSubstructMatch(Chem.MolFromSmarts(ketone_pattern))
                for reactant in reactants
            )
            
            # Claisen condensation: forms 1,3-dicarbonyl from ester + carbonyl compound
            return (product_has_dicarbonyl and 
                    not reactants_have_dicarbonyl and
                    has_ester_reactant and 
                    has_carbonyl_reactant)
                    
        except Exception:
            return False
