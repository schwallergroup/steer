"""Generated evaluation code for: Early stage Wittig methylenation approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyWittigMethylenation(BaseScoring):
    """
    Evaluates whether a Wittig methylenation reaction occurs early in the synthesis route.
    
    Detects Wittig reactions by identifying the characteristic phosphonium ylide pattern
    and C=C double bond formation, then checks if it occurs before the stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        """
        Score based on how early the Wittig reaction occurs.
        Early reactions (before threshold) get higher scores.
        """
        if x < 0:
            return 0  # Wittig reaction not found
        
        if x <= self.stage_threshold:
            return 10 * (1 - x)  # Earlier is better, max score 10
        else:
            return max(0, 5 * (1 - x))  # Late reactions get lower scores
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a Wittig methylenation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for Wittig reaction characteristics
            return self._is_wittig_methylenation(product, reactants)
            
        except Exception:
            return False
    
    def _is_wittig_methylenation(self, product, reactants) -> bool:
        """
        Detect Wittig methylenation by checking for:
        1. Phosphonium ylide reactant (P-C bond)
        2. Carbonyl reactant (C=O)
        3. Alkene product formation (C=C)
        4. Methylenation pattern (=CH2 or =CHR)
        """
        # Phosphonium ylide patterns
        phosphonium_patterns = [
            "[P+]([#6])([#6])([#6])[CH2-]",  # Phosphonium methylide
            "[P+]([#6])([#6])([#6])[CH-]",   # Phosphonium alkylidene
            "P([#6])([#6])([#6])=C",         # Ylide form
        ]
        
        # Carbonyl pattern
        carbonyl_pattern = "[CX3]=[OX1]"
        
        # Terminal alkene patterns (methylenation products)
        alkene_patterns = [
            "C=C",           # General alkene
            "[CH2]=C",       # Terminal methylene
            "C=[CH2]",       # Terminal methylene (other direction)
        ]
        
        # Check for phosphonium ylide in reactants
        has_phosphonium = False
        has_carbonyl = False
        
        for reactant in reactants:
            # Check for phosphonium ylide
            for p_pattern in phosphonium_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(p_pattern)):
                    has_phosphonium = True
                    break
            
            # Check for carbonyl
            if reactant.HasSubstructMatch(Chem.MolFromSmarts(carbonyl_pattern)):
                has_carbonyl = True
        
        # Check for alkene formation in product
        has_alkene = False
        for a_pattern in alkene_patterns:
            if product.HasSubstructMatch(Chem.MolFromSmarts(a_pattern)):
                has_alkene = True
                break
        
        # Additional check: verify no carbonyl in product at the reaction site
        # (carbonyl should be consumed to form alkene)
        product_carbonyls = len(product.GetSubstructMatches(Chem.MolFromSmarts(carbonyl_pattern)))
        reactant_carbonyls = sum(len(r.GetSubstructMatches(Chem.MolFromSmarts(carbonyl_pattern))) 
                               for r in reactants)
        
        carbonyl_consumed = reactant_carbonyls > product_carbonyls
        
        return has_phosphonium and has_carbonyl and has_alkene and carbonyl_consumed
