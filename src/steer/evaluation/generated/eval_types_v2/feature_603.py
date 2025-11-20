"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage amide coupling reactions.
    Rewards routes where amide bond formation occurs near the end of the synthesis,
    typically between a macrocycle amine and carboxylic acid.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # No amide coupling found
        
        if self.timing_preference == "late":
            # Reward late-stage amide coupling (lower depth fractions are better)
            return 10 * (1 - x)
        else:
            # For early-stage preference (if needed)
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """Check if current reaction node contains amide coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            return self._is_amide_coupling(product, reactants)
            
        except Exception:
            return False
    
    def _is_amide_coupling(self, product, reactants) -> bool:
        """Detect if reaction represents amide bond formation"""
        # SMARTS pattern for amide bond
        amide_pattern = Chem.MolFromSmarts("[C](=O)[NH]")
        
        # Check if product contains amide bond
        if not product.HasSubstructMatch(amide_pattern):
            return False
            
        # Patterns for typical amide coupling reactants
        carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        
        # Alternative: activated carboxylic acid derivatives
        activated_acid_patterns = [
            Chem.MolFromSmarts("[C](=O)[Cl]"),  # Acid chloride
            Chem.MolFromSmarts("[C](=O)O[C](=O)"),  # Anhydride
            Chem.MolFromSmarts("[C](=O)[N]1C(=O)CCC1=O"),  # NHS ester
        ]
        
        has_amine = False
        has_acid_source = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(amine_pattern):
                has_amine = True
            
            if reactant.HasSubstructMatch(carboxylic_acid_pattern):
                has_acid_source = True
            else:
                # Check for activated acid derivatives
                for pattern in activated_acid_patterns:
                    if pattern and reactant.HasSubstructMatch(pattern):
                        has_acid_source = True
                        break
        
        return has_amine and has_acid_source
