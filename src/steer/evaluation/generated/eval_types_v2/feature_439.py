"""Generated evaluation code for: Late stage alcohol deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAlcoholDeprotection(BaseScoring):
    """
    Evaluates if ethoxyethyl acetal protecting group removal from alcohol occurs at late stage.
    Rewards routes where the deprotection happens closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.protection_type = config.get("protection_type", "ethoxyethyl_acetal")
        self.timing_preference = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score. Late stage deprotection gets higher score."""
        if x < 0:
            return 0  # Deprotection doesn't happen
        else:
            if self.timing_preference == "late":
                return 1 - x  # Late-stage deprotection is better (lower depth fraction)
            else:
                return x  # Early-stage deprotection is better (higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves ethoxyethyl acetal deprotection to reveal alcohol."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        reactant_smiles = rxn_parts[0]
        product_smiles = rxn_parts[1]
        
        try:
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
            product_mol = Chem.MolFromSmiles(product_smiles)
            
            if not reactant_mol or not product_mol:
                return False
            
            # Ethoxyethyl acetal pattern: R-O-CH(CH3)-O-CH2-CH3
            ethoxyethyl_acetal_pattern = Chem.MolFromSmarts("[#6]-[#8]-[CH1]([CH3])-[#8]-[CH2]-[CH3]")
            
            # Free alcohol pattern
            alcohol_pattern = Chem.MolFromSmarts("[#6]-[OH1]")
            
            # Check if reactant has ethoxyethyl acetal and product has corresponding free alcohol
            reactant_has_acetal = reactant_mol.HasSubstructMatch(ethoxyethyl_acetal_pattern)
            product_has_alcohol = product_mol.HasSubstructMatch(alcohol_pattern)
            
            # Additional check: ensure the acetal is removed (fewer acetal groups in product)
            reactant_acetal_count = len(reactant_mol.GetSubstructMatches(ethoxyethyl_acetal_pattern))
            product_acetal_count = len(product_mol.GetSubstructMatches(ethoxyethyl_acetal_pattern))
            
            return (reactant_has_acetal and 
                   product_has_alcohol and 
                   product_acetal_count < reactant_acetal_count)
                   
        except Exception:
            return False
