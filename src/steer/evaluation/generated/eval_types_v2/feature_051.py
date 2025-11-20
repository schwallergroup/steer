"""Generated evaluation code for: TMS alkyne protecting group deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TMSAlkyneDeprotection(BaseScoring):
    """
    Evaluates synthesis routes based on TMS alkyne protecting group deprotection.
    Checks if a route uses trimethylsilyl protection for terminal alkyne with 
    subsequent deprotection using reagents like TBAF or K2CO3.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Deprotection doesn't happen
            return max(0, 1 - abs(x - self.target_depth))  # Closer to target is better
    
    def hit_condition(self, d):
        """Check if this reaction represents TMS alkyne deprotection"""
        metadata = d.get("metadata", {})
        
        # Check if reaction SMILES is available
        if "mapped_reaction_smiles" not in metadata:
            return False
        
        rxn_smiles = metadata["mapped_reaction_smiles"]
        
        try:
            # Split reaction into reactants and products
            reactant_part, product_part = rxn_smiles.split(">>")
            
            # Parse molecules
            reactant_mol = Chem.MolFromSmiles(reactant_part)
            product_mol = Chem.MolFromSmiles(product_part)
            
            if not reactant_mol or not product_mol:
                return False
            
            # Define TMS-protected alkyne pattern: C#C-Si(C)(C)C
            tms_alkyne_pattern = Chem.MolFromSmarts("C#C[Si](C)(C)C")
            
            # Define terminal alkyne pattern: C#C
            terminal_alkyne_pattern = Chem.MolFromSmarts("C#C")
            
            # Check if reactant has TMS-protected alkyne
            has_tms_alkyne_reactant = reactant_mol.HasSubstructMatch(tms_alkyne_pattern)
            
            # Check if product has terminal alkyne
            has_terminal_alkyne_product = product_mol.HasSubstructMatch(terminal_alkyne_pattern)
            
            # Check if product lacks TMS group (deprotection occurred)
            has_tms_alkyne_product = product_mol.HasSubstructMatch(tms_alkyne_pattern)
            
            # Additional check for common deprotection reagents in reaction context
            deprotection_reagents = ["TBAF", "K2CO3", "KF", "CsF", "Bu4NF"]
            has_deprotection_reagent = any(reagent.lower() in rxn_smiles.lower() 
                                         for reagent in deprotection_reagents)
            
            # Condition: reactant has TMS-alkyne, product has terminal alkyne but no TMS-alkyne
            is_deprotection = (has_tms_alkyne_reactant and 
                             has_terminal_alkyne_product and 
                             not has_tms_alkyne_product)
            
            return is_deprotection
            
        except Exception:
            return False
