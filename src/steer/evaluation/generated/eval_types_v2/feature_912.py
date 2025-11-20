"""Generated evaluation code for: Terminal ester deprotection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TerminalEsterDeprotection(BaseScoring):
    """
    Checks if ethyl ester deprotection occurs as the final step in the synthesis route.
    Scores based on whether the deprotection happens at the terminal position (depth = 1).
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config.get("protecting_group", "ethyl_ester")
        self.deprotection_timing = config.get("deprotection_timing", "final_step")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Deprotection doesn't happen
        elif x == 1.0:
            return 1.0  # Perfect - deprotection at final step
        else:
            return max(0, 1 - abs(x - 1.0) * 5)  # Penalty for non-terminal deprotection
    
    def hit_condition(self, d):
        """Check if this reaction is an ethyl ester deprotection"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactant = rxn_parts[0]
        products = rxn_parts[1].split(".")
        
        try:
            reactant_mol = Chem.MolFromSmiles(reactant)
            product_mols = [Chem.MolFromSmiles(p) for p in products if p]
            
            if not reactant_mol or not all(product_mols):
                return False
                
            # Check for ethyl ester in reactant
            ethyl_ester_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3][CH2:4][CH3:5]")
            if not reactant_mol.HasSubstructMatch(ethyl_ester_pattern):
                return False
                
            # Check for carboxylic acid formation in products
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[OH:3]")
            has_carboxylic_acid = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in product_mols)
            
            # Check for ethanol or ethoxide as leaving group
            ethanol_pattern = Chem.MolFromSmarts("[CH2:1][CH3:2]")
            has_ethyl_leaving = any(mol.HasSubstructMatch(ethanol_pattern) for mol in product_mols)
            
            return has_carboxylic_acid and has_ethyl_leaving
            
        except:
            return False
