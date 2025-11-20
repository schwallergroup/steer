"""Generated evaluation code for: Late lactam reduction to amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateLactamReduction(BaseScoring):
    """
    Evaluates whether lactam reduction to amine occurs late in the synthesis route.
    Checks for amide reduction reactions where the substrate contains a lactam ring.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["step_number"]
        self.total_steps = config["parameters"]["total_steps"]
        self.target_depth_fraction = self.target_step / self.total_steps
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        # Score based on how close to target depth (late stage preferred)
        depth_diff = abs(x - self.target_depth_fraction)
        return max(0, 1 - depth_diff * 5)  # Scale to 0-1 range
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a lactam reduction to amine"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactant_smiles, product_smiles = rxn_smiles.split(">>")
        
        # Check if reactant contains lactam and product contains corresponding amine
        reactant_mol = Chem.MolFromSmiles(reactant_smiles)
        product_mol = Chem.MolFromSmiles(product_smiles)
        
        if not reactant_mol or not product_mol:
            return False
            
        # Define lactam patterns (4-7 membered rings with N-C=O)
        lactam_patterns = [
            "[NH1]C(=O)C[CH2]",  # 4-membered lactam
            "[NH1]C(=O)CC[CH2]",  # 5-membered lactam  
            "[NH1]C(=O)CCC[CH2]",  # 6-membered lactam
            "[NH1]C(=O)CCCC[CH2]"  # 7-membered lactam
        ]
        
        # Check if reactant has lactam
        has_lactam = any(reactant_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                        for pattern in lactam_patterns)
        
        if not has_lactam:
            return False
            
        # Check if this is an amide reduction (C=O -> CH2 with N preservation)
        # Look for reduction pattern: amide carbonyl disappears, N remains
        reactant_amide_pattern = Chem.MolFromSmarts("[NH1]C(=O)")
        product_amine_pattern = Chem.MolFromSmarts("[NH1]C")
        
        has_reactant_amide = reactant_mol.HasSubstructMatch(reactant_amide_pattern)
        has_product_amine = product_mol.HasSubstructMatch(product_amine_pattern)
        
        # Additional check: carbonyl should be reduced
        reactant_carbonyls = len(reactant_mol.GetSubstructMatches(Chem.MolFromSmarts("C=O")))
        product_carbonyls = len(product_mol.GetSubstructMatches(Chem.MolFromSmarts("C=O")))
        
        return (has_reactant_amide and has_product_amine and 
                reactant_carbonyls > product_carbonyls)
