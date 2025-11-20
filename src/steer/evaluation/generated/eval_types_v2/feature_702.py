"""Generated evaluation code for: Late stage nitrile hydrolysis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileHydrolysis(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nitrile hydrolysis reactions.
    
    Detects nitrile to amide/carboxylic acid conversions and scores based on
    how late in the synthesis they occur. Earlier occurrence (lower depth)
    results in higher scores.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No nitrile hydrolysis found
        
        # For late-stage preference, lower depth (closer to target) is better
        if self.timing == "late":
            if x <= self.depth_threshold:
                return 10 * (1 - x / self.depth_threshold)  # Scale 10-0 for depths 0 to threshold
            else:
                return 0  # Too early in synthesis
        else:
            # For early-stage preference (if needed)
            return max(0, 10 * (x / 10))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents nitrile hydrolysis."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Define nitrile and hydrolysis product patterns
            nitrile_pattern = Chem.MolFromSmarts("[#6]#[#7]")  # C≡N
            amide_pattern = Chem.MolFromSmarts("[#6](=[#8])[#7]")  # C(=O)N (amide)
            carboxylic_acid_pattern = Chem.MolFromSmarts("[#6](=[#8])[#8H]")  # C(=O)OH (carboxylic acid)
            
            # Check if reactants contain nitrile
            has_nitrile_reactant = any(
                mol.HasSubstructMatch(nitrile_pattern) for mol in reactants
            )
            
            # Check if products contain amide or carboxylic acid
            has_hydrolysis_product = any(
                mol.HasSubstructMatch(amide_pattern) or mol.HasSubstructMatch(carboxylic_acid_pattern)
                for mol in products
            )
            
            # Additional check: ensure nitrile count decreases and amide/acid count increases
            if has_nitrile_reactant and has_hydrolysis_product:
                nitrile_count_reactants = sum(
                    len(mol.GetSubstructMatches(nitrile_pattern)) for mol in reactants
                )
                nitrile_count_products = sum(
                    len(mol.GetSubstructMatches(nitrile_pattern)) for mol in products
                )
                
                amide_acid_count_reactants = sum(
                    len(mol.GetSubstructMatches(amide_pattern)) + len(mol.GetSubstructMatches(carboxylic_acid_pattern))
                    for mol in reactants
                )
                amide_acid_count_products = sum(
                    len(mol.GetSubstructMatches(amide_pattern)) + len(mol.GetSubstructMatches(carboxylic_acid_pattern))
                    for mol in products
                )
                
                return (nitrile_count_reactants > nitrile_count_products and 
                       amide_acid_count_products > amide_acid_count_reactants)
            
            return False
            
        except Exception:
            return False
