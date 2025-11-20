"""Generated evaluation code for: Curtius rearrangement for aniline formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CurtiusRearrangement(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Curtius rearrangement reactions.
    The Curtius rearrangement converts carboxylic acids to amines via acyl azide 
    intermediates, typically used for aniline formation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            return 1 - abs(x - self.target_depth) / 10
    
    def hit_condition(self, d):
        """
        Detects Curtius rearrangement by looking for:
        1. Carboxylic acid substrate pattern
        2. Primary amine product (especially aniline-like)
        3. Carbon count decrease by 1 (CO2 loss)
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # SMARTS patterns
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")  # Carboxylic acid
            primary_amine_pattern = Chem.MolFromSmarts("[NH2]")  # Primary amine
            aniline_pattern = Chem.MolFromSmarts("c[NH2]")  # Aniline (aromatic primary amine)
            
            # Check for carboxylic acid in reactants
            has_carboxylic_acid = any(
                mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in reactants
            )
            
            # Check for primary amine (preferably aniline) in products
            has_primary_amine = any(
                mol.HasSubstructMatch(primary_amine_pattern) for mol in products
            )
            
            has_aniline = any(
                mol.HasSubstructMatch(aniline_pattern) for mol in products
            )
            
            # Additional check: carbon count should decrease (CO2 loss)
            reactant_carbons = sum(
                sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
                for mol in reactants
            )
            
            product_carbons = sum(
                sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
                for mol in products
            )
            
            carbon_decrease = reactant_carbons > product_carbons
            
            # Curtius rearrangement signature:
            # - Has carboxylic acid substrate
            # - Produces primary amine (bonus for aniline)
            # - Carbon count decreases (CO2 elimination)
            if has_carboxylic_acid and has_primary_amine and carbon_decrease:
                return True
            
            # Alternative check for known Curtius intermediates/reagents
            # Look for azide-related patterns in reactants
            azide_pattern = Chem.MolFromSmarts("[N-]=[N+]=[N-]")  # Azide anion
            acyl_azide_pattern = Chem.MolFromSmarts("[C](=O)[N]=[N+]=[N-]")  # Acyl azide
            
            has_azide_reagent = any(
                mol.HasSubstructMatch(azide_pattern) or mol.HasSubstructMatch(acyl_azide_pattern)
                for mol in reactants
            )
            
            if has_azide_reagent and has_primary_amine:
                return True
                
            return False
            
        except Exception:
            return False
