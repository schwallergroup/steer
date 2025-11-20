"""Generated evaluation code for: Mitsunobu stereochemical inversion strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MitsunobuStereoinversion(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Mitsunobu reactions used for stereochemical inversion.
    The Mitsunobu reaction converts alcohols to various nucleophile-substituted products with inversion 
    of stereochemistry, commonly used to transform cis alcohols to trans products.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Mitsunobu reaction not found
        else:
            # Earlier use of stereoinversion is generally better for synthetic efficiency
            return 1 - x

    def hit_condition(self, d) -> bool:
        """
        Detects Mitsunobu reaction by looking for characteristic patterns:
        - Alcohol (C-OH) to C-N bond formation
        - Presence of typical Mitsunobu reagents or byproducts
        - Stereochemical inversion pattern
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for alcohol to amine/amide transformation
            alcohol_pattern = Chem.MolFromSmarts("[C:1][OH:2]")
            c_n_pattern = Chem.MolFromSmarts("[C:1][N:2]")
            
            # Look for alcohol in reactants
            has_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) for mol in reactants)
            
            # Look for C-N bond formation in products
            has_c_n = any(mol.HasSubstructMatch(c_n_pattern) for mol in products)
            
            # Check for typical Mitsunobu reagent patterns
            pph3_pattern = Chem.MolFromSmarts("P(c1ccccc1)(c2ccccc2)(c3ccccc3)")  # PPh3
            dead_pattern = Chem.MolFromSmarts("N1C(=O)C=CC1=O")  # DEAD/DIAD diethyl azodicarboxylate
            diad_pattern = Chem.MolFromSmarts("N(C(C)C)N=NC(C(=O)OCC)C")  # DIAD pattern
            
            mitsunobu_reagents = [pph3_pattern, dead_pattern, diad_pattern]
            has_mitsunobu_reagent = any(
                any(mol.HasSubstructMatch(reagent) for mol in reactants)
                for reagent in mitsunobu_reagents if reagent is not None
            )
            
            # Additional check for nucleophile (amine, azide, etc.)
            nucleophile_patterns = [
                Chem.MolFromSmarts("[NH2:1]"),  # Primary amine
                Chem.MolFromSmarts("[NH:1]"),   # Secondary amine  
                Chem.MolFromSmarts("[N-:1]=[N+:2]=[N-:3]"),  # Azide
                Chem.MolFromSmarts("O[C:1](=O)[NH:2]")  # Carbamate
            ]
            
            has_nucleophile = any(
                any(mol.HasSubstructMatch(pattern) for mol in reactants)
                for pattern in nucleophile_patterns if pattern is not None
            )
            
            # Mitsunobu reaction identified if:
            # 1. Alcohol present in reactants
            # 2. C-N bond formed in products
            # 3. Either Mitsunobu reagents present OR nucleophile present
            return has_alcohol and has_c_n and (has_mitsunobu_reagent or has_nucleophile)
            
        except Exception:
            return False
