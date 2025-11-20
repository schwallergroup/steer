"""Generated evaluation code for: Selective mono-Boc protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SelectiveMonoBocProtection(BaseScoring):
    """
    Evaluates routes for selective mono-Boc protection of diamines.
    Checks if the route performs selective protection of one amine group
    while leaving another amine unprotected.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "normalized")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        if self.condition_type == "bool":
            return 1  # Condition met
        else:
            # Earlier selective protection is generally better
            return max(0, 1 - x)
    
    def hit_condition(self, d):
        """Check if this reaction performs selective mono-Boc protection"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".") if smi.strip()]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".") if smi.strip()]
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check for Boc reagent in reactants
            boc_patterns = [
                "[C:1](=O)([O:2][C:3]([CH3:4])([CH3:5])[CH3:6])[O:7][C:8]([CH3:9])([CH3:10])[CH3:11]",  # Boc2O
                "C(=O)(OC(C)(C)C)OC(C)(C)C",  # Boc2O pattern
                "[NH2:1][C:2](=O)[O:3][C:4]([CH3:5])([CH3:6])[CH3:7]"  # Boc-NH2
            ]
            
            has_boc_reagent = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for mol in reactant_mols)
                for pattern in boc_patterns
            )
            
            if not has_boc_reagent:
                return False
            
            # Find the main substrate (largest molecule, not Boc reagent)
            main_reactant = None
            main_product = None
            
            for mol in reactant_mols:
                if mol.GetNumAtoms() > 5 and not any(mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in boc_patterns):
                    main_reactant = mol
                    break
            
            for mol in product_mols:
                if mol.GetNumAtoms() > 5:
                    main_product = mol
                    break
            
            if not main_reactant or not main_product:
                return False
            
            # Count primary and secondary amines in reactant and product
            primary_amine_pattern = "[NX3H2:1][CX4:2]"
            secondary_amine_pattern = "[NX3H1:1]([CX4:2])[CX4:3]"
            boc_protected_amine_pattern = "[NX3:1]([CX4:2])[C:3](=O)[O:4][C:5]([CH3:6])([CH3:7])[CH3:8]"
            
            # Count free amines in reactant
            reactant_primary_amines = len(main_reactant.GetSubstructMatches(Chem.MolFromSmarts(primary_amine_pattern)))
            reactant_secondary_amines = len(main_reactant.GetSubstructMatches(Chem.MolFromSmarts(secondary_amine_pattern)))
            reactant_total_amines = reactant_primary_amines + reactant_secondary_amines
            
            # Count free and protected amines in product
            product_primary_amines = len(main_product.GetSubstructMatches(Chem.MolFromSmarts(primary_amine_pattern)))
            product_secondary_amines = len(main_product.GetSubstructMatches(Chem.MolFromSmarts(secondary_amine_pattern)))
            product_boc_amines = len(main_product.GetSubstructMatches(Chem.MolFromSmarts(boc_protected_amine_pattern)))
            product_total_amines = product_primary_amines + product_secondary_amines + product_boc_amines
            
            # Check for selective mono-Boc protection:
            # 1. At least 2 amines in starting material
            # 2. Exactly 1 Boc group added
            # 3. At least 1 free amine remains
            is_selective_mono_boc = (
                reactant_total_amines >= 2 and  # Diamine or polyamine substrate
                product_boc_amines == 1 and     # Exactly one Boc group added
                (product_primary_amines + product_secondary_amines) >= 1 and  # At least one free amine remains
                product_total_amines == reactant_total_amines  # Total amine count preserved
            )
            
            return is_selective_mono_boc
            
        except Exception:
            return False
