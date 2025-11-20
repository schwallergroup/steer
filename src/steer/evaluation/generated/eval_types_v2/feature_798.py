"""Generated evaluation code for: Chiral auxiliary stereoinduction strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChiralAuxiliaryStereoinduction(BaseScoring):
    """
    Evaluates synthesis routes for the use of chiral auxiliary stereoinduction strategy.
    
    Specifically looks for (R)-phenylethylamine chiral auxiliary attached to lactam nitrogen
    for controlling stereochemistry in alkylation reactions.
    """
    
    def __init__(self, config: Dict):
        self.auxiliary_type = config.get("auxiliary_type", "chiral_auxiliary")
        self.location = config.get("location", "nitrogen")
        self.purpose = config.get("purpose", "stereoinduction")
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Strategy not found
        else:
            # Earlier use of chiral auxiliary is generally better
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves chiral auxiliary stereoinduction"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Define SMARTS patterns for chiral auxiliary components
            phenylethylamine_pattern = Chem.MolFromSmarts("[CH3][CH]([NH2,NH1])[c]1[cH][cH][cH][cH][cH]1")
            lactam_nitrogen_pattern = Chem.MolFromSmarts("[NH1]C(=O)")
            chiral_center_pattern = Chem.MolFromSmarts("[C@H1,C@@H1]")
            
            # Check for presence of chiral auxiliary attachment/detachment
            auxiliary_in_reactants = any(
                mol.HasSubstructMatch(phenylethylamine_pattern) 
                for mol in reactants if mol is not None
            )
            
            auxiliary_in_products = any(
                mol.HasSubstructMatch(phenylethylamine_pattern) 
                for mol in products if mol is not None
            )
            
            # Check for lactam nitrogen involvement
            lactam_in_reaction = any(
                mol.HasSubstructMatch(lactam_nitrogen_pattern) 
                for mol in reactants + products if mol is not None
            )
            
            # Check for chiral center formation/modification
            chiral_centers_reactants = sum(
                len(mol.GetSubstructMatches(chiral_center_pattern)) 
                for mol in reactants if mol is not None
            )
            
            chiral_centers_products = sum(
                len(mol.GetSubstructMatches(chiral_center_pattern)) 
                for mol in products if mol is not None
            )
            
            # Chiral auxiliary stereoinduction criteria:
            # 1. Auxiliary present in reaction
            # 2. Lactam nitrogen involved
            # 3. Change in stereochemistry (chiral center formation/modification)
            has_auxiliary = auxiliary_in_reactants or auxiliary_in_products
            has_stereochemical_change = chiral_centers_products > chiral_centers_reactants
            
            # Additional check for alkylation pattern (C-C bond formation at alpha position)
            alkylation_pattern = Chem.MolFromSmarts("[NH1]C(=O)[CH2,CH1][CH2,CH1]")
            has_alkylation = any(
                mol.HasSubstructMatch(alkylation_pattern) 
                for mol in products if mol is not None
            )
            
            return has_auxiliary and lactam_in_reaction and (has_stereochemical_change or has_alkylation)
            
        except Exception:
            return False
