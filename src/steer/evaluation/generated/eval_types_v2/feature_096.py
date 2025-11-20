"""Generated evaluation code for: Benzyl ester protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEsterProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes for the use of benzyl ester protecting groups.
    Checks if benzyl ester protection/deprotection strategy is employed for carboxylic acids.
    Returns higher scores when the strategy is used at appropriate depths.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # SMARTS patterns for benzyl ester protection/deprotection
        self.benzyl_ester_pattern = Chem.MolFromSmarts("[CH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[OH:8][C:9](=[O:10])[*:11]>>[CH2:1]([c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1)[O:8][C:9](=[O:10])[*:11]")
        self.benzyl_ester_formation = Chem.MolFromSmarts("C(=O)OCc1ccccc1")  # Benzyl ester product
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")  # Free carboxylic acid
        self.benzyl_chloride_pattern = Chem.MolFromSmarts("ClCc1ccccc1")  # Common benzyl source
        self.benzyl_bromide_pattern = Chem.MolFromSmarts("BrCc1ccccc1")  # Alternative benzyl source
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0  # Protection strategy not found
            # Earlier protection (lower depth) is generally better
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction involves benzyl ester protection or deprotection
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        
        # Check for protection reaction (formation of benzyl ester)
        if self._is_protection_reaction(rxn_smiles):
            return True
            
        # Check for deprotection reaction (hydrogenolysis)
        if self._is_deprotection_reaction(rxn_smiles):
            return True
            
        return False
    
    def _is_protection_reaction(self, rxn_smiles: str) -> bool:
        """Check if reaction forms benzyl ester from carboxylic acid"""
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if reactants contain carboxylic acid and benzyl halide/alcohol
            has_carboxylic_acid = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactants)
            has_benzyl_source = any(mol.HasSubstructMatch(self.benzyl_chloride_pattern) or 
                                  mol.HasSubstructMatch(self.benzyl_bromide_pattern) for mol in reactants)
            
            # Check if products contain benzyl ester
            has_benzyl_ester = any(mol.HasSubstructMatch(self.benzyl_ester_formation) for mol in products)
            
            return has_carboxylic_acid and has_benzyl_source and has_benzyl_ester
            
        except Exception:
            return False
    
    def _is_deprotection_reaction(self, rxn_smiles: str) -> bool:
        """Check if reaction removes benzyl ester to form carboxylic acid (hydrogenolysis)"""
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if reactants contain benzyl ester
            has_benzyl_ester = any(mol.HasSubstructMatch(self.benzyl_ester_formation) for mol in reactants)
            
            # Check if products contain free carboxylic acid
            has_carboxylic_acid = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in products)
            
            # Check for toluene or benzyl alcohol as byproduct (typical in hydrogenolysis)
            toluene_pattern = Chem.MolFromSmarts("Cc1ccccc1")
            benzyl_alcohol_pattern = Chem.MolFromSmarts("OCc1ccccc1")
            has_benzyl_byproduct = any(mol.HasSubstructMatch(toluene_pattern) or 
                                     mol.HasSubstructMatch(benzyl_alcohol_pattern) for mol in products)
            
            return has_benzyl_ester and has_carboxylic_acid and has_benzyl_byproduct
            
        except Exception:
            return False
