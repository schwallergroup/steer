"""Generated evaluation code for: Baeyer-Villiger oxidation for phenol introduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BaeyerVilligerPhenolFormation(BaseScoring):
    """
    Detects Baeyer-Villiger oxidation reactions that convert aryl methyl ketones 
    to phenolic acetate functionality for phenol introduction.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)
    
    def hit_condition(self, d):
        """
        Detects Baeyer-Villiger oxidation by checking for:
        1. Aryl methyl ketone substrate (reactant)
        2. Phenolic acetate or phenol product
        3. Oxidizing conditions typical of B-V reactions
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn[1].split(".")]
            
            # Pattern for aryl methyl ketone
            aryl_ketone_pattern = Chem.MolFromSmarts("[cH:1][c:2]C(=O)C")
            
            # Pattern for phenolic acetate (acetylated phenol)
            phenolic_acetate_pattern = Chem.MolFromSmarts("[cH:1][c:2]OC(=O)C")
            
            # Pattern for phenol
            phenol_pattern = Chem.MolFromSmarts("[cH:1][c:2]O")
            
            # Check if reactants contain aryl methyl ketone
            has_aryl_ketone = any(
                mol.HasSubstructMatch(aryl_ketone_pattern) for mol in reactants if mol
            )
            
            # Check if products contain phenolic acetate or phenol
            has_phenolic_product = any(
                mol.HasSubstructMatch(phenolic_acetate_pattern) or 
                mol.HasSubstructMatch(phenol_pattern)
                for mol in products if mol
            )
            
            # Additional check for oxidizing agents typical in B-V reactions
            oxidant_patterns = [
                Chem.MolFromSmarts("C(=O)OO"),  # Peracetic acid
                Chem.MolFromSmarts("c1ccc(cc1)C(=O)OO"),  # Perbenzoic acid
                Chem.MolFromSmarts("CC(C)(C)OO"),  # tert-butyl hydroperoxide
            ]
            
            has_oxidant = any(
                any(mol.HasSubstructMatch(pattern) for mol in reactants if mol)
                for pattern in oxidant_patterns
            )
            
            return has_aryl_ketone and has_phenolic_product and has_oxidant
            
        except Exception:
            return False
