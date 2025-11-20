"""Generated evaluation code for: Late stage Cbz deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCbzDeprotection(BaseScoring):
    """
    Evaluates whether Cbz (benzyloxycarbonyl) deprotection occurs at a late stage in the synthesis.
    Cbz protecting group removal should happen near the end of the route to reveal the free amine.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Late stage default
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cbz deprotection doesn't occur
        
        if self.condition_type == "bool":
            # Reward late-stage deprotection (depth > 0.8)
            return 1 if x > 0.8 else 0
        else:
            # Penalize early deprotection, reward late deprotection
            if x < self.target_depth:
                return 10 * (1 - abs(x - self.target_depth))
            else:
                return 10 * (1 - (x - self.target_depth) * 2)
    
    def hit_condition(self, d):
        """Check if this reaction involves Cbz deprotection"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Define Cbz protecting group pattern
            cbz_pattern = Chem.MolFromSmarts("[CH2:1][c:2]1[c:3][c:4][c:5][c:6][c:7]1")  # Benzyl part
            cbz_carbamate_pattern = Chem.MolFromSmarts("O=C(O[CH2][c:1]1[c:2][c:3][c:4][c:5][c:6]1)N")  # Full Cbz group
            
            if cbz_pattern is None or cbz_carbamate_pattern is None:
                return False
            
            # Check if reactants contain Cbz-protected amine
            has_cbz_reactant = any(mol.HasSubstructMatch(cbz_carbamate_pattern) for mol in reactants)
            
            # Check if products contain free amine and benzyl alcohol/CO2 byproducts
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1,NH0]")
            benzyl_alcohol_pattern = Chem.MolFromSmarts("[CH2:1][c:2]1[c:3][c:4][c:5][c:6][c:7]1")
            
            has_free_amine = any(mol.HasSubstructMatch(free_amine_pattern) for mol in products)
            has_benzyl_product = any(mol.HasSubstructMatch(benzyl_alcohol_pattern) for mol in products)
            
            # Cbz deprotection: Cbz-protected amine -> free amine + benzyl-containing byproduct
            return has_cbz_reactant and has_free_amine and has_benzyl_product
            
        except Exception:
            return False
