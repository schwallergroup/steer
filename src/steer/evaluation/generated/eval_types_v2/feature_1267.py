"""Generated evaluation code for: Gabriel synthesis for amine installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class GabrielSynthesis(BaseScoring):
    """
    Evaluates presence of Gabriel synthesis for amine installation.
    Detects phthalimide protection/deprotection sequence for primary amine formation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Gabriel synthesis not found
        else:
            # Earlier use of Gabriel synthesis is better (closer to target)
            return 1 - x
    
    def hit_condition(self, d):
        """
        Detects Gabriel synthesis by looking for:
        1. Phthalimide deprotection (phthalimide -> primary amine)
        2. Formation of primary amine from phthalimide derivative
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Phthalimide SMARTS pattern
            phthalimide_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]2:[#6](:[#6]:1)C(=O)N(*)C2=O")
            # Primary amine pattern (not amide)
            primary_amine_pattern = Chem.MolFromSmarts("[NX3H2][CX4]")
            
            if phthalimide_pattern is None or primary_amine_pattern is None:
                return False
            
            # Check for phthalimide in reactants
            has_phthalimide_reactant = any(
                mol.HasSubstructMatch(phthalimide_pattern) for mol in reactants
            )
            
            # Check for primary amine in products
            has_primary_amine_product = any(
                mol.HasSubstructMatch(primary_amine_pattern) for mol in products
            )
            
            # Gabriel synthesis: phthalimide derivative -> primary amine
            if has_phthalimide_reactant and has_primary_amine_product:
                return True
            
            # Also check for Gabriel alkylation (phthalimide + alkyl halide)
            alkyl_halide_pattern = Chem.MolFromSmarts("[CX4][F,Cl,Br,I]")
            phthalimide_anion_pattern = Chem.MolFromSmarts("c1ccc2c(c1)C(=O)N([K,Na])C2=O")
            
            if alkyl_halide_pattern is None or phthalimide_anion_pattern is None:
                return False
            
            has_alkyl_halide = any(
                mol.HasSubstructMatch(alkyl_halide_pattern) for mol in reactants
            )
            has_phthalimide_anion = any(
                mol.HasSubstructMatch(phthalimide_anion_pattern) for mol in reactants
            )
            
            # Check if product contains N-alkyl phthalimide
            n_alkyl_phthalimide_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]2:[#6](:[#6]:1)C(=O)N([CX4])C2=O")
            
            if n_alkyl_phthalimide_pattern is None:
                return False
            
            has_n_alkyl_phthalimide_product = any(
                mol.HasSubstructMatch(n_alkyl_phthalimide_pattern) for mol in products
            )
            
            # Gabriel alkylation step
            if (has_phthalimide_anion or has_phthalimide_reactant) and has_alkyl_halide and has_n_alkyl_phthalimide_product:
                return True
                
            return False
            
        except Exception:
            return False
