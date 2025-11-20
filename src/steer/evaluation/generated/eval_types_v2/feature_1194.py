"""Generated evaluation code for: Late stage sulfonamide formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSulfonamideFormation(BaseScoring):
    """
    Evaluates whether sulfonamide formation occurs at a late stage in the synthesis route.
    Detects the formation of sulfonamide bonds (S-N) from sulfonyl chlorides and amines.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sulfonamide formation doesn't occur
        else:
            # Late-stage formation is better (higher depth fraction preferred)
            if self.condition_type == "bool":
                return 10 if x >= self.target_depth else 0
            else:
                # Score based on how close to late-stage (depth = 1.0)
                return max(0, 10 * (x - 0.2))  # Reward formations after 20% depth
    
    def hit_condition(self, d):
        """
        Detects sulfonamide formation by checking for:
        1. Sulfonyl chloride reactant (R-SO2-Cl)
        2. Amine reactant (R-NH2 or R-NH-R')
        3. Sulfonamide product (R-SO2-NH-R')
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for sulfonyl chloride pattern in reactants
            sulfonyl_chloride_pattern = Chem.MolFromSmarts("[S](=O)(=O)[Cl]")
            has_sulfonyl_chloride = any(mol.HasSubstructMatch(sulfonyl_chloride_pattern) 
                                     for mol in reactants)
            
            # Check for amine pattern in reactants (primary or secondary amine)
            primary_amine_pattern = Chem.MolFromSmarts("[N;H2]")
            secondary_amine_pattern = Chem.MolFromSmarts("[N;H1]")
            has_amine = any(mol.HasSubstructMatch(primary_amine_pattern) or 
                          mol.HasSubstructMatch(secondary_amine_pattern)
                          for mol in reactants)
            
            # Check for sulfonamide pattern in products
            sulfonamide_pattern = Chem.MolFromSmarts("[S](=O)(=O)[N]")
            has_sulfonamide_product = any(mol.HasSubstructMatch(sulfonamide_pattern) 
                                        for mol in products)
            
            return has_sulfonyl_chloride and has_amine and has_sulfonamide_product
            
        except Exception:
            return False
