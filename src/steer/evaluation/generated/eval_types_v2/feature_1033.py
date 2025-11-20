"""Generated evaluation code for: Late stage amide bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideBondFormation(BaseScoring):
    """
    Evaluates whether amide bond formation occurs late in the synthesis route.
    Rewards routes where amide coupling reactions happen within a specified 
    distance from the end of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.target_position = config["parameters"].get("position_from_end", 2)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        else:
            # Convert depth fraction to position from end
            # x is depth fraction (0 = root, 1 = leaf)
            position_from_end = 1 / x if x > 0 else float('inf')
            
            # Reward if within target position from end
            if position_from_end <= self.target_position:
                return 1.0  # Perfect score for late-stage
            else:
                # Penalize based on how early it occurs
                penalty = min(1.0, (position_from_end - self.target_position) / 5.0)
                return max(0.0, 1.0 - penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Detects amide bond formation by checking for amide coupling patterns.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            products = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) 
                           for r in reactants_smiles.split(".") if r.strip()]
            
            if not products or not reactant_mols:
                return False
                
            # Check for amide bond formation
            return self._detect_amide_formation(reactant_mols, products)
            
        except Exception:
            return False
    
    def _detect_amide_formation(self, reactants, product):
        """
        Detects amide bond formation by looking for:
        1. Amide bonds in product that aren't in reactants
        2. Common amide coupling patterns (carboxylic acid/ester + amine)
        """
        # Amide bond pattern: C(=O)N
        amide_pattern = Chem.MolFromSmarts("[C](=O)[N]")
        
        if not amide_pattern:
            return False
            
        # Count amide bonds in product
        product_amides = len(product.GetSubstructMatches(amide_pattern))
        
        # Count amide bonds in all reactants
        reactant_amides = sum(len(mol.GetSubstructMatches(amide_pattern)) 
                            for mol in reactants)
        
        # If product has more amide bonds than reactants, amide formation occurred
        if product_amides > reactant_amides:
            return True
            
        # Additional check for common amide coupling reactant patterns
        carboxyl_patterns = [
            Chem.MolFromSmarts("[C](=O)[OH]"),  # Carboxylic acid
            Chem.MolFromSmarts("[C](=O)[O][C]"), # Ester
            Chem.MolFromSmarts("[C](=O)[Cl]"),   # Acid chloride
        ]
        
        amine_pattern = Chem.MolFromSmarts("[N;!$(N=*);!$(N#*)]")  # Primary/secondary amine
        
        has_carboxyl = any(
            any(mol.HasSubstructMatch(pattern) for pattern in carboxyl_patterns if pattern)
            for mol in reactants
        )
        
        has_amine = any(
            mol.HasSubstructMatch(amine_pattern) 
            for mol in reactants if amine_pattern
        )
        
        # If we have both carboxyl and amine reactants, and product has amide bonds
        return has_carboxyl and has_amine and product_amides > 0
