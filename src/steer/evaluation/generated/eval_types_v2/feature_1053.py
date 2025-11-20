"""Generated evaluation code for: Benzyl protecting group for amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylAmineProtection(MultiRxnCondBase):
    """
    Evaluates the use of benzyl protecting groups on amine nitrogens.
    Checks for benzyl protection/deprotection reactions and scores based on
    the strategic use throughout the synthesis route.
    """
    
    def __init__(self, config):
        self.require_protection = config.get("require_protection", True)
        self.require_deprotection = config.get("require_deprotection", True)
        self.prefer_late_deprotection = config.get("prefer_late_deprotection", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_protection = any(self.detect_benzyl_protection(r) for r in reactions)
        has_deprotection = any(self.detect_benzyl_deprotection(r) for r in reactions)
        
        # Calculate deprotection depth if present
        deprotection_depth = -1
        for i, r in enumerate(reactions):
            if self.detect_benzyl_deprotection(r):
                deprotection_depth = i / len(reactions) if len(reactions) > 0 else 0
                break
        
        # Store deprotection timing for scoring
        self._deprotection_depth = deprotection_depth
        
        condition_met = True
        if self.require_protection and not has_protection:
            condition_met = False
        if self.require_deprotection and not has_deprotection:
            condition_met = False
            
        return condition_met, len(reactions)
    
    def route_scoring(self, x):
        if x < 0:
            return 0  # Condition not met
        
        # Base score for meeting protection/deprotection requirements
        score = 7
        
        # Bonus for late-stage deprotection (closer to 1.0 is later)
        if self.prefer_late_deprotection and hasattr(self, '_deprotection_depth') and self._deprotection_depth >= 0:
            if self._deprotection_depth > 0.7:  # Late stage
                score += 2
            elif self._deprotection_depth > 0.4:  # Mid stage
                score += 1
            # Early deprotection gets no bonus
        
        return min(score, 10)
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection of amine (N-H -> N-Bn)"""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Look for benzyl chloride or benzyl bromide as reactant
            benzyl_halide_patterns = [
                Chem.MolFromSmarts("ClCc1ccccc1"),  # Benzyl chloride
                Chem.MolFromSmarts("BrCc1ccccc1"),  # Benzyl bromide
                Chem.MolFromSmarts("c1ccc(CO)cc1")   # Benzyl alcohol (under activating conditions)
            ]
            
            has_benzyl_reagent = any(
                any(mol and pattern and mol.HasSubstructMatch(pattern) 
                    for pattern in benzyl_halide_patterns)
                for mol in reactants if mol
            )
            
            if not has_benzyl_reagent:
                return False
            
            # Look for formation of N-benzyl bond in products
            n_benzyl_pattern = Chem.MolFromSmarts("NCc1ccccc1")  # N-benzyl substructure
            
            has_n_benzyl_product = any(
                mol and n_benzyl_pattern and mol.HasSubstructMatch(n_benzyl_pattern)
                for mol in products if mol
            )
            
            return has_n_benzyl_product
            
        except Exception:
            return False
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl deprotection via hydrogenation (N-Bn -> N-H)"""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Check for N-benzyl in reactants
            n_benzyl_pattern = Chem.MolFromSmarts("NCc1ccccc1")
            has_n_benzyl_reactant = any(
                mol and n_benzyl_pattern and mol.HasSubstructMatch(n_benzyl_pattern)
                for mol in reactants if mol
            )
            
            if not has_n_benzyl_reactant:
                return False
            
            # Check for toluene as product (indicates benzyl group removal)
            toluene_pattern = Chem.MolFromSmarts("Cc1ccccc1")
            has_toluene_product = any(
                mol and toluene_pattern and mol.HasSubstructMatch(toluene_pattern)
                for mol in products if mol
            )
            
            # Also check for H2 as reactant (hydrogenation conditions)
            has_hydrogen = any(
                mol and Chem.MolToSmiles(mol) == "[H][H]"
                for mol in reactants if mol
            )
            
            return has_toluene_product or has_hydrogen
            
        except Exception:
            return False
