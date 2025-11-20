"""Generated evaluation code for: Late stage SNAr ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSnArEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage SNAr ether formation reactions.
    Checks for nucleophilic aromatic substitution that forms C-O bonds,
    with preference for reactions occurring later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Prefer late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr ether formation doesn't happen
        else:
            # Late-stage (closer to 0) is better, early stage (closer to 1) is worse
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                # Continuous scoring: reward late-stage reactions
                return max(0, 1 - x)
    
    def hit_condition(self, d):
        """Check if this reaction is an SNAr ether formation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = Chem.MolFromSmiles(rxn_parts[0])
            reactants_smiles = rxn_parts[1].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not products or not reactants:
                return False
            
            # Check if this is SNAr ether formation
            return self._is_snar_ether_formation(products, reactants)
            
        except Exception:
            return False
    
    def _is_snar_ether_formation(self, products, reactants):
        """Detect SNAr ether formation by checking for aryl halide + nucleophile -> aryl ether"""
        
        # Pattern for electron-deficient aromatic rings (with electron-withdrawing groups)
        # Common activating groups for SNAr: NO2, CN, CF3, CHO, COR
        activated_aryl_halide_patterns = [
            "[cH0:1]1[cH][c]([N+](=O)[O-])[cH][cH][c]1[F,Cl,Br,I]",  # ortho to NO2
            "[cH0:1]1[cH][cH][c]([N+](=O)[O-])[cH][c]1[F,Cl,Br,I]",  # meta to NO2
            "[cH0:1]1[c]([N+](=O)[O-])[cH][cH][cH][c]1[F,Cl,Br,I]",  # para to NO2
            "[cH0:1]1[cH][c](C#N)[cH][cH][c]1[F,Cl,Br,I]",  # ortho to CN
            "[cH0:1]1[cH][cH][c](C#N)[cH][c]1[F,Cl,Br,I]",  # meta to CN
            "[cH0:1]1[c](C#N)[cH][cH][cH][c]1[F,Cl,Br,I]",  # para to CN
        ]
        
        # Pattern for aryl ether product
        aryl_ether_pattern = "[cH0:1]1[cH][cH][cH][cH][c]1[O][C,c]"
        
        # Check if product contains aryl ether
        has_aryl_ether = False
        ether_pattern = Chem.MolFromSmarts(aryl_ether_pattern)
        if ether_pattern and products.HasSubstructMatch(ether_pattern):
            has_aryl_ether = True
        
        # Check if reactants contain activated aryl halide
        has_activated_aryl_halide = False
        for pattern_smarts in activated_aryl_halide_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern:
                for reactant in reactants:
                    if reactant.HasSubstructMatch(pattern):
                        has_activated_aryl_halide = True
                        break
                if has_activated_aryl_halide:
                    break
        
        # Check for nucleophile (alcohol, phenol, or alkoxide)
        nucleophile_patterns = [
            "[OH]",  # alcohol/phenol
            "[O-]",  # alkoxide
            "[OH][c]",  # phenol specifically
        ]
        
        has_nucleophile = False
        for pattern_smarts in nucleophile_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern:
                for reactant in reactants:
                    if reactant.HasSubstructMatch(pattern):
                        has_nucleophile = True
                        break
                if has_nucleophile:
                    break
        
        # Also check for leaving group (halide) in byproducts
        has_leaving_group = False
        halide_patterns = ["[F-]", "[Cl-]", "[Br-]", "[I-]", "F", "Cl", "Br", "I"]
        for pattern_smarts in halide_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern:
                for reactant in reactants:
                    if reactant.HasSubstructMatch(pattern):
                        has_leaving_group = True
                        break
                if has_leaving_group:
                    break
        
        # SNAr ether formation if we have aryl ether product and activated aryl halide reactant
        # and either nucleophile or evidence of substitution
        return has_aryl_ether and has_activated_aryl_halide and (has_nucleophile or has_leaving_group)
