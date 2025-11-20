"""Generated evaluation code for: Late stage C-O ether bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCOEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage C-O ether bond formation via SNAr mechanism.
    Detects nucleophilic aromatic substitution reactions where an alcohol forms an ether bond
    with an aromatic system, particularly targeting macrocycle-heterocycle connections.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")  # "early", "middle", "late"
        self.mechanism_filter = config.get("mechanism", "SNAr")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # C-O ether formation doesn't occur
        
        if self.timing_preference == "late":
            return 1 - x  # Later is better, score decreases with earlier depth
        elif self.timing_preference == "early":
            return x  # Earlier is better, score increases with depth
        else:  # middle
            return 1 - abs(x - 0.5) * 2  # Peak scoring at middle depth
    
    def hit_condition(self, d) -> bool:
        """
        Detects C-O ether bond formation via SNAr mechanism by analyzing
        mapped reaction SMILES for characteristic patterns.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles.strip())
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is C-O ether formation
            return self._detect_co_ether_formation(reactants, product)
            
        except Exception:
            return False
    
    def _detect_co_ether_formation(self, reactants, product) -> bool:
        """
        Detects C-O ether bond formation by comparing reactants and product.
        Looks for SNAr pattern: aromatic halide + alcohol -> ether + HX
        """
        # Pattern for aromatic halide (electron-deficient aromatic with leaving group)
        aromatic_halide_patterns = [
            "[cH0:1][F,Cl,Br,I:2]",  # Aromatic carbon with halide
            "[c:1][N+:2]",            # Aromatic carbon with nitro/nitrile
        ]
        
        # Pattern for alcohol
        alcohol_pattern = "[CH2,CH:1][OH:2]"
        
        # Pattern for C-O ether in product
        ether_pattern = "[c:1][O:2][CH2,CH:3]"  # Aromatic C-O-alkyl
        
        # Check if product contains the ether pattern
        ether_mol = Chem.MolFromSmarts(ether_pattern)
        if not product.HasSubstructMatch(ether_mol):
            return False
        
        # Check if reactants contain aromatic halide and alcohol
        has_aromatic_halide = False
        has_alcohol = False
        
        for reactant in reactants:
            # Check for aromatic halide
            for pattern_smarts in aromatic_halide_patterns:
                pattern_mol = Chem.MolFromSmarts(pattern_smarts)
                if pattern_mol and reactant.HasSubstructMatch(pattern_mol):
                    has_aromatic_halide = True
                    break
            
            # Check for alcohol
            alcohol_mol = Chem.MolFromSmarts(alcohol_pattern)
            if alcohol_mol and reactant.HasSubstructMatch(alcohol_mol):
                has_alcohol = True
        
        # Additional check for SNAr mechanism: look for electron-withdrawing groups
        if has_aromatic_halide and has_alcohol:
            return self._has_electron_withdrawing_groups(reactants)
        
        return False
    
    def _has_electron_withdrawing_groups(self, reactants) -> bool:
        """
        Checks for electron-withdrawing groups that facilitate SNAr mechanism.
        """
        ewg_patterns = [
            "[c][N+](=O)[O-]",  # Nitro group
            "[c][C]#N",         # Cyano group  
            "[c][C](=O)",       # Carbonyl
            "[c][S](=O)(=O)",   # Sulfonyl
            "[c][C](F)(F)F",    # Trifluoromethyl
            "n",                # Pyridine nitrogen
        ]
        
        for reactant in reactants:
            for pattern_smarts in ewg_patterns:
                pattern_mol = Chem.MolFromSmarts(pattern_smarts)
                if pattern_mol and reactant.HasSubstructMatch(pattern_mol):
                    return True
        
        return False
