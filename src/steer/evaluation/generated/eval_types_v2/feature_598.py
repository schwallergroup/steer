"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates whether a nucleophilic aromatic substitution (SNAr) reaction 
    occurs at a late stage in the synthesis route.
    
    SNAr reactions typically involve electron-deficient aromatic rings being
    attacked by nucleophiles, often forming C-O, C-N, or C-S bonds.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr reaction doesn't occur
        
        if self.timing == "late":
            # Reward reactions that occur early in the route (low depth)
            # since depth is measured from target molecule
            if x <= self.depth_threshold:
                return 10.0  # Perfect score for late-stage (low depth)
            else:
                # Penalize reactions that occur too early in synthesis
                return max(0, 10.0 - 2.0 * (x - self.depth_threshold))
        else:
            # For other timing preferences, could implement different scoring
            return 5.0
    
    def hit_condition(self, d) -> bool:
        """
        Detect nucleophilic aromatic substitution by looking for:
        1. Aromatic ring in both product and reactant
        2. Nucleophile attachment to aromatic carbon
        3. Loss of leaving group from aromatic ring
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Look for characteristic SNAr patterns
            return self._detect_snar_pattern(prod_mol, react_mols)
            
        except Exception:
            return False
    
    def _detect_snar_pattern(self, product, reactants):
        """
        Detect SNAr by identifying:
        - Electron-deficient aromatic rings
        - Nucleophilic substitution patterns
        """
        # Common SNAr patterns: nucleophile attacking electron-deficient aromatic
        snar_patterns = [
            # Aromatic ether formation (C-O bond)
            "[cH0:1][O:2]",
            # Aromatic amine formation (C-N bond) 
            "[cH0:1][N:2]",
            # Aromatic thioether formation (C-S bond)
            "[cH0:1][S:2]",
            # Electron-withdrawing groups that activate SNAr
            "c[N+](=O)[O-]",  # nitro group
            "c[C](=O)",       # carbonyl
            "c[F,Cl,Br]",     # halogens
        ]
        
        # Check if product contains potential SNAr product patterns
        has_snar_product = False
        for pattern in snar_patterns[:3]:  # C-O, C-N, C-S patterns
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and product.HasSubstructMatch(patt_mol):
                has_snar_product = True
                break
        
        if not has_snar_product:
            return False
        
        # Check for electron-deficient aromatic system in reactants
        has_activated_aromatic = False
        activating_patterns = [
            "c1c([N+](=O)[O-])cccc1",  # nitrobenzene
            "c1c([F,Cl,Br])cccc1",     # halobenzene
            "c1c(C(=O))cccc1",         # aromatic carbonyl
            "c1c([F,Cl,Br])cc([N+](=O)[O-])cc1",  # diactivated
        ]
        
        for reactant in reactants:
            for pattern in activating_patterns:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol and reactant.HasSubstructMatch(patt_mol):
                    has_activated_aromatic = True
                    break
            if has_activated_aromatic:
                break
        
        return has_activated_aromatic
