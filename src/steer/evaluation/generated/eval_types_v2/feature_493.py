"""Generated evaluation code for: Late stage SNAr coupling for C-N bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSnArCoupling(BaseScoring):
    """
    Evaluates whether late-stage nucleophilic aromatic substitution (SNAr) 
    coupling for C-N bond formation occurs in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("timing", {}).get("type", "depth")
        self.target_depth = config.get("timing", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr reaction doesn't occur
        else:
            # Reward later occurrence (closer to 1.0 depth fraction)
            if self.condition_type == "bool":
                return 10 if x >= self.target_depth else 0
            else:
                # Higher score for reactions occurring later in synthesis
                return min(10, 10 * x)
    
    def hit_condition(self, d) -> bool:
        """
        Detects SNAr coupling by checking for:
        1. Aromatic C-N bond formation
        2. Presence of electron-withdrawing groups on aromatic ring
        3. Nucleophilic nitrogen source
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            products = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not products or not all(reactants):
                return False
            
            # Check if C-N bond is formed between aromatic carbon and nitrogen
            if not self._has_aromatic_cn_formation(products, reactants):
                return False
            
            # Check for electron-withdrawing groups that activate SNAr
            if not self._has_electron_withdrawing_groups(reactants):
                return False
            
            # Check for nucleophilic nitrogen source
            if not self._has_nucleophilic_nitrogen(reactants):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _has_aromatic_cn_formation(self, products, reactants) -> bool:
        """Check if aromatic C-N bond is formed in the reaction"""
        # Pattern for aromatic carbon bonded to nitrogen
        aromatic_cn_pattern = Chem.MolFromSmarts("[c][N]")
        
        # Count aromatic C-N bonds in products
        product_matches = len(products.GetSubstructMatches(aromatic_cn_pattern))
        
        # Count aromatic C-N bonds in all reactants combined
        reactant_matches = sum(len(r.GetSubstructMatches(aromatic_cn_pattern)) for r in reactants)
        
        # Check if new aromatic C-N bond is formed
        return product_matches > reactant_matches
    
    def _has_electron_withdrawing_groups(self, reactants) -> bool:
        """Check for electron-withdrawing groups that activate SNAr"""
        # Common electron-withdrawing groups for SNAr activation
        ewg_patterns = [
            "[c][N+](=O)[O-]",  # Nitro group
            "[c]C(=O)",         # Carbonyl
            "[c]C#N",           # Cyano
            "[c]C(F)(F)F",      # Trifluoromethyl
            "[c][N+]",          # Quaternary nitrogen
            "[c]S(=O)(=O)",     # Sulfonyl
        ]
        
        ewg_mols = [Chem.MolFromSmarts(pattern) for pattern in ewg_patterns]
        
        for reactant in reactants:
            for ewg_mol in ewg_mols:
                if ewg_mol and reactant.HasSubstructMatch(ewg_mol):
                    return True
        return False
    
    def _has_nucleophilic_nitrogen(self, reactants) -> bool:
        """Check for nucleophilic nitrogen sources"""
        # Patterns for nucleophilic nitrogen sources
        nucleophile_patterns = [
            "[N][CH2]",         # Primary/secondary amines
            "[N]([CH3])[CH2]",  # Secondary amines
            "[N]c1ccccc1",      # Anilines
            "[N]C1CCCCC1",      # Cyclic amines
            "[N]C1CCNCC1",      # Piperazines
            "c1ccc2c(c1)CCN2",  # Tetrahydroisoquinoline-like structures
        ]
        
        nucleophile_mols = [Chem.MolFromSmarts(pattern) for pattern in nucleophile_patterns]
        
        for reactant in reactants:
            for nuc_mol in nucleophile_mols:
                if nuc_mol and reactant.HasSubstructMatch(nuc_mol):
                    return True
        return False
