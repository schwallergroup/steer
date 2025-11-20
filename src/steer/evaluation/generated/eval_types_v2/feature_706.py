"""Generated evaluation code for: Benzyl protecting group with aryl bromide present"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectionArylBromide(BaseScoring):
    """
    Detects benzyl protection of nitrogen when aryl bromide functionality is present.
    This creates a chemoselectivity issue during hydrogenolytic deprotection due to 
    competing hydrodebromination of the aryl bromide.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
        # SMARTS patterns
        self.benzyl_nitrogen_pattern = "[N;!$(N=*);!$(N#*)]C[cH0,c]1ccccc1"  # N-benzyl pattern
        self.aryl_bromide_pattern = "c1ccc(Br)cc1"  # Simple aryl bromide
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to penalty score (0-10, higher = worse)"""
        if x < 0:
            return 0  # Issue not detected
        
        if self.condition_type == "bool":
            return 10  # Maximum penalty when detected
        else:
            # Earlier detection (lower x) gets higher penalty
            penalty = 10 * (1 - x)
            return max(0, min(10, penalty))
    
    def hit_condition(self, d) -> bool:
        """Check if reaction introduces benzyl protection with aryl bromide present"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products_smiles)
            if not product_mol:
                return False
                
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactant_mols.append(mol)
            
            # Check if product has both benzyl-nitrogen and aryl bromide
            has_benzyl_n = self._has_benzyl_nitrogen(product_mol)
            has_aryl_br = self._has_aryl_bromide(product_mol)
            
            if not (has_benzyl_n and has_aryl_br):
                return False
            
            # Check if this reaction actually formed the benzyl-nitrogen bond
            # (i.e., benzyl-N present in product but not in any single reactant)
            benzyl_n_in_reactants = any(self._has_benzyl_nitrogen(mol) for mol in reactant_mols)
            
            # True if benzyl protection occurred in presence of aryl bromide
            return has_benzyl_n and has_aryl_br and not benzyl_n_in_reactants
            
        except Exception:
            return False
    
    def _has_benzyl_nitrogen(self, mol) -> bool:
        """Check if molecule contains N-benzyl substructure"""
        if not mol:
            return False
        pattern = Chem.MolFromSmarts(self.benzyl_nitrogen_pattern)
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_aryl_bromide(self, mol) -> bool:
        """Check if molecule contains aryl bromide substructure"""
        if not mol:
            return False
        pattern = Chem.MolFromSmarts(self.aryl_bromide_pattern)
        return mol.HasSubstructMatch(pattern) if pattern else False
