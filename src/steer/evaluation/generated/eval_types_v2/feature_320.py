"""Generated evaluation code for: Late stage S-arylation coupling on unprotected sugar"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SugarSArylationCoupling(BaseScoring):
    """
    Evaluates late-stage S-arylation coupling on unprotected sugar substrates.
    Checks if S-arylation occurs on a sugar molecule without hydroxyl protection.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "inverse")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction not found
        else:
            return 1 - x  # Later stage is better for this coupling
    
    def hit_condition(self, d) -> bool:
        """Check if reaction is S-arylation on unprotected sugar"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        product_smiles, reactants_smiles = rxn_smiles.split(">>")
        product = Chem.MolFromSmiles(product_smiles)
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
        
        if not product or not all(reactants):
            return False
        
        # Check if product contains sugar scaffold and S-aryl linkage
        if not (self._has_sugar_scaffold(product) and self._has_s_aryl_linkage(product)):
            return False
        
        # Check if any reactant has unprotected sugar (>=3 free OH groups)
        sugar_reactant = None
        for reactant in reactants:
            if self._has_sugar_scaffold(reactant):
                sugar_reactant = reactant
                break
        
        if not sugar_reactant:
            return False
        
        # Verify sugar is unprotected (has multiple free OH groups)
        free_oh_count = self._count_free_hydroxyl_groups(sugar_reactant)
        
        # Check if this is actually an S-arylation (C-S bond formation)
        is_s_arylation = self._is_s_arylation_reaction(product, reactants)
        
        return free_oh_count >= 3 and is_s_arylation
    
    def _has_sugar_scaffold(self, mol) -> bool:
        """Check if molecule contains sugar-like scaffold"""
        # Sugar patterns: pyranose and furanose rings with OH groups
        pyranose_pattern = Chem.MolFromSmarts("[CH]1[CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])O1")
        furanose_pattern = Chem.MolFromSmarts("[CH]1[CH]([OH])[CH]([OH])[CH]([OH])O1")
        # More flexible sugar patterns
        sugar_pattern1 = Chem.MolFromSmarts("[C,CH]1O[C,CH]([C,CH]([OH])[C,CH]([OH])[C,CH]1[OH])")
        sugar_pattern2 = Chem.MolFromSmarts("O1[CH][CH]([OH])[CH]([OH])[CH]([OH])[CH]1")
        
        patterns = [pyranose_pattern, furanose_pattern, sugar_pattern1, sugar_pattern2]
        return any(mol.HasSubstructMatch(pattern) for pattern in patterns if pattern)
    
    def _has_s_aryl_linkage(self, mol) -> bool:
        """Check if molecule has sulfur-aryl linkage"""
        s_aryl_pattern = Chem.MolFromSmarts("c-S-[C,CH]")  # Aromatic carbon bonded to sulfur
        return mol.HasSubstructMatch(s_aryl_pattern) if s_aryl_pattern else False
    
    def _count_free_hydroxyl_groups(self, mol) -> int:
        """Count free hydroxyl groups in molecule"""
        oh_pattern = Chem.MolFromSmarts("[OH1]")  # Free OH (not protected)
        if oh_pattern:
            matches = mol.GetSubstructMatches(oh_pattern)
            return len(matches)
        return 0
    
    def _is_s_arylation_reaction(self, product, reactants) -> bool:
        """Verify this is actually an S-arylation by checking bond formation"""
        # Check if C-S bonds increased from reactants to product
        product_cs_bonds = self._count_c_s_bonds(product)
        reactant_cs_bonds = sum(self._count_c_s_bonds(r) for r in reactants)
        
        return product_cs_bonds > reactant_cs_bonds
    
    def _count_c_s_bonds(self, mol) -> int:
        """Count C-S bonds in molecule"""
        count = 0
        for bond in mol.GetBonds():
            atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
            atom_symbols = [atom.GetSymbol() for atom in atoms]
            if 'C' in atom_symbols and 'S' in atom_symbols:
                count += 1
        return count
