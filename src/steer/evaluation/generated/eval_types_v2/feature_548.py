"""Generated evaluation code for: Protecting group swap from Boc to nosyl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocToNosylSwap(BaseScoring):
    """
    Evaluates synthesis routes for Boc to nosyl protecting group swap strategy.
    Detects sequential Boc deprotection followed by nosylation on nitrogen atoms.
    Rewards late-stage execution of this protecting group strategy.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late_stage")
        self.atom_type = config.get("atom", "nitrogen")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        
        if self.timing == "late_stage":
            return 1 - x  # Late-stage is better (higher score for higher depth fraction)
        else:
            return x  # Early-stage preferred
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction step represents a Boc to nosyl swap"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactant = Chem.MolFromSmiles(rxn_parts[0])
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            if not reactant or not all(products):
                return False
            
            # Check for Boc deprotection (loss of Boc group)
            boc_pattern = Chem.MolFromSmarts("[N:1]C(=O)OC(C)(C)C")  # Boc protecting group
            nosyl_pattern = Chem.MolFromSmarts("[N:1]S(=O)(=O)c1ccc(cc1)[N+](=O)[O-]")  # Nosyl group
            
            # Check if reactant has Boc-protected nitrogen
            has_boc_reactant = reactant.HasSubstructMatch(boc_pattern)
            
            # Check if any product has nosyl-protected nitrogen
            has_nosyl_product = any(prod.HasSubstructMatch(nosyl_pattern) for prod in products)
            
            # Check that Boc is removed (not present in products)
            has_boc_product = any(prod.HasSubstructMatch(boc_pattern) for prod in products)
            
            # Look for the specific swap: Boc in reactant, nosyl in product, no Boc in product
            if has_boc_reactant and has_nosyl_product and not has_boc_product:
                # Verify same nitrogen is involved by checking atom mapping
                boc_matches = reactant.GetSubstructMatches(boc_pattern)
                for prod in products:
                    if prod.HasSubstructMatch(nosyl_pattern):
                        nosyl_matches = prod.GetSubstructMatches(nosyl_pattern)
                        # Check if mapped atoms suggest same nitrogen involved
                        for boc_match in boc_matches:
                            boc_n_atom = reactant.GetAtomWithIdx(boc_match[0])
                            boc_map_num = boc_n_atom.GetAtomMapNum()
                            
                            if boc_map_num > 0:  # Has atom mapping
                                for nosyl_match in nosyl_matches:
                                    nosyl_n_atom = prod.GetAtomWithIdx(nosyl_match[0])
                                    if nosyl_n_atom.GetAtomMapNum() == boc_map_num:
                                        return True
                
                # If no clear mapping, accept based on structural change
                return True
            
            return False
            
        except Exception:
            return False
