"""Generated evaluation code for: Multiple Williamson ether formations"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleWilliamsonEther(MultiRxnCondBase):
    """
    Checks if a synthesis route contains at least a specified number of Williamson ether synthesis reactions.
    Williamson ether synthesis involves nucleophilic substitution between an alkoxide and an alkyl halide/tosylate.
    """
    
    def __init__(self, config):
        self.minimum_count = config.get("minimum_count", 3)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        williamson_count = sum(1 for r in reactions if self.detect_williamson_ether(r))
        
        condition = williamson_count >= self.minimum_count
        return condition, len(reactions)
    
    def detect_williamson_ether(self, rxn):
        """
        Detects Williamson ether synthesis by looking for:
        1. Formation of C-O-C ether bond
        2. Presence of leaving group (halide, tosylate, etc.) in products
        3. Loss of nucleophile counter-ion
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check for ether formation (C-O-C pattern)
            ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
            ether_in_products = any(mol.HasSubstructMatch(ether_pattern) for mol in products)
            
            if not ether_in_products:
                return False
            
            # Check for typical leaving groups in products
            leaving_groups = [
                "[Cl-]",  # Chloride
                "[Br-]",  # Bromide  
                "[I-]",   # Iodide
                "OS(=O)(=O)c1ccc(C)cc1",  # Tosylate
                "OS(=O)(=O)C(F)(F)F",     # Triflate
            ]
            
            leaving_group_present = False
            for lg_smarts in leaving_groups:
                lg_pattern = Chem.MolFromSmarts(lg_smarts)
                if lg_pattern and any(mol.HasSubstructMatch(lg_pattern) for mol in products):
                    leaving_group_present = True
                    break
            
            # Alternative check: look for alkyl halide consumption
            if not leaving_group_present:
                alkyl_halide_patterns = [
                    "[C][Cl]",  # Alkyl chloride
                    "[C][Br]",  # Alkyl bromide
                    "[C][I]",   # Alkyl iodide
                ]
                
                alkyl_halide_in_reactants = False
                for pattern_smarts in alkyl_halide_patterns:
                    pattern = Chem.MolFromSmarts(pattern_smarts)
                    if pattern and any(mol.HasSubstructMatch(pattern) for mol in reactants):
                        alkyl_halide_in_reactants = True
                        break
                
                # Check if alkyl halide is consumed (not present in products)
                if alkyl_halide_in_reactants:
                    alkyl_halide_in_products = False
                    for pattern_smarts in alkyl_halide_patterns:
                        pattern = Chem.MolFromSmarts(pattern_smarts)
                        if pattern and any(mol.HasSubstructMatch(pattern) for mol in products):
                            alkyl_halide_in_products = True
                            break
                    leaving_group_present = not alkyl_halide_in_products
            
            return ether_in_products and leaving_group_present
            
        except Exception:
            return False
