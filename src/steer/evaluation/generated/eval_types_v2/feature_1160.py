"""Generated evaluation code for: Boc protecting group for piperidine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocPiperidineProtection(BaseScoring):
    """
    Evaluates synthesis routes based on the use of Boc protecting groups for piperidine nitrogen.
    Checks if Boc protection is applied to secondary amine nitrogen in piperidine rings.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier application of protection is generally better
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves Boc protection of piperidine nitrogen.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if Boc group appears in products but not reactants
            boc_in_products = self._has_boc_piperidine(products)
            boc_in_reactants = self._has_boc_piperidine(reactants)
            
            # Protection reaction: Boc appears in products but not in reactants
            return boc_in_products and not boc_in_reactants
            
        except Exception:
            return False
    
    def _has_boc_piperidine(self, mol_smiles_list) -> bool:
        """
        Check if any molecule contains Boc-protected piperidine nitrogen.
        """
        # Boc-protected piperidine patterns
        boc_piperidine_patterns = [
            "C1CCN(CC1)C(=O)OC(C)(C)C",  # N-Boc piperidine
            "CC(C)(C)OC(=O)N1CCCCC1",    # Alternative representation
            "[CH2]1[CH2]N([CH2][CH2][CH2]1)C(=O)OC(C)(C)C"  # More explicit pattern
        ]
        
        for smiles in mol_smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # Check against Boc-piperidine patterns
                for pattern in boc_piperidine_patterns:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        return True
                        
                # Additional check for Boc group and piperidine ring separately
                boc_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")  # Boc group
                piperidine_pattern = Chem.MolFromSmarts("N1CCCCC1")   # Piperidine ring
                
                if (boc_pattern and mol.HasSubstructMatch(boc_pattern) and 
                    piperidine_pattern and mol.HasSubstructMatch(piperidine_pattern)):
                    return True
                    
            except Exception:
                continue
                
        return False
