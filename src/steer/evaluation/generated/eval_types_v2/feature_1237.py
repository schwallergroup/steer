"""Generated evaluation code for: Late stage thiolactam to amidine conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ThiolactamToAmidineConversion(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiolactam to amidine conversion.
    
    Detects reactions where a thiolactam substrate is converted to a cyclic amidine
    using hydrazine, favoring late-stage occurrence of this transformation.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur in route
        else:
            if self.timing_preference == "late":
                return 1 - x  # Later occurrence gets higher score
            else:
                return x  # Earlier occurrence gets higher score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a thiolactam to amidine conversion."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for thiolactam substrate and hydrazine reactant
            has_thiolactam = any(self._is_thiolactam(mol) for mol in reactants)
            has_hydrazine = any(self._is_hydrazine(mol) for mol in reactants)
            has_cyclic_amidine = any(self._is_cyclic_amidine(mol) for mol in products)
            
            return has_thiolactam and has_hydrazine and has_cyclic_amidine
            
        except Exception:
            return False
    
    def _is_thiolactam(self, mol) -> bool:
        """Check if molecule contains a thiolactam (cyclic thioamide) structure."""
        if mol is None:
            return False
        
        # SMARTS for thiolactam: cyclic structure with C(=S)N
        thiolactam_patterns = [
            "[#6]1~*~*~[#6](=[#16])[#7]~*~1",  # 6-membered thiolactam
            "[#6]1~*~[#6](=[#16])[#7]~*~1",    # 5-membered thiolactam
            "[#6]1~*~*~*~[#6](=[#16])[#7]~*~1", # 7-membered thiolactam
        ]
        
        for pattern in thiolactam_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        return False
    
    def _is_hydrazine(self, mol) -> bool:
        """Check if molecule is hydrazine or hydrazine derivative."""
        if mol is None:
            return False
        
        # SMARTS for hydrazine: N-N bond
        hydrazine_pattern = "[#7]-[#7]"
        return mol.HasSubstructMatch(Chem.MolFromSmarts(hydrazine_pattern))
    
    def _is_cyclic_amidine(self, mol) -> bool:
        """Check if molecule contains a cyclic amidine structure."""
        if mol is None:
            return False
        
        # SMARTS for cyclic amidine: cyclic structure with C(=N)N
        amidine_patterns = [
            "[#6]1~*~*~[#6](=[#7])[#7]~*~1",    # 6-membered amidine
            "[#6]1~*~[#6](=[#7])[#7]~*~1",      # 5-membered amidine
            "[#6]1~*~*~*~[#6](=[#7])[#7]~*~1",  # 7-membered amidine
        ]
        
        for pattern in amidine_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        return False
