"""Generated evaluation code for: Sequential amide to triazole transformation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialAmideTriazole(MultiRxnCondBase):
    """
    Checks for sequential amide to triazole transformation in synthesis routes.
    Looks for a sequence where an ester is converted to amide, then the amide
    serves as precursor for triazole formation.
    """
    
    def __init__(self, config):
        self.sequential = config.get("sequential", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find ester to amide reactions
        ester_to_amide_indices = []
        for i, rxn in enumerate(reactions):
            if self.detect_ester_to_amide(rxn):
                ester_to_amide_indices.append(i)
        
        # Find amide to triazole reactions  
        amide_to_triazole_indices = []
        for i, rxn in enumerate(reactions):
            if self.detect_amide_to_triazole(rxn):
                amide_to_triazole_indices.append(i)
        
        # Check if both reaction types are present
        has_both = len(ester_to_amide_indices) > 0 and len(amide_to_triazole_indices) > 0
        
        if not has_both:
            return False, len(reactions)
        
        # If sequential is required, check that ester->amide comes before amide->triazole
        if self.sequential:
            # In synthesis trees, earlier reactions have higher indices
            max_ester_amide = max(ester_to_amide_indices)
            min_amide_triazole = min(amide_to_triazole_indices)
            sequential_condition = max_ester_amide > min_amide_triazole
            return sequential_condition, len(reactions)
        
        return True, len(reactions)
    
    def detect_ester_to_amide(self, rxn):
        """Detect ester to amide conversion"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants + products):
            return False
        
        # Ester pattern: R-COO-R'
        ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C,c]")
        # Amide pattern: R-CO-NR'R''
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        
        # Check if reactants contain ester
        has_ester_reactant = any(mol.HasSubstructMatch(ester_pattern) for mol in reactants)
        # Check if products contain amide
        has_amide_product = any(mol.HasSubstructMatch(amide_pattern) for mol in products)
        
        return has_ester_reactant and has_amide_product
    
    def detect_amide_to_triazole(self, rxn):
        """Detect amide to triazole conversion"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants + products):
            return False
        
        # Amide pattern: R-CO-NR'R''
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        # Triazole patterns: 1,2,3-triazole and 1,2,4-triazole
        triazole_123_pattern = Chem.MolFromSmarts("c1nnnc1")  # 1,2,3-triazole
        triazole_124_pattern = Chem.MolFromSmarts("c1ncnn1")  # 1,2,4-triazole
        
        # Check if reactants contain amide
        has_amide_reactant = any(mol.HasSubstructMatch(amide_pattern) for mol in reactants)
        # Check if products contain triazole
        has_triazole_product = any(
            mol.HasSubstructMatch(triazole_123_pattern) or mol.HasSubstructMatch(triazole_124_pattern) 
            for mol in products
        )
        
        return has_amide_reactant and has_triazole_product
