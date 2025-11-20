"""Generated evaluation code for: Lactam reduction before ester protection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LactamReductionEsterPresence(BaseScoring):
    """
    Evaluates routes that attempt lactam reduction in the presence of ester groups,
    which creates chemoselectivity challenges. Returns a score based on the depth
    at which this problematic combination occurs.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met - no chemoselectivity issue
        else:
            # Earlier occurrence (lower depth) is more problematic
            return max(0, 10 - x * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves lactam/amide reduction while ester groups are present
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if this is an amide/lactam reduction reaction
            is_amide_reduction = self._is_amide_reduction(reactants, products)
            
            if not is_amide_reduction:
                return False
            
            # Check if ester groups are present in the substrate
            has_ester = self._has_ester_group(reactants)
            
            return is_amide_reduction and has_ester
            
        except Exception:
            return False
    
    def _is_amide_reduction(self, reactants, products) -> bool:
        """
        Detect if this reaction involves reduction of amide/lactam to amine
        """
        # Lactam patterns (common ring sizes)
        lactam_patterns = [
            "[#6]1[#6][#6][#6][#7]([#1,#6])[#6]1=[#8]",  # 6-membered lactam
            "[#6]1[#6][#6][#7]([#1,#6])[#6]1=[#8]",      # 5-membered lactam
            "[#6]1[#6][#6][#6][#6][#7]([#1,#6])[#6]1=[#8]", # 7-membered lactam
        ]
        
        # General amide pattern
        amide_pattern = "[#6][#6](=[#8])[#7]([#1,#6])[#1,#6]"
        
        # Amine patterns (reduction products)
        amine_patterns = [
            "[#6][#6][#7]([#1,#6])[#1,#6]",  # General amine
            "[#6]1[#6][#6][#6][#7]([#1,#6])[#6]1",       # 6-membered cyclic amine
            "[#6]1[#6][#6][#7]([#1,#6])[#6]1",           # 5-membered cyclic amine
        ]
        
        # Check if reactants contain lactam/amide
        has_lactam_amide = False
        for reactant in reactants:
            # Check lactam patterns
            for pattern in lactam_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_lactam_amide = True
                    break
            # Check general amide
            if not has_lactam_amide and reactant.HasSubstructMatch(Chem.MolFromSmarts(amide_pattern)):
                has_lactam_amide = True
            if has_lactam_amide:
                break
        
        if not has_lactam_amide:
            return False
        
        # Check if products contain corresponding amine
        has_amine_product = False
        for product in products:
            for pattern in amine_patterns:
                if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_amine_product = True
                    break
            if has_amine_product:
                break
        
        return has_lactam_amide and has_amine_product
    
    def _has_ester_group(self, reactants) -> bool:
        """
        Check if any reactant contains an ester group
        """
        # Ester patterns
        ester_patterns = [
            "[#6][#6](=[#8])[#8][#6]",      # General ester
            "[#6][#6](=[#8])[#8]c",         # Aromatic ester (like benzoyl ester)
            "c[#6](=[#8])[#8][#6]",         # Benzoyl ester
        ]
        
        for reactant in reactants:
            for pattern in ester_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    return True
        
        return False
