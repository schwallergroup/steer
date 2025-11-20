"""Generated evaluation code for: Late stage lactam reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LactamReductionTiming(BaseScoring):
    """
    Evaluates the timing of lactam reduction reactions in synthesis routes.
    Rewards late-stage lactam reduction (conversion of cyclic amide to cyclic amine).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No lactam reduction found
        
        if self.condition_type == "bool":
            # Boolean scoring: reward if reduction happens late (depth > 0.5)
            return 1 if x > 0.5 else 0
        else:
            # Continuous scoring: penalize deviation from target depth
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Detects lactam reduction by checking for:
        1. Lactam (cyclic amide) in reactants
        2. Corresponding cyclic amine in products
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            return self._is_lactam_reduction(reactants, products)
            
        except Exception:
            return False
    
    def _is_lactam_reduction(self, reactants, products) -> bool:
        """
        Check if reaction converts lactam to cyclic amine.
        """
        # SMARTS patterns for different lactam sizes
        lactam_patterns = [
            "[#6]1[#6][#6][#7]([#1,#6])[#6](=[#8])[#6]1",  # 6-membered lactam
            "[#6]1[#6][#7]([#1,#6])[#6](=[#8])[#6]1",       # 5-membered lactam  
            "[#6]1[#6][#6][#6][#7]([#1,#6])[#6](=[#8])[#6]1", # 7-membered lactam
            "[#6]1[#7]([#1,#6])[#6](=[#8])[#6]1"             # 4-membered lactam
        ]
        
        # SMARTS patterns for corresponding cyclic amines
        cyclic_amine_patterns = [
            "[#6]1[#6][#6][#7]([#1,#6])[#6][#6]1",  # 6-membered cyclic amine
            "[#6]1[#6][#7]([#1,#6])[#6][#6]1",      # 5-membered cyclic amine
            "[#6]1[#6][#6][#6][#7]([#1,#6])[#6][#6]1", # 7-membered cyclic amine
            "[#6]1[#7]([#1,#6])[#6][#6]1"           # 4-membered cyclic amine
        ]
        
        # Check for lactam in reactants
        has_lactam = False
        for reactant in reactants:
            for pattern in lactam_patterns:
                try:
                    lactam_query = Chem.MolFromSmarts(pattern)
                    if lactam_query and reactant.HasSubstructMatch(lactam_query):
                        has_lactam = True
                        break
                except Exception:
                    continue
            if has_lactam:
                break
        
        if not has_lactam:
            return False
        
        # Check for cyclic amine in products
        has_cyclic_amine = False
        for product in products:
            for pattern in cyclic_amine_patterns:
                try:
                    amine_query = Chem.MolFromSmarts(pattern)
                    if amine_query and product.HasSubstructMatch(amine_query):
                        has_cyclic_amine = True
                        break
                except Exception:
                    continue
            if has_cyclic_amine:
                break
        
        return has_lactam and has_cyclic_amine
