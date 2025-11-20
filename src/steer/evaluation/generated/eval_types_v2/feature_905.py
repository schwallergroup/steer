"""Generated evaluation code for: Boc protection strategy for piperidine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(BaseScoring):
    """
    Evaluates Boc protection strategy for piperidine nitrogen.
    Checks if Boc protection of piperidine secondary amine occurs in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection not found
        else:
            # Earlier Boc protection is better (lower depth)
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of piperidine nitrogen"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for Boc protection reaction
            return self._is_boc_protection_reaction(reactants, products)
            
        except Exception:
            return False
    
    def _is_boc_protection_reaction(self, reactants, products):
        """Check if reaction involves Boc protection of piperidine nitrogen"""
        
        # Piperidine pattern (secondary amine in 6-membered ring)
        piperidine_pattern = Chem.MolFromSmarts("[NH1;R1][CH2][CH2][CH2][CH2][CH2]1")
        
        # Boc-protected piperidine pattern
        boc_piperidine_pattern = Chem.MolFromSmarts("[N;R1]([CH2][CH2][CH2][CH2][CH2]1)C(=O)OC(C)(C)C")
        
        # Alternative Boc pattern (more general)
        boc_group_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")
        
        if not piperidine_pattern or not boc_piperidine_pattern or not boc_group_pattern:
            return False
        
        # Check if reactants contain unprotected piperidine
        has_unprotected_piperidine = False
        for reactant in reactants:
            if reactant.HasSubstructMatch(piperidine_pattern):
                has_unprotected_piperidine = True
                break
        
        # Check if products contain Boc-protected piperidine
        has_boc_protected_piperidine = False
        for product in products:
            if (product.HasSubstructMatch(boc_piperidine_pattern) or 
                product.HasSubstructMatch(boc_group_pattern)):
                has_boc_protected_piperidine = True
                break
        
        # Check if Boc reagent is present in reactants
        boc_reagents = [
            Chem.MolFromSmarts("(C(C)(C)C)OC(=O)ON1C(=O)CCC1=O"),  # Boc-OSu
            Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"),    # Boc2O
            Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl")                   # Boc-Cl
        ]
        
        has_boc_reagent = False
        for reactant in reactants:
            for boc_reagent_pattern in boc_reagents:
                if boc_reagent_pattern and reactant.HasSubstructMatch(boc_reagent_pattern):
                    has_boc_reagent = True
                    break
            if has_boc_reagent:
                break
        
        return (has_unprotected_piperidine and 
                has_boc_protected_piperidine and 
                has_boc_reagent)
