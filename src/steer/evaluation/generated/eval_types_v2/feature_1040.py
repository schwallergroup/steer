"""Generated evaluation code for: Boc protection strategy for secondary amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(BaseScoring):
    """
    Evaluates the timing of Boc protection for secondary amines in synthesis routes.
    Rewards early-stage Boc protection to prevent side reactions during subsequent steps.
    """
    
    def __init__(self, config: Dict):
        self.timing = config["parameters"]["timing"]  # "early", "late", or specific depth
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection doesn't occur
        
        if self.timing == "early":
            return 1 - x  # Earlier protection is better (lower depth gets higher score)
        elif self.timing == "late":
            return x  # Later protection is better (higher depth gets higher score)
        else:
            # If timing is a specific depth value
            target_depth = float(self.timing)
            return 1 - abs(x - target_depth)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Boc protection of secondary amines by checking for:
        1. Formation of Boc-protected amine (C(C)(C)OC(=O)N pattern)
        2. Presence of secondary amine in reactants
        3. Boc-protected secondary amine in products
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        # Parse reactants and products
        try:
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
        except:
            return False
        
        # SMARTS patterns
        secondary_amine_pattern = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]([#6])[#6]")  # Secondary amine
        boc_protected_amine_pattern = Chem.MolFromSmarts("[NX3]([#6])([#6])C(=O)OC(C)(C)C")  # Boc-protected amine
        boc_reagent_pattern = Chem.MolFromSmarts("C(C)(C)OC(=O)OC(=O)OC(C)(C)C")  # Boc2O reagent
        
        # Check if we have secondary amine in reactants
        has_secondary_amine_reactant = any(
            mol.HasSubstructMatch(secondary_amine_pattern) for mol in reactants
        )
        
        # Check if we have Boc reagent in reactants
        has_boc_reagent = any(
            mol.HasSubstructMatch(boc_reagent_pattern) for mol in reactants
        )
        
        # Check if we have Boc-protected amine in products
        has_boc_protected_product = any(
            mol.HasSubstructMatch(boc_protected_amine_pattern) for mol in products
        )
        
        # Alternative check: look for the simpler Boc anhydride pattern
        boc_anhydride_simple = Chem.MolFromSmarts("CC(C)(C)OC(=O)")
        has_boc_anhydride = any(
            mol.HasSubstructMatch(boc_anhydride_simple) for mol in reactants
        )
        
        # Boc protection occurs when:
        # 1. Secondary amine present in reactants
        # 2. Boc reagent (Boc2O or similar) present in reactants  
        # 3. Boc-protected amine formed in products
        return (has_secondary_amine_reactant and 
                (has_boc_reagent or has_boc_anhydride) and 
                has_boc_protected_product)
