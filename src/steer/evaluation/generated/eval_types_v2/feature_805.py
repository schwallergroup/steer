"""Generated evaluation code for: Standard Boc protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupStrategy(BaseScoring):
    """
    Evaluates routes for Boc protecting group strategy.
    Checks if Boc protection is applied at the specified timing (late-stage preferred).
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.frequency = config["parameters"]["frequency"]  # single
        self.timing = config["parameters"]["timing"]  # late
        
        # Boc group SMARTS patterns
        self.boc_patterns = [
            "NC(=O)OC(C)(C)C",  # Boc-protected amine
            "[NH]C(=O)OC(C)(C)C",  # Boc on secondary amine
            "N([CH2,CH3,c])C(=O)OC(C)(C)C"  # Boc on substituted amine
        ]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection not found
        
        if self.timing == "late":
            # Late-stage protection preferred (higher depth fraction is better)
            return 10 * x
        elif self.timing == "early":
            # Early-stage protection preferred (lower depth fraction is better)
            return 10 * (1 - x)
        else:
            # Any timing acceptable
            return 10 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection/deprotection"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Count Boc groups in reactants and products
            reactant_boc_count = sum(self._count_boc_groups(mol) for mol in reactants)
            product_boc_count = sum(self._count_boc_groups(mol) for mol in products)
            
            # Check if Boc protection occurred (more Boc groups in products)
            # or Boc deprotection occurred (fewer Boc groups in products)
            return reactant_boc_count != product_boc_count
            
        except Exception:
            return False
    
    def _count_boc_groups(self, mol) -> int:
        """Count number of Boc protecting groups in a molecule"""
        if mol is None:
            return 0
            
        count = 0
        for pattern_smarts in self.boc_patterns:
            try:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    count += len(matches)
            except Exception:
                continue
                
        return count
