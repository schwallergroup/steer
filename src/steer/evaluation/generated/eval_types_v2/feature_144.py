"""Generated evaluation code for: Boc protection strategy for piperidine amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocPiperidineProtection(BaseScoring):
    """
    Evaluates the use of Boc protection strategy for piperidine amine.
    Checks if Boc protection/deprotection occurs on a piperidine nitrogen
    and rewards earlier implementation of this strategy.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection strategy not found
        else:
            # Earlier Boc protection is better (lower depth fraction)
            if self.condition_type == "bool":
                return 10  # Strategy found
            else:
                # Reward early use of Boc protection
                return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves Boc protection/deprotection on piperidine"""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        reactants = []
        for r_smi in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smi)
            if mol:
                reactants.append(mol)
        
        products = []
        for p_smi in products_smiles.split("."):
            mol = Chem.MolFromSmiles(p_smi)
            if mol:
                products.append(mol)
        
        # Check for Boc protection (reactant has free piperidine N, product has Boc-protected)
        if self._has_free_piperidine_n(reactants) and self._has_boc_protected_piperidine(products):
            return True
            
        # Check for Boc deprotection (reactant has Boc-protected, product has free piperidine N)
        if self._has_boc_protected_piperidine(reactants) and self._has_free_piperidine_n(products):
            return True
            
        return False
    
    def _has_free_piperidine_n(self, molecules) -> bool:
        """Check if any molecule contains a free (unsubstituted) piperidine nitrogen"""
        # Piperidine with secondary amine (free NH)
        free_piperidine_pattern = Chem.MolFromSmarts("[NH1]1CCCCC1")
        
        for mol in molecules:
            if mol and mol.HasSubstructMatch(free_piperidine_pattern):
                return True
        return False
    
    def _has_boc_protected_piperidine(self, molecules) -> bool:
        """Check if any molecule contains a Boc-protected piperidine nitrogen"""
        # Piperidine nitrogen with Boc protection: N(C(=O)OC(C)(C)C)
        boc_piperidine_pattern = Chem.MolFromSmarts("[N]1([C](=[O])[O][C]([C])([C])[C])CCCCC1")
        
        for mol in molecules:
            if mol and mol.HasSubstructMatch(boc_piperidine_pattern):
                return True
        return False
