"""Generated evaluation code for: Cbz protecting group strategy on piperazine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzPiperazineProtection(BaseScoring):
    """
    Evaluates synthesis routes for Cbz protecting group strategy on piperazine.
    Checks for the presence of Cbz-protected piperazine and subsequent deprotection.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # SMARTS patterns for detection
        self.piperazine_pattern = "[N;R1]1CC[N;R1]CC1"  # Piperazine ring
        self.cbz_pattern = "[N;R1]C(=O)O[CH2]c1ccccc1"  # Cbz protecting group
        self.cbz_piperazine_pattern = "[N;R1]1CC[N;R1](C(=O)O[CH2]c2ccccc2)CC1"  # Cbz-protected piperazine
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0
        else:
            # Earlier protection is generally better for synthetic strategy
            return 1 - x
    
    def hit_condition(self, d):
        """
        Check if this reaction involves Cbz protection/deprotection of piperazine.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not react_mols:
                return False
            
            # Check for Cbz protection: piperazine + Cbz reagent -> Cbz-protected piperazine
            if self._is_cbz_protection(prod_mol, react_mols):
                return True
            
            # Check for Cbz deprotection: Cbz-protected piperazine -> piperazine
            if self._is_cbz_deprotection(prod_mol, react_mols):
                return True
                
        except Exception:
            return False
        
        return False
    
    def _is_cbz_protection(self, product, reactants):
        """Check if reaction is Cbz protection of piperazine."""
        # Product should contain Cbz-protected piperazine
        if not product.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_piperazine_pattern)):
            return False
        
        # One reactant should be piperazine, another should be Cbz reagent
        has_piperazine = False
        has_cbz_reagent = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(Chem.MolFromSmarts(self.piperazine_pattern)):
                # Check if it's unprotected piperazine (not already Cbz-protected)
                if not reactant.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_pattern)):
                    has_piperazine = True
            
            # Common Cbz reagents: CbzCl (benzyl chloroformate)
            cbz_reagent_pattern = "ClC(=O)O[CH2]c1ccccc1"
            if reactant.HasSubstructMatch(Chem.MolFromSmarts(cbz_reagent_pattern)):
                has_cbz_reagent = True
        
        return has_piperazine and has_cbz_reagent
    
    def _is_cbz_deprotection(self, product, reactants):
        """Check if reaction is Cbz deprotection of piperazine."""
        # Product should contain free piperazine
        if not product.HasSubstructMatch(Chem.MolFromSmarts(self.piperazine_pattern)):
            return False
        
        # Check that product doesn't have Cbz protection
        if product.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_pattern)):
            return False
        
        # At least one reactant should be Cbz-protected piperazine
        has_cbz_protected_piperazine = False
        for reactant in reactants:
            if reactant.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_piperazine_pattern)):
                has_cbz_protected_piperazine = True
                break
        
        return has_cbz_protected_piperazine
