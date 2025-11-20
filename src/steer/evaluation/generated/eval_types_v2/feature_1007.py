"""Generated evaluation code for: Cbz protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzProtectingGroupStrategy(BaseScoring):
    """
    Evaluates the use of Cbz (benzyloxycarbonyl) protecting group strategy for amines.
    Checks if Cbz protection is applied to mask amines during synthetic transformations.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        
        # SMARTS patterns for Cbz-protected amine and free amine
        self.cbz_pattern = "[NH1][C](=O)[O][CH2]c1ccccc1"  # N-Cbz pattern
        self.free_amine_pattern = "[NH2,NH1]"  # Primary or secondary amine
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Cbz protection found
        else:
            # Earlier protection is generally better (allows more flexibility)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Cbz protection of an amine.
        Returns True if a free amine in the product is protected with Cbz in the reactants.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            products = [Chem.MolFromSmiles(p) for p in rxn[0].split(".")]
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not all(products) or not all(reactants):
                return False
            
            # Check if any product has a free amine
            product_has_free_amine = any(
                self._has_free_amine(mol) for mol in products if mol
            )
            
            # Check if any reactant has Cbz-protected amine
            reactant_has_cbz = any(
                self._has_cbz_protection(mol) for mol in reactants if mol
            )
            
            # Cbz protection strategy: free amine in product, Cbz-protected in reactants
            # This indicates Cbz was used as a protecting group strategy
            return product_has_free_amine and reactant_has_cbz
            
        except Exception:
            return False
    
    def _has_free_amine(self, mol) -> bool:
        """Check if molecule contains a free amine group."""
        if not mol:
            return False
        pattern = Chem.MolFromSmarts(self.free_amine_pattern)
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _has_cbz_protection(self, mol) -> bool:
        """Check if molecule contains Cbz-protected amine."""
        if not mol:
            return False
        pattern = Chem.MolFromSmarts(self.cbz_pattern)
        return mol.HasSubstructMatch(pattern) if pattern else False
