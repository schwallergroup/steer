"""Generated evaluation code for: Boc protection for cross-coupling compatibility"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(BaseScoring):
    """
    Evaluates if Boc protection is used for amine groups to enable cross-coupling compatibility.
    Checks for the presence of Boc-protected amines in reactions and rewards earlier protection.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Boc protection found
        else:
            # Earlier protection is better (lower depth fraction)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of amines"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        # Split reaction into product and reactants
        parts = mapped_rxn.split(">>")
        if len(parts) != 2:
            return False
        
        product_smiles = parts[0]
        reactants_smiles = parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check for Boc protection: free amine -> Boc-protected amine
            return self._is_boc_protection_reaction(product_mol, reactant_mols)
            
        except Exception:
            return False
    
    def _is_boc_protection_reaction(self, product_mol, reactant_mols) -> bool:
        """Check if reaction converts free amine to Boc-protected amine"""
        
        # Boc-protected amine pattern: NC(=O)OC(C)(C)C
        boc_amine_pattern = Chem.MolFromSmarts("[NH1,NH2]C(=O)OC(C)(C)C")
        # Free amine patterns
        primary_amine_pattern = Chem.MolFromSmarts("[NH2]")
        secondary_amine_pattern = Chem.MolFromSmarts("[NH1]")
        
        if not boc_amine_pattern:
            return False
        
        # Check if product has Boc-protected amine
        has_boc_in_product = product_mol.HasSubstructMatch(boc_amine_pattern)
        
        if not has_boc_in_product:
            return False
        
        # Check if reactants have free amines
        has_free_amine_in_reactants = False
        for reactant in reactant_mols:
            if (reactant.HasSubstructMatch(primary_amine_pattern) or 
                reactant.HasSubstructMatch(secondary_amine_pattern)):
                # Make sure it's not already Boc-protected
                if not reactant.HasSubstructMatch(boc_amine_pattern):
                    has_free_amine_in_reactants = True
                    break
        
        # Also check for presence of Boc2O or similar Boc reagent
        boc_reagent_pattern = Chem.MolFromSmarts("C(=O)OC(=O)OC(C)(C)C")  # Boc2O
        has_boc_reagent = any(reactant.HasSubstructMatch(boc_reagent_pattern) 
                             for reactant in reactant_mols if boc_reagent_pattern)
        
        return has_free_amine_in_reactants and (has_boc_reagent or len(reactant_mols) > 1)
