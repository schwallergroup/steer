"""Generated evaluation code for: Benzylic bromination on unprotected aniline"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylicBrominationUnprotectedAniline(BaseScoring):
    """
    Checks if radical bromination occurs on a substrate containing an unprotected aniline group.
    Detects benzylic bromination reactions where a free aniline NH2 group is present.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met in route
        else:
            # Earlier bromination with unprotected aniline is riskier/less ideal
            return x  # Higher score for later stage protection strategy

    def hit_condition(self, d) -> bool:
        """Check if this reaction is benzylic bromination on unprotected aniline"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mol = Chem.MolFromSmiles(reactant_smiles.split(".")[0])  # Main reactant
            
            if not product_mol or not reactant_mol:
                return False
                
            # Check if bromination occurred (Br added to product)
            if not self._is_bromination_reaction(reactant_mol, product_mol):
                return False
                
            # Check if reactant has unprotected aniline
            if not self._has_unprotected_aniline(reactant_mol):
                return False
                
            # Check if bromination is benzylic
            if not self._is_benzylic_bromination(reactant_mol, product_mol):
                return False
                
            return True
            
        except Exception:
            return False

    def _is_bromination_reaction(self, reactant, product):
        """Check if reaction involves bromination"""
        # Count bromine atoms in reactant vs product
        reactant_br_count = sum(1 for atom in reactant.GetAtoms() if atom.GetSymbol() == 'Br')
        product_br_count = sum(1 for atom in product.GetAtoms() if atom.GetSymbol() == 'Br')
        
        return product_br_count > reactant_br_count

    def _has_unprotected_aniline(self, mol):
        """Check if molecule contains unprotected aniline (free NH2 on aromatic ring)"""
        # Pattern for aniline: aromatic carbon connected to NH2
        aniline_pattern = Chem.MolFromSmarts("[cH0,cH1:1]-[NH2:2]")
        if not aniline_pattern:
            return False
            
        matches = mol.HasSubstructMatch(aniline_pattern)
        if not matches:
            return False
            
        # Additional check: ensure NH2 is not part of amide or other protected forms
        protected_patterns = [
            "[NH1]C=O",  # Amide
            "[NH1]S(=O)=O",  # Sulfonamide
            "[NH1]C(=O)OC",  # Carbamate
        ]
        
        for pattern_smarts in protected_patterns:
            protected_pattern = Chem.MolFromSmarts(pattern_smarts)
            if protected_pattern and mol.HasSubstructMatch(protected_pattern):
                return False
                
        return True

    def _is_benzylic_bromination(self, reactant, product):
        """Check if bromination occurred at benzylic position"""
        # Pattern for benzylic carbon: aromatic ring connected to aliphatic carbon
        benzylic_pattern = Chem.MolFromSmarts("[c:1]-[CH3,CH2,CH:2]")
        if not benzylic_pattern:
            return False
            
        # Check if product has Br at position that was benzylic H in reactant
        product_benzylic_br = Chem.MolFromSmarts("[c]-[CH2,CH1,C:1]-[Br:2]")
        if not product_benzylic_br:
            return False
            
        return (reactant.HasSubstructMatch(benzylic_pattern) and 
                product.HasSubstructMatch(product_benzylic_br))
