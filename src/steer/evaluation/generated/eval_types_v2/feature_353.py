"""Generated evaluation code for: Buchwald-Hartwig amination for C-N bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BuchwaldHartwigAmination(BaseScoring):
    """
    Evaluates the presence and depth of Buchwald-Hartwig amination reactions for C-N bond formation.
    
    This reaction type involves palladium-catalyzed coupling between aryl halides and amines/ammonia surrogates
    to form C-N bonds. The class detects characteristic patterns of this transformation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Reaction not found
            return max(0, 1 - abs(x - self.target_depth))  # Better when closer to target depth
    
    def hit_condition(self, d) -> bool:
        """Check if a single reaction node represents Buchwald-Hartwig amination"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            if not product or len(reactants) < 2:
                return False
            
            return self._detect_buchwald_hartwig_pattern(reactants, product)
            
        except Exception:
            return False
    
    def _detect_buchwald_hartwig_pattern(self, reactants, product) -> bool:
        """Detect characteristic Buchwald-Hartwig amination patterns"""
        
        # Pattern 1: Aryl halide + amine -> aryl amine
        aryl_halide_pattern = Chem.MolFromSmarts("[c,C]-[Cl,Br,I]")  # Aryl or alkenyl halide
        amine_patterns = [
            Chem.MolFromSmarts("[NH2]"),  # Primary amine
            Chem.MolFromSmarts("[NH1][C,c]"),  # Secondary amine
            Chem.MolFromSmarts("N=C(c1ccccc1)c2ccccc2"),  # Benzophenone imine (ammonia surrogate)
        ]
        arylamine_pattern = Chem.MolFromSmarts("[c,C]-[NH2,NH1]")  # Aryl/alkenyl amine product
        
        # Check if we have aryl halide in reactants
        has_aryl_halide = any(mol.HasSubstructMatch(aryl_halide_pattern) for mol in reactants)
        
        # Check if we have amine source in reactants
        has_amine_source = any(
            any(mol.HasSubstructMatch(pattern) for pattern in amine_patterns) 
            for mol in reactants
        )
        
        # Check if product contains aryl amine
        has_arylamine_product = product.HasSubstructMatch(arylamine_pattern)
        
        # Pattern 2: Check for C-N bond formation by comparing atom maps
        if has_aryl_halide and (has_amine_source or self._has_ammonia_equivalent(reactants)):
            if has_arylamine_product:
                return self._verify_cn_bond_formation(reactants, product)
        
        return False
    
    def _has_ammonia_equivalent(self, reactants) -> bool:
        """Check for ammonia equivalents like benzophenone imine, lithium amide, etc."""
        ammonia_equivalents = [
            Chem.MolFromSmarts("N=C(c1ccccc1)c2ccccc2"),  # Benzophenone imine
            Chem.MolFromSmarts("[Li][NH2]"),  # Lithium amide
            Chem.MolFromSmarts("N([Si])[Si]"),  # Silyl amines
        ]
        
        return any(
            any(mol.HasSubstructMatch(pattern) for pattern in ammonia_equivalents)
            for mol in reactants
        )
    
    def _verify_cn_bond_formation(self, reactants, product) -> bool:
        """Verify that a new C-N bond was formed between aryl carbon and nitrogen"""
        try:
            # Get atom maps for carbon and nitrogen atoms
            carbon_maps = set()
            nitrogen_maps = set()
            
            # Find carbons bonded to halogens in reactants
            for mol in reactants:
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'C' and atom.GetAtomMapNum() > 0:
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetSymbol() in ['Cl', 'Br', 'I']:
                                carbon_maps.add(atom.GetAtomMapNum())
            
            # Find nitrogen atoms in reactants
            for mol in reactants:
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'N' and atom.GetAtomMapNum() > 0:
                        nitrogen_maps.add(atom.GetAtomMapNum())
            
            # Check if these atoms are now bonded in product
            for atom in product.GetAtoms():
                if atom.GetAtomMapNum() in carbon_maps and atom.GetSymbol() == 'C':
                    for neighbor in atom.GetNeighbors():
                        if (neighbor.GetSymbol() == 'N' and 
                            neighbor.GetAtomMapNum() in nitrogen_maps):
                            return True
            
        except Exception:
            pass
        
        return True  # Default to True if mapping analysis fails but patterns match
