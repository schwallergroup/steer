"""Generated evaluation code for: Weinreb amide prevents over-addition in ketone formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WeinrebAmideProtection(BaseScoring):
    """
    Checks if Weinreb amide is used as a protecting group strategy for ketone formation.
    
    This evaluates whether the synthesis route uses N-methoxy-N-methylamide (Weinreb amide)
    to prevent over-addition during ketone formation from carboxylic acid derivatives.
    The Weinreb amide allows controlled addition of one equivalent of organometallic reagent
    to form ketones without further reduction to tertiary alcohols.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score. Earlier use of protection is better."""
        if x < 0:
            return 0  # Strategy not used
        else:
            # Earlier protection (lower depth) gets higher score
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Weinreb amide formation or ketone formation from Weinreb amide."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Weinreb amide pattern: N-methoxy-N-methylamide
            weinreb_pattern = Chem.MolFromSmarts("[C](=[O])[N]([CH3])[O][CH3]")
            
            # Check for Weinreb amide formation (carboxylic acid/ester -> Weinreb amide)
            has_weinreb_product = any(mol.HasSubstructMatch(weinreb_pattern) for mol in products)
            has_carboxyl_reactant = any(self._has_carboxyl_group(mol) for mol in reactants)
            
            # Check for ketone formation from Weinreb amide
            has_weinreb_reactant = any(mol.HasSubstructMatch(weinreb_pattern) for mol in reactants)
            has_ketone_product = any(self._has_ketone_group(mol) for mol in products)
            
            return (has_weinreb_product and has_carboxyl_reactant) or \
                   (has_weinreb_reactant and has_ketone_product)
                   
        except Exception:
            return False
    
    def _has_carboxyl_group(self, mol) -> bool:
        """Check if molecule has carboxylic acid or ester group."""
        if mol is None:
            return False
        # Carboxylic acid pattern
        carboxyl_acid = Chem.MolFromSmarts("[C](=[O])[OH]")
        # Ester pattern  
        ester = Chem.MolFromSmarts("[C](=[O])[O][C]")
        return mol.HasSubstructMatch(carboxyl_acid) or mol.HasSubstructMatch(ester)
    
    def _has_ketone_group(self, mol) -> bool:
        """Check if molecule has ketone group (not aldehyde, amide, or ester)."""
        if mol is None:
            return False
        # Ketone pattern: carbonyl carbon bonded to two carbons
        ketone_pattern = Chem.MolFromSmarts("[C][C](=[O])[C]")
        return mol.HasSubstructMatch(ketone_pattern)
