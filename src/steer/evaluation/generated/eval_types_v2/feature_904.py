"""Generated evaluation code for: Late stage amide coupling for convergent assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage amide coupling reactions that
    assemble the target molecule from two complex fragments.
    """
    
    def __init__(self, config: Dict):
        self.min_fragments = config.get("fragments", 2)
        self.timing_preference = config.get("timing", "late")  # "late" or "any"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        else:
            # Late-stage coupling is better (lower depth fraction)
            # Convert to 0-10 score where late coupling gets higher score
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is an amide coupling between two complex fragments"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            products, reactants = rxn_smiles.split(">>")
            product_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not product_mol or len(reactant_mols) < self.min_fragments:
                return False
            
            # Check if this is an amide formation reaction
            if not self._is_amide_formation(product_mol, reactant_mols):
                return False
            
            # Check if reactants are complex fragments (not simple starting materials)
            if not self._are_complex_fragments(reactant_mols):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_amide_formation(self, product_mol, reactant_mols) -> bool:
        """Check if the reaction forms an amide bond"""
        # Count amide bonds in product
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        if not amide_pattern:
            return False
            
        product_amides = len(product_mol.GetSubstructMatches(amide_pattern))
        
        # Count amide bonds in all reactants combined
        reactant_amides = sum(len(mol.GetSubstructMatches(amide_pattern)) 
                            for mol in reactant_mols if mol is not None)
        
        # Check if net amide bonds increased
        if product_amides <= reactant_amides:
            return False
        
        # Look for carboxylic acid/ester and amine/amide coupling patterns
        acid_pattern = Chem.MolFromSmarts("[C](=[O])[O]")  # Carboxylic acid/ester
        amine_pattern = Chem.MolFromSmarts("[N]")  # Amine/amide nitrogen
        
        has_acid_source = any(mol.HasSubstructMatch(acid_pattern) 
                            for mol in reactant_mols if mol is not None)
        has_amine_source = any(mol.HasSubstructMatch(amine_pattern) 
                             for mol in reactant_mols if mol is not None)
        
        return has_acid_source and has_amine_source
    
    def _are_complex_fragments(self, reactant_mols) -> bool:
        """Check if reactants are complex fragments rather than simple starting materials"""
        complex_fragments = 0
        
        for mol in reactant_mols:
            if mol is None:
                continue
                
            # Skip small molecules (likely reagents/catalysts)
            if mol.GetNumAtoms() < 8:
                continue
                
            # Count as complex if it has multiple rings or significant size
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            num_atoms = mol.GetNumAtoms()
            
            # Consider complex if: multiple rings OR large single ring system OR significant acyclic structure
            if num_rings >= 2 or (num_rings >= 1 and num_atoms >= 15) or (num_rings == 0 and num_atoms >= 20):
                complex_fragments += 1
        
        return complex_fragments >= self.min_fragments
