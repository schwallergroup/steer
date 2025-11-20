"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Suzuki coupling reactions that form biaryl bonds.
    Rewards routes where Suzuki coupling occurs later in the synthesis (closer to the target molecule).
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late_stage")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Suzuki coupling found
        else:
            # Late-stage is better (higher depth fraction gives higher score)
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling forming a biaryl bond."""
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
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
                
            # Check for Suzuki coupling pattern: organoborane + aryl halide -> biaryl
            has_organoborane = self._has_organoborane(reactants)
            has_aryl_halide = self._has_aryl_halide(reactants)
            forms_biaryl_bond = self._forms_new_aryl_aryl_bond(reactants, products)
            
            return has_organoborane and has_aryl_halide and forms_biaryl_bond
            
        except Exception:
            return False
    
    def _has_organoborane(self, reactants) -> bool:
        """Check for presence of organoborane reagent (boronic acid/ester)."""
        # Boronic acid pattern
        boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(-O)-O")
        # Boronic ester pattern  
        boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1-O-C-C-O-1")
        # Simple organoborane
        organoborane_pattern = Chem.MolFromSmarts("[#6]-B")
        
        for mol in reactants:
            if mol and (mol.HasSubstructMatch(boronic_acid_pattern) or 
                       mol.HasSubstructMatch(boronic_ester_pattern) or
                       mol.HasSubstructMatch(organoborane_pattern)):
                return True
        return False
    
    def _has_aryl_halide(self, reactants) -> bool:
        """Check for presence of aryl halide."""
        # Aryl halides (Br, I, Cl on aromatic ring)
        aryl_bromide = Chem.MolFromSmarts("c-Br")
        aryl_iodide = Chem.MolFromSmarts("c-I") 
        aryl_chloride = Chem.MolFromSmarts("c-Cl")
        aryl_triflate = Chem.MolFromSmarts("c-OS(=O)(=O)C(F)(F)F")
        
        for mol in reactants:
            if mol and (mol.HasSubstructMatch(aryl_bromide) or
                       mol.HasSubstructMatch(aryl_iodide) or 
                       mol.HasSubstructMatch(aryl_chloride) or
                       mol.HasSubstructMatch(aryl_triflate)):
                return True
        return False
    
    def _forms_new_aryl_aryl_bond(self, reactants, products) -> bool:
        """Check if a new aryl-aryl bond is formed between reactants and products."""
        # Count biaryl bonds in reactants vs products
        biaryl_pattern = Chem.MolFromSmarts("c-c")  # Aromatic carbon to aromatic carbon
        
        reactant_biaryl_count = 0
        for mol in reactants:
            if mol:
                reactant_biaryl_count += len(mol.GetSubstructMatches(biaryl_pattern))
                
        product_biaryl_count = 0  
        for mol in products:
            if mol:
                product_biaryl_count += len(mol.GetSubstructMatches(biaryl_pattern))
                
        # New biaryl bond formed if product has more aryl-aryl bonds
        return product_biaryl_count > reactant_biaryl_count
