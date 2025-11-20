"""Generated evaluation code for: Early diketone formation via linear assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyDiketoneFormation(BaseScoring):
    """
    Evaluates whether diketone formation occurs early in the synthesis route
    through linear assembly before aryl substituent introduction.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")  # early/late preference
        self.max_early_depth = config.get("max_early_depth", 0.3)  # fraction of route depth
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Diketone formation doesn't happen
        
        if self.timing_preference == "early":
            if x <= self.max_early_depth:
                return 10  # Perfect score for early formation
            else:
                # Linearly decrease score as formation gets later
                return max(0, 10 * (1 - x))
        else:
            # Late preference (reverse scoring)
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a diketone through linear assembly
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1].split(".")
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains diketone pattern
            diketone_patterns = [
                "[#6](=O)[#6][#6](=O)",  # Generic diketone C(=O)CC(=O)
                "[#6](=O)[#6]=[#6](=O)",  # Alpha-diketone C(=O)C=C(=O)
                "[#6](=O)[#6][#6][#6](=O)"  # 1,4-diketone
            ]
            
            product_has_diketone = any(
                product_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                for pattern in diketone_patterns
            )
            
            # Check if reactants lack diketone (formation reaction)
            reactants_have_diketone = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                    for pattern in diketone_patterns)
                for mol in reactant_mols
            )
            
            # Diketone formation: product has it, reactants don't
            diketone_formed = product_has_diketone and not reactants_have_diketone
            
            if not diketone_formed:
                return False
            
            # Check for linear assembly pattern
            return self._is_linear_assembly(product_mol, reactant_mols)
            
        except Exception:
            return False
    
    def _is_linear_assembly(self, product_mol, reactant_mols) -> bool:
        """
        Check if the reaction represents linear assembly (chain extension)
        rather than cyclization or complex rearrangement
        """
        # Linear assembly indicators:
        # 1. Number of reactants is small (typically 2-3)
        # 2. Product has more bonds than largest reactant
        # 3. No significant ring formation in this step
        
        if len(reactant_mols) > 3:
            return False  # Too many components for linear assembly
            
        try:
            product_bonds = product_mol.GetNumBonds()
            max_reactant_bonds = max(mol.GetNumBonds() for mol in reactant_mols)
            
            # Linear assembly should increase bond count
            if product_bonds <= max_reactant_bonds:
                return False
                
            # Check that we're not forming large ring systems in this step
            product_rings = product_mol.GetRingInfo().NumRings()
            total_reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactant_mols)
            
            # Allow small ring formation but not large macrocycles
            new_rings = product_rings - total_reactant_rings
            if new_rings > 2:  # Arbitrary threshold for "linear" vs complex cyclization
                return False
                
            return True
            
        except Exception:
            return False
