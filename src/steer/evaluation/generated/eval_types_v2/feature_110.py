"""Generated evaluation code for: Late stage urea formation coupling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageUreaFormation(BaseScoring):
    """
    Evaluates whether urea formation occurs late in the synthesis route using
    activated carbamate coupling strategy to avoid isocyanate hazards.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")  # "early", "late", or "any"
        self.require_activated_carbamate = config.get("require_activated_carbamate", True)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Urea formation doesn't happen
        
        if self.timing_preference == "late":
            return 1 - x  # Late-stage is better, so invert depth fraction
        elif self.timing_preference == "early":
            return x  # Early-stage is better
        else:  # "any"
            return 1  # Just presence matters
    
    def hit_condition(self, d) -> bool:
        """
        Detects urea formation reaction by checking for:
        1. Urea bond formation (N-C(=O)-N pattern)
        2. Optional: activated carbamate starting material (phenyl carbamate)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".")]
            react_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
            
            # Filter out None molecules
            prod_mols = [mol for mol in prod_mols if mol is not None]
            react_mols = [mol for mol in react_mols if mol is not None]
            
            if not prod_mols or not react_mols:
                return False
            
            # Check for urea formation
            if not self._is_urea_formation(react_mols, prod_mols):
                return False
            
            # If activated carbamate is required, check for it
            if self.require_activated_carbamate:
                return self._has_activated_carbamate(react_mols)
            
            return True
            
        except Exception:
            return False
    
    def _is_urea_formation(self, reactants, products) -> bool:
        """
        Check if reaction forms a urea bond by comparing urea count in reactants vs products
        """
        # Urea pattern: N-C(=O)-N
        urea_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[NX3]")
        
        if urea_pattern is None:
            return False
        
        # Count urea groups in reactants
        reactant_urea_count = sum(
            len(mol.GetSubstructMatches(urea_pattern)) 
            for mol in reactants
        )
        
        # Count urea groups in products  
        product_urea_count = sum(
            len(mol.GetSubstructMatches(urea_pattern))
            for mol in products
        )
        
        # Urea formation should increase urea count
        return product_urea_count > reactant_urea_count
    
    def _has_activated_carbamate(self, reactants) -> bool:
        """
        Check for activated carbamate (e.g., phenyl carbamate) in reactants
        """
        # Phenyl carbamate pattern: N-C(=O)-O-Ar
        phenyl_carbamate_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2]c1ccccc1")
        
        # General activated carbamate pattern (aryl or other good leaving groups)
        activated_carbamate_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2][cR,$(O[CX4](F)(F)F)]")
        
        if phenyl_carbamate_pattern is None and activated_carbamate_pattern is None:
            return False
        
        for mol in reactants:
            # Check for phenyl carbamate first
            if phenyl_carbamate_pattern and mol.HasSubstructMatch(phenyl_carbamate_pattern):
                return True
            # Check for other activated carbamates
            if activated_carbamate_pattern and mol.HasSubstructMatch(activated_carbamate_pattern):
                return True
                
        return False
