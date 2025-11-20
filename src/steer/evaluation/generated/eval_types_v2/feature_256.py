"""Generated evaluation code for: Sequential protecting group cycling on phenols"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for sequential protecting group cycling on phenols.
    Detects MOM protection followed by acetylation then MOM deprotection sequences.
    """
    
    def __init__(self, config):
        self.functional_group = config["functional_group"]
        self.strategy_type = config["strategy_type"]
        self.protecting_groups = config["protecting_groups"]
        self.cycle_count = config["cycle_count"]
        
        # Define SMARTS patterns for phenol protecting group transformations
        self.phenol_pattern = "[OH1][cR6]"  # Phenolic OH
        self.mom_protected_phenol = "[O]([CH2][O][CH3])[cR6]"  # MOM-protected phenol
        self.acetyl_protected_phenol = "[O](C(=O)[CH3])[cR6]"  # Acetyl-protected phenol
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the route contains sequential protecting group cycling."""
        reactions = self.get_rxns(d)
        
        # Track the sequence of protecting group operations
        sequence = []
        
        for rxn in reactions:
            operation = self.identify_protection_operation(rxn)
            if operation:
                sequence.append(operation)
        
        # Check for the required cycling pattern
        has_cycle = self.detect_cycling_pattern(sequence)
        
        return has_cycle, len(reactions)
    
    def identify_protection_operation(self, rxn):
        """Identify the type of protecting group operation in a reaction."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for MOM protection (phenol -> MOM-protected)
            if self.has_phenol_substructure(reactant_mols) and \
               self.has_mom_protected_substructure(product_mols):
                return "MOM_protection"
            
            # Check for MOM deprotection (MOM-protected -> phenol)
            if self.has_mom_protected_substructure(reactant_mols) and \
               self.has_phenol_substructure(product_mols):
                return "MOM_deprotection"
            
            # Check for acetyl protection (phenol -> acetyl-protected)
            if self.has_phenol_substructure(reactant_mols) and \
               self.has_acetyl_protected_substructure(product_mols):
                return "acetyl_protection"
            
            # Check for acetyl deprotection (acetyl-protected -> phenol)
            if self.has_acetyl_protected_substructure(reactant_mols) and \
               self.has_phenol_substructure(product_mols):
                return "acetyl_deprotection"
            
            # Check for protection interchange (MOM -> acetyl or acetyl -> MOM)
            if self.has_mom_protected_substructure(reactant_mols) and \
               self.has_acetyl_protected_substructure(product_mols):
                return "MOM_to_acetyl"
            
            if self.has_acetyl_protected_substructure(reactant_mols) and \
               self.has_mom_protected_substructure(product_mols):
                return "acetyl_to_MOM"
                
        except Exception:
            pass
        
        return None
    
    def has_phenol_substructure(self, mols):
        """Check if any molecule contains a phenol group."""
        phenol_pattern = Chem.MolFromSmarts(self.phenol_pattern)
        if phenol_pattern is None:
            return False
        return any(mol.HasSubstructMatch(phenol_pattern) for mol in mols if mol is not None)
    
    def has_mom_protected_substructure(self, mols):
        """Check if any molecule contains a MOM-protected phenol."""
        mom_pattern = Chem.MolFromSmarts(self.mom_protected_phenol)
        if mom_pattern is None:
            return False
        return any(mol.HasSubstructMatch(mom_pattern) for mol in mols if mol is not None)
    
    def has_acetyl_protected_substructure(self, mols):
        """Check if any molecule contains an acetyl-protected phenol."""
        acetyl_pattern = Chem.MolFromSmarts(self.acetyl_protected_phenol)
        if acetyl_pattern is None:
            return False
        return any(mol.HasSubstructMatch(acetyl_pattern) for mol in mols if mol is not None)
    
    def detect_cycling_pattern(self, sequence):
        """
        Detect if the sequence contains the required protecting group cycling pattern.
        Looking for: MOM protection -> acetyl protection -> MOM deprotection (or similar cycles)
        """
        if len(sequence) < 3:
            return False
        
        # Define valid cycling patterns
        valid_cycles = [
            ["MOM_protection", "acetyl_protection", "MOM_deprotection"],
            ["acetyl_protection", "MOM_protection", "acetyl_deprotection"],
            ["MOM_protection", "MOM_to_acetyl", "acetyl_deprotection"],
            ["acetyl_protection", "acetyl_to_MOM", "MOM_deprotection"]
        ]
        
        # Check for any valid cycling pattern in the sequence
        for i in range(len(sequence) - 2):
            for cycle in valid_cycles:
                if sequence[i:i+3] == cycle:
                    return True
        
        return False
