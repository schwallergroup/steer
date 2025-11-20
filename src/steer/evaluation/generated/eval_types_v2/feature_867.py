"""Generated evaluation code for: Orthogonal protecting group strategy nosyl and silyl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Checks for orthogonal protecting group strategy using nosyl and silyl groups.
    Evaluates whether the route uses both nosyl (for amine protection) and 
    silyl ethers (TBS/TBDPS for alcohol protection) simultaneously.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["nosyl", "TBS", "TBDPS"])
        self.orthogonal = config.get("orthogonal", True)
        self.simultaneous_presence = config.get("simultaneous_presence", True)
        
        # SMARTS patterns for protecting groups
        self.nosyl_pattern = "[NH1,NH0]-S(=O)(=O)-c1ccc([N+](=O)[O-])cc1"  # Nosyl group
        self.tbs_pattern = "[OH0]-[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]"  # TBS silyl ether
        self.tbdps_pattern = "[OH0]-[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]"  # TBDPS (simplified)
        self.general_silyl_pattern = "[OH0]-[Si]"  # General silyl ether pattern
    
    def condition_depth(self, d):
        """Check if orthogonal protecting group strategy is employed."""
        reactions = self.get_rxns(d)
        
        nosyl_present = False
        silyl_present = False
        
        # Check all molecules in all reactions for protecting group patterns
        for rxn in reactions:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            # Check both reactants and products
            all_smiles = rxn_parts[0].split(".") + rxn_parts[1].split(".")
            
            for smi in all_smiles:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                        
                    # Check for nosyl protection
                    if self.has_nosyl_protection(mol):
                        nosyl_present = True
                        
                    # Check for silyl protection
                    if self.has_silyl_protection(mol):
                        silyl_present = True
                        
                except:
                    continue
        
        # Evaluate strategy condition
        if self.simultaneous_presence and self.orthogonal:
            condition = nosyl_present and silyl_present
        elif self.orthogonal:
            condition = nosyl_present or silyl_present
        else:
            condition = nosyl_present and silyl_present
            
        return condition, len(reactions)
    
    def has_nosyl_protection(self, mol):
        """Check if molecule contains nosyl protecting group."""
        try:
            nosyl_mol = Chem.MolFromSmarts(self.nosyl_pattern)
            if nosyl_mol and mol.HasSubstructMatch(nosyl_mol):
                return True
                
            # Alternative pattern for nosyl
            alt_nosyl = "[NH,N]-S(=O)(=O)-c1ccc(N(=O)=O)cc1"
            alt_nosyl_mol = Chem.MolFromSmarts(alt_nosyl)
            if alt_nosyl_mol and mol.HasSubstructMatch(alt_nosyl_mol):
                return True
                
        except:
            pass
        return False
    
    def has_silyl_protection(self, mol):
        """Check if molecule contains silyl ether protecting groups (TBS, TBDPS, etc.)."""
        try:
            # Check for general silyl ether pattern
            silyl_mol = Chem.MolFromSmarts(self.general_silyl_pattern)
            if silyl_mol and mol.HasSubstructMatch(silyl_mol):
                # Additional check for common silyl patterns
                tbs_mol = Chem.MolFromSmarts("[OH0]-[Si]([CH3])([CH3])[CH3]")
                tbdps_mol = Chem.MolFromSmarts("[OH0]-[Si](c1ccccc1)(c2ccccc2)")
                
                if (tbs_mol and mol.HasSubstructMatch(tbs_mol)) or \
                   (tbdps_mol and mol.HasSubstructMatch(tbdps_mol)):
                    return True
                    
                # Check for any silyl ether (broader pattern)
                return True
                
        except:
            pass
        return False
