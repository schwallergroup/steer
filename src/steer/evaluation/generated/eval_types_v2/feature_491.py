"""Generated evaluation code for: Orthogonal protecting group strategy for alcohols"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses orthogonal protecting group strategy for alcohols.
    Specifically checks for the presence of both TBDMS and acetate protecting groups
    applied to different alcohol positions for differential functionalization.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "orthogonal")
        self.functional_groups = config.get("functional_groups", ["alcohol"])
        self.protecting_groups = config.get("protecting_groups", ["TBDMS", "acetate"])
        self.purpose = config.get("purpose", "differential_functionalization")
        
        # SMARTS patterns for detecting protecting group installation/removal
        self.tbdms_protection_pattern = "[OH1][CH2,CH1,CH0]>>[O][Si](C)(C)C(C)(C)C"
        self.acetate_protection_pattern = "[OH1][CH2,CH1,CH0]>>[O]C(=O)C"
        
        # Substructure patterns for protected alcohols
        self.tbdms_protected = Chem.MolFromSmarts("[O][Si](C)(C)C(C)(C)C")
        self.acetate_protected = Chem.MolFromSmarts("[O]C(=O)C")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_tbdms_protection = False
        has_acetate_protection = False
        tbdms_sites = set()
        acetate_sites = set()
        
        # Check each reaction for protecting group installation
        for rxn in reactions:
            if self.detect_tbdms_protection(rxn):
                has_tbdms_protection = True
                tbdms_sites.update(self.get_protection_sites(rxn, "TBDMS"))
                
            if self.detect_acetate_protection(rxn):
                has_acetate_protection = True
                acetate_sites.update(self.get_protection_sites(rxn, "acetate"))
        
        # Check for orthogonal strategy: both protecting groups used on different sites
        orthogonal_strategy = (has_tbdms_protection and has_acetate_protection and 
                             len(tbdms_sites.intersection(acetate_sites)) == 0 and
                             len(tbdms_sites) > 0 and len(acetate_sites) > 0)
        
        return orthogonal_strategy, len(reactions)
    
    def detect_tbdms_protection(self, rxn):
        """Detect TBDMS protection reaction"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            # Check if reaction introduces TBDMS group
            reactant_has_tbdms = any(mol.HasSubstructMatch(self.tbdms_protected) for mol in reactants if mol)
            product_has_tbdms = any(mol.HasSubstructMatch(self.tbdms_protected) for mol in products if mol)
            
            return not reactant_has_tbdms and product_has_tbdms
            
        except:
            return False
    
    def detect_acetate_protection(self, rxn):
        """Detect acetate protection reaction"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            # Check if reaction introduces acetate group
            reactant_has_acetate = any(mol.HasSubstructMatch(self.acetate_protected) for mol in reactants if mol)
            product_has_acetate = any(mol.HasSubstructMatch(self.acetate_protected) for mol in products if mol)
            
            return not reactant_has_acetate and product_has_acetate
            
        except:
            return False
    
    def get_protection_sites(self, rxn, protection_type):
        """Get atom map numbers of sites where protection occurs"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return set()
                
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            sites = set()
            
            pattern = self.tbdms_protected if protection_type == "TBDMS" else self.acetate_protected
            
            for mol in products:
                if mol and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        # Get atom map number of the oxygen atom in the protecting group
                        oxygen_idx = match[0]  # First atom in pattern is oxygen
                        atom = mol.GetAtomWithIdx(oxygen_idx)
                        if atom.GetAtomMapNum() > 0:
                            sites.add(atom.GetAtomMapNum())
            
            return sites
            
        except:
            return set()
