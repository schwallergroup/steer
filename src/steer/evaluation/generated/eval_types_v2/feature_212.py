"""Generated evaluation code for: Sequential phenol protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialPhenolProtection(MultiRxnCondBase):
    """
    Evaluates routes for sequential phenol protecting group strategy.
    Checks for the use of TBS and benzyl ethers to differentiate between phenols
    for sequential alkylation reactions.
    """
    
    def __init__(self, config):
        self.target_functional_group = config["parameters"]["functional_group"]
        self.protecting_groups = config["parameters"]["protecting_groups"]
        self.strategy = config["parameters"]["strategy"]
        
        # Define SMARTS patterns for phenols and protecting groups
        self.phenol_pattern = "[OH1][c]"  # Phenol OH
        self.tbs_pattern = "[OH0]([c])[Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]"  # TBS-protected phenol
        self.benzyl_pattern = "[OH0]([c])[CH2][c]1[cH][cH][cH][cH][cH]1"  # Benzyl-protected phenol
        
        # Alkylation pattern (C-O bond formation at phenolic oxygen)
        self.alkylation_pattern = "[OH0]([c])[CH2,CH3,C]"
    
    def condition_depth(self, d) -> tuple:
        """Check if sequential phenol protection strategy is employed."""
        reactions = self.get_rxns(d)
        
        has_tbs_protection = False
        has_benzyl_protection = False
        has_sequential_alkylation = False
        
        # Track protection and deprotection events
        protection_events = []
        alkylation_events = []
        
        for i, rxn in enumerate(reactions):
            # Check for TBS protection/deprotection
            if self.detect_tbs_protection(rxn):
                has_tbs_protection = True
                protection_events.append(('TBS_protect', i))
            elif self.detect_tbs_deprotection(rxn):
                protection_events.append(('TBS_deprotect', i))
            
            # Check for benzyl protection/deprotection
            if self.detect_benzyl_protection(rxn):
                has_benzyl_protection = True
                protection_events.append(('benzyl_protect', i))
            elif self.detect_benzyl_deprotection(rxn):
                protection_events.append(('benzyl_deprotect', i))
            
            # Check for alkylation of phenol
            if self.detect_phenol_alkylation(rxn):
                alkylation_events.append(('alkylation', i))
        
        # Check if we have both protecting groups
        has_both_protecting_groups = has_tbs_protection and has_benzyl_protection
        
        # Check for sequential alkylation (multiple alkylation events)
        has_sequential_alkylation = len(alkylation_events) >= 2
        
        # Strategy is successful if we use both protecting groups and have sequential reactions
        condition = has_both_protecting_groups and has_sequential_alkylation
        
        return condition, len(reactions)
    
    def detect_tbs_protection(self, rxn):
        """Detect TBS protection of phenol."""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Check if product has TBS-protected phenol and reactants have free phenol
        if prod_mol and react_mols:
            prod_has_tbs = prod_mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbs_pattern))
            reactants_have_phenol = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)) 
                                      for mol in react_mols)
            return prod_has_tbs and reactants_have_phenol
        return False
    
    def detect_tbs_deprotection(self, rxn):
        """Detect TBS deprotection to reveal phenol."""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        if prod_mol and react_mols:
            reactants_have_tbs = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbs_pattern)) 
                                   for mol in react_mols)
            prod_has_phenol = prod_mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern))
            return reactants_have_tbs and prod_has_phenol
        return False
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection of phenol."""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        if prod_mol and react_mols:
            prod_has_benzyl = prod_mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_pattern))
            reactants_have_phenol = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)) 
                                      for mol in react_mols)
            return prod_has_benzyl and reactants_have_phenol
        return False
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl deprotection to reveal phenol."""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        if prod_mol and react_mols:
            reactants_have_benzyl = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_pattern)) 
                                      for mol in react_mols)
            prod_has_phenol = prod_mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern))
            return reactants_have_benzyl and prod_has_phenol
        return False
    
    def detect_phenol_alkylation(self, rxn):
        """Detect alkylation of phenolic OH group."""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        if prod_mol and react_mols:
            reactants_have_phenol = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)) 
                                      for mol in react_mols)
            prod_has_alkylated = prod_mol.HasSubstructMatch(Chem.MolFromSmarts(self.alkylation_pattern))
            return reactants_have_phenol and prod_has_alkylated
        return False
    
    def parse_reaction(self, rxn_smiles):
        """Parse reaction SMILES to get product and reactant molecules."""
        try:
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return None, None
            
            reactant_smiles = parts[0].split(".")
            product_smiles = parts[1]
            
            prod_mol = Chem.MolFromSmiles(product_smiles)
            react_mols = [Chem.MolFromSmiles(smi) for smi in reactant_smiles]
            react_mols = [mol for mol in react_mols if mol is not None]
            
            return prod_mol, react_mols
        except:
            return None, None
