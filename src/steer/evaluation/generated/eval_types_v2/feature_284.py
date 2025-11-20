"""Generated evaluation code for: Aldehyde protection as alcohol during alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AldehydeProtectionAlkylation(MultiRxnCondBase):
    """
    Evaluates synthesis routes for aldehyde protection as alcohol during C-alkylation reactions.
    Checks if aldehydes are protected as alcohols before alkylation steps and later oxidized back.
    """
    
    def __init__(self, config):
        self.require_protection = config.get("require_protection", True)
        self.aldehyde_pattern = Chem.MolFromSmarts("[CH1]=O")  # Aldehyde
        self.alcohol_pattern = Chem.MolFromSmarts("[CH2][OH1]")  # Primary alcohol
        self.alkylation_patterns = [
            Chem.MolFromSmarts("[C:1][CH2:2][C:3]"),  # C-C bond formation
            Chem.MolFromSmarts("[c:1][CH2:2][C:3]"),  # Aromatic C-alkylation
        ]
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_aldehyde_protection = False
        has_alkylation = False
        has_alcohol_oxidation = False
        
        for i, rxn in enumerate(reactions):
            # Check for aldehyde reduction (protection)
            if self.detect_aldehyde_reduction(rxn):
                has_aldehyde_protection = True
                
            # Check for C-alkylation reaction
            if self.detect_alkylation(rxn):
                has_alkylation = True
                
            # Check for alcohol oxidation (deprotection)
            if self.detect_alcohol_oxidation(rxn):
                has_alcohol_oxidation = True
        
        # Strategy is present if all three steps are found
        strategy_present = has_aldehyde_protection and has_alkylation and has_alcohol_oxidation
        
        if self.require_protection:
            condition_met = strategy_present
        else:
            condition_met = not strategy_present
            
        return condition_met, len(reactions)
    
    def detect_aldehyde_reduction(self, rxn):
        """Detect aldehyde -> alcohol reduction"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Check if any reactant has aldehyde and any product has corresponding alcohol
            reactant_has_aldehyde = any(mol and mol.HasSubstructMatch(self.aldehyde_pattern) 
                                     for mol in reactants if mol)
            product_has_alcohol = any(mol and mol.HasSubstructMatch(self.alcohol_pattern) 
                                    for mol in products if mol)
            
            return reactant_has_aldehyde and product_has_alcohol
            
        except:
            return False
    
    def detect_alkylation(self, rxn):
        """Detect C-alkylation reactions"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Count C-C bonds in reactants vs products
            reactant_cc_bonds = sum(self.count_alkyl_bonds(mol) for mol in reactants if mol)
            product_cc_bonds = sum(self.count_alkyl_bonds(mol) for mol in products if mol)
            
            # Alkylation should increase C-C bond count
            return product_cc_bonds > reactant_cc_bonds
            
        except:
            return False
    
    def detect_alcohol_oxidation(self, rxn):
        """Detect alcohol -> aldehyde oxidation"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Check if any reactant has alcohol and any product has corresponding aldehyde
            reactant_has_alcohol = any(mol and mol.HasSubstructMatch(self.alcohol_pattern) 
                                     for mol in reactants if mol)
            product_has_aldehyde = any(mol and mol.HasSubstructMatch(self.aldehyde_pattern) 
                                     for mol in products if mol)
            
            return reactant_has_alcohol and product_has_aldehyde
            
        except:
            return False
    
    def count_alkyl_bonds(self, mol):
        """Count C-C bonds that could be formed by alkylation"""
        if not mol:
            return 0
        count = 0
        for pattern in self.alkylation_patterns:
            matches = mol.GetSubstructMatches(pattern)
            count += len(matches)
        return count
