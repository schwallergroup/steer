"""Generated evaluation code for: Boc protection-deprotection cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionCycling(MultiRxnCondBase):
    """
    Detects Boc protection-deprotection cycling in synthesis routes.
    
    Identifies routes that install Boc protection and then remove it,
    potentially indicating inefficient protecting group strategy.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.has_cycling = config.get("has_cycling", True)
        self.target_cycle_count = config.get("cycle_count", 1)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track Boc protection/deprotection events in order
        boc_events = []
        for i, rxn in enumerate(reactions):
            if self.detect_boc_protection(rxn):
                boc_events.append(('protection', i))
            elif self.detect_boc_deprotection(rxn):
                boc_events.append(('deprotection', i))
        
        # Count protection-deprotection cycles
        cycle_count = 0
        protection_count = 0
        
        for event_type, _ in boc_events:
            if event_type == 'protection':
                protection_count += 1
            elif event_type == 'deprotection' and protection_count > 0:
                cycle_count += 1
                protection_count -= 1
        
        # Check if cycling condition is met
        if self.has_cycling:
            condition = cycle_count >= self.target_cycle_count
        else:
            condition = cycle_count == 0
            
        return condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """Detect Boc protection reactions"""
        try:
            reactants_smiles = rxn.split(">>")[0]
            products_smiles = rxn.split(">>")[1]
            
            # Boc anhydride or Boc chloride patterns
            boc_reagents = [
                "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc anhydride
                "CC(C)(C)OC(=O)Cl"  # Boc chloride
            ]
            
            has_boc_reagent = any(reagent in reactants_smiles for reagent in boc_reagents)
            
            # Check for Boc group formation (t-butoxycarbonyl)
            boc_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
            
            if has_boc_reagent:
                products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".") if Chem.MolFromSmiles(p)]
                return any(mol.HasSubstructMatch(boc_pattern) for mol in products if mol)
            
            return False
            
        except:
            return False
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection reactions"""
        try:
            reactants_smiles = rxn.split(">>")[0]
            products_smiles = rxn.split(">>")[1]
            
            # Boc group pattern
            boc_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
            
            # Check if reactant has Boc group
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".") if Chem.MolFromSmiles(r)]
            has_boc_reactant = any(mol.HasSubstructMatch(boc_pattern) for mol in reactants if mol)
            
            if has_boc_reactant:
                # Check if Boc group is removed in products
                products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".") if Chem.MolFromSmiles(p)]
                # Look for typical deprotection conditions (acid, TFA, HCl)
                deprotection_conditions = ["TFA", "HCl", "CF3COOH", "O=C(O)C(F)(F)F"]
                has_acid = any(cond in reactants_smiles for cond in deprotection_conditions)
                
                # Or check if Boc group is absent in main product
                main_products = [mol for mol in products if mol and mol.GetNumAtoms() > 5]
                boc_removed = not any(mol.HasSubstructMatch(boc_pattern) for mol in main_products if mol)
                
                return has_acid or boc_removed
            
            return False
            
        except:
            return False
