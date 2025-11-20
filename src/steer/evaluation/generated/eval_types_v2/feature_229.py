"""Generated evaluation code for: Multiple protecting group cycling on pyrazole nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazoleProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for multiple protecting group cycling on pyrazole nitrogen.
    Tracks PMB protection, deprotection to H, and final methylation across the route.
    """
    
    def __init__(self, config):
        self.heteroatom = config.get("heteroatom", "N")
        self.protecting_groups = config.get("protecting_groups", ["PMB", "H", "methyl"])
        self.cycle_count = config.get("cycle_count", 2)
        
        # Define SMARTS patterns for pyrazole and protecting groups
        self.pyrazole_pattern = "c1cc[nH]n1"  # Basic pyrazole pattern
        self.pmb_pattern = "[#7]-[CH2]-c1ccc(OC)cc1"  # PMB on nitrogen
        self.methyl_n_pattern = "[#7]-[CH3]"  # Methyl on nitrogen
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group changes across reactions
        pg_changes = []
        for rxn in reactions:
            change = self.detect_protecting_group_change(rxn)
            if change:
                pg_changes.append(change)
        
        # Check if we have the required cycle pattern
        condition = self.has_required_cycling(pg_changes)
        return condition, len(reactions)
    
    def detect_protecting_group_change(self, rxn):
        """Detect protecting group changes on pyrazole nitrogen"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = rxn_parts[0]
        products = rxn_parts[1].split(".")[0]  # Take main product
        
        try:
            reactant_mol = Chem.MolFromSmiles(reactants.split(".")[0])  # Main reactant
            product_mol = Chem.MolFromSmiles(products)
            
            if not reactant_mol or not product_mol:
                return None
            
            # Check if both contain pyrazole
            pyrazole_smarts = Chem.MolFromSmarts(self.pyrazole_pattern)
            if not (reactant_mol.HasSubstructMatch(pyrazole_smarts) and 
                   product_mol.HasSubstructMatch(pyrazole_smarts)):
                return None
            
            # Determine protecting groups in reactant and product
            reactant_pg = self.identify_protecting_group(reactant_mol)
            product_pg = self.identify_protecting_group(product_mol)
            
            if reactant_pg != product_pg:
                return {"from": reactant_pg, "to": product_pg}
                
        except Exception:
            pass
            
        return None
    
    def identify_protecting_group(self, mol):
        """Identify the protecting group on pyrazole nitrogen"""
        pmb_smarts = Chem.MolFromSmarts(self.pmb_pattern)
        methyl_smarts = Chem.MolFromSmarts(self.methyl_n_pattern)
        free_nh_smarts = Chem.MolFromSmarts("c1cc[nH]n1")
        
        if mol.HasSubstructMatch(pmb_smarts):
            return "PMB"
        elif mol.HasSubstructMatch(methyl_smarts):
            return "methyl"
        elif mol.HasSubstructMatch(free_nh_smarts):
            return "H"
        else:
            return "unknown"
    
    def has_required_cycling(self, pg_changes):
        """Check if the protecting group changes match the required cycling pattern"""
        if len(pg_changes) < self.cycle_count:
            return False
        
        # Look for the pattern: something -> PMB -> H -> methyl (or similar cycles)
        target_groups = set(self.protecting_groups)
        observed_groups = set()
        
        # Count transitions between different protecting groups
        valid_transitions = 0
        for change in pg_changes:
            from_pg = change["from"]
            to_pg = change["to"]
            
            if from_pg in target_groups and to_pg in target_groups:
                observed_groups.add(from_pg)
                observed_groups.add(to_pg)
                valid_transitions += 1
        
        # Check if we have enough transitions and cover the required groups
        has_pmb_cycle = any(change["from"] == "PMB" or change["to"] == "PMB" 
                           for change in pg_changes)
        has_methylation = any(change["to"] == "methyl" for change in pg_changes)
        has_deprotection = any(change["to"] == "H" for change in pg_changes)
        
        return (valid_transitions >= self.cycle_count and 
                has_pmb_cycle and 
                has_methylation and 
                has_deprotection and
                len(observed_groups) >= 3)
