"""Generated evaluation code for: Benzyl protecting group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects if a synthesis route uses benzyl protecting group in a cycling pattern
    where the same intermediate appears multiple times through protection/deprotection cycles.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "benzyl")
        self.cycles = config.get("cycles", True)
        self.group_smarts = config.get("group_smarts", "[OH1]-[CH2]-c1ccccc1")
        self.benzyl_pattern = Chem.MolFromSmarts(self.group_smarts)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if benzyl protecting group cycling occurs in the route.
        Returns (condition_met, total_reactions)
        """
        reactions = self.get_rxns(d)
        
        # Track intermediates and their canonical SMILES
        intermediates = set()
        has_benzyl_protection = False
        has_benzyl_deprotection = False
        has_cycling = False
        
        # Extract all molecules in the route
        all_molecules = []
        
        for rxn in reactions:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) == 2:
                products = rxn_parts[0].split(".")
                reactants = rxn_parts[1].split(".")
                
                # Check for benzyl protection (adds benzyl group)
                if self.is_benzyl_protection(rxn):
                    has_benzyl_protection = True
                
                # Check for benzyl deprotection (removes benzyl group)
                if self.is_benzyl_deprotection(rxn):
                    has_benzyl_deprotection = True
                
                # Collect canonical SMILES of all intermediates
                for mol_smiles in products + reactants:
                    try:
                        mol = Chem.MolFromSmiles(mol_smiles)
                        if mol:
                            # Remove atom mapping for canonical comparison
                            for atom in mol.GetAtoms():
                                atom.SetAtomMapNum(0)
                            canonical_smiles = Chem.MolToSmiles(mol)
                            
                            # Check for cycling - same intermediate appears multiple times
                            if canonical_smiles in intermediates:
                                has_cycling = True
                            intermediates.add(canonical_smiles)
                            all_molecules.append(canonical_smiles)
                    except:
                        continue
        
        # Condition is met if we have both protection and deprotection with cycling
        condition = has_benzyl_protection and has_benzyl_deprotection and has_cycling
        
        return condition, len(reactions)
    
    def is_benzyl_protection(self, rxn):
        """Check if reaction involves benzyl protection"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[0].split(".") if Chem.MolFromSmiles(p)]
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".") if Chem.MolFromSmiles(r)]
            
            # Count benzyl groups in products vs reactants
            product_benzyl_count = sum(len(mol.GetSubstructMatches(self.benzyl_pattern)) for mol in products)
            reactant_benzyl_count = sum(len(mol.GetSubstructMatches(self.benzyl_pattern)) for mol in reactants)
            
            # Protection increases benzyl group count
            return product_benzyl_count > reactant_benzyl_count
            
        except:
            return False
    
    def is_benzyl_deprotection(self, rxn):
        """Check if reaction involves benzyl deprotection"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[0].split(".") if Chem.MolFromSmiles(p)]
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".") if Chem.MolFromSmiles(r)]
            
            # Count benzyl groups in products vs reactants
            product_benzyl_count = sum(len(mol.GetSubstructMatches(self.benzyl_pattern)) for mol in products)
            reactant_benzyl_count = sum(len(mol.GetSubstructMatches(self.benzyl_pattern)) for mol in reactants)
            
            # Deprotection decreases benzyl group count
            return product_benzyl_count < reactant_benzyl_count
            
        except:
            return False
