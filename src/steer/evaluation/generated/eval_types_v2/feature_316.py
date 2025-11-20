"""Generated evaluation code for: Boc protecting group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for Boc protecting group cycling - consecutive protection 
    followed immediately by deprotection of the same or similar functional groups.
    """
    
    def __init__(self, config):
        self.consecutive = config.get("consecutive", True)
        self.boc_protection_pattern = Chem.MolFromSmarts("[NH2,NH1][C](=O)[O][C](C)(C)C")
        self.boc_deprotection_pattern = Chem.MolFromSmarts("[NH1,NH2]")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        boc_cycle_found = False
        
        if self.consecutive:
            # Check for consecutive Boc protection/deprotection
            for i in range(len(reactions) - 1):
                current_rxn = reactions[i]
                next_rxn = reactions[i + 1]
                
                if self.is_boc_protection(current_rxn) and self.is_boc_deprotection(next_rxn):
                    # Check if they involve the same nitrogen atom by atom mapping
                    if self.same_nitrogen_involved(current_rxn, next_rxn):
                        boc_cycle_found = True
                        break
        else:
            # Check for any Boc protection followed by deprotection in the route
            protection_found = any(self.is_boc_protection(rxn) for rxn in reactions)
            deprotection_found = any(self.is_boc_deprotection(rxn) for rxn in reactions)
            boc_cycle_found = protection_found and deprotection_found
        
        return boc_cycle_found, len(reactions)
    
    def is_boc_protection(self, rxn):
        """Check if reaction introduces a Boc protecting group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Check if Boc group appears in products but not in reactants
            reactant_has_boc = any(mol and mol.HasSubstructMatch(self.boc_protection_pattern) 
                                 for mol in reactant_mols if mol)
            product_has_boc = any(mol and mol.HasSubstructMatch(self.boc_protection_pattern) 
                                for mol in product_mols if mol)
            
            # Also check for common Boc reagents in reactants
            boc_reagent_patterns = [
                Chem.MolFromSmarts("[C](=O)[O][C](C)(C)C"),  # Boc anhydride
                Chem.MolFromSmarts("O[C](=O)[O][C](C)(C)C")   # Boc carbonate
            ]
            
            has_boc_reagent = any(
                any(mol and mol.HasSubstructMatch(pattern) for mol in reactant_mols if mol)
                for pattern in boc_reagent_patterns
            )
            
            return (not reactant_has_boc) and product_has_boc and has_boc_reagent
            
        except:
            return False
    
    def is_boc_deprotection(self, rxn):
        """Check if reaction removes a Boc protecting group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Check if Boc group disappears from reactants to products
            reactant_has_boc = any(mol and mol.HasSubstructMatch(self.boc_protection_pattern) 
                                 for mol in reactant_mols if mol)
            product_has_boc = any(mol and mol.HasSubstructMatch(self.boc_protection_pattern) 
                                for mol in product_mols if mol)
            
            return reactant_has_boc and not product_has_boc
            
        except:
            return False
    
    def same_nitrogen_involved(self, protection_rxn, deprotection_rxn):
        """Check if the same nitrogen atom is involved in both reactions using atom mapping"""
        try:
            # Get the protected intermediate (product of protection, reactant of deprotection)
            protection_products = protection_rxn.split(">>")[1]
            deprotection_reactants = deprotection_rxn.split(">>")[0]
            
            # Find the main organic molecule (usually the largest)
            protection_mols = [Chem.MolFromSmiles(smi.strip()) for smi in protection_products.split(".")]
            deprotection_mols = [Chem.MolFromSmiles(smi.strip()) for smi in deprotection_reactants.split(".")]
            
            # Get the molecule with Boc protection from both reactions
            protection_boc_mol = None
            deprotection_boc_mol = None
            
            for mol in protection_mols:
                if mol and mol.HasSubstructMatch(self.boc_protection_pattern):
                    protection_boc_mol = mol
                    break
                    
            for mol in deprotection_mols:
                if mol and mol.HasSubstructMatch(self.boc_protection_pattern):
                    deprotection_boc_mol = mol
                    break
            
            if not protection_boc_mol or not deprotection_boc_mol:
                return False
            
            # Check if they have the same SMILES (simplified check)
            return Chem.MolToSmiles(protection_boc_mol) == Chem.MolToSmiles(deprotection_boc_mol)
            
        except:
            return True  # Conservative assumption if parsing fails
