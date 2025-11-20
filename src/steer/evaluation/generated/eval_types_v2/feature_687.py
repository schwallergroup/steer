"""Generated evaluation code for: Sequential protection deprotection alcohol strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectionDeprotection(MultiRxnCondBase):
    """
    Evaluates synthesis routes for sequential protection-deprotection strategy.
    Checks if alcohols are protected with acetyl groups early in the route,
    then deprotected later, ensuring temporary protection for chemoselectivity.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "acetyl")
        self.functional_group = config.get("functional_group", "alcohol")
        self.strategy = config.get("strategy", "temporary")
        
        # Define SMARTS patterns for detection
        self.alcohol_pattern = "[OH1]"  # Primary/secondary alcohol
        self.acetyl_ester_pattern = "[CH3]C(=O)O[CH,CH2]"  # Acetyl ester
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_found = False
        deprotection_found = False
        protection_depth = -1
        deprotection_depth = -1
        
        # Check reactions in order (earliest to latest)
        for i, rxn in enumerate(reactions):
            if self.detect_protection(rxn):
                protection_found = True
                if protection_depth == -1:
                    protection_depth = i
                    
            if self.detect_deprotection(rxn):
                deprotection_found = True
                if deprotection_depth == -1:
                    deprotection_depth = i
        
        # For temporary strategy, need both protection and deprotection
        # Protection should occur before deprotection
        if self.strategy == "temporary":
            condition = (protection_found and deprotection_found and 
                        protection_depth < deprotection_depth)
        else:
            condition = protection_found
            
        return condition, len(reactions)
    
    def detect_protection(self, rxn):
        """Detect acetylation of alcohol (protection reaction)"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Parse molecules
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check if alcohol is consumed and acetyl ester is formed
            alcohol_in_reactants = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.alcohol_pattern)) 
                                     for mol in reactant_mols if mol)
            acetyl_in_products = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.acetyl_ester_pattern)) 
                                   for mol in product_mols if mol)
            
            # Also check for acetylating reagents (acetyl chloride, acetic anhydride)
            acetylating_reagents = ["CC(=O)Cl", "CC(=O)OC(=O)C"]  # acetyl chloride, acetic anhydride
            reagent_present = any(any(mol.HasSubstructMatch(Chem.MolFromSmiles(reagent)) 
                                    for reagent in acetylating_reagents) 
                                for mol in reactant_mols if mol)
            
            return alcohol_in_reactants and acetyl_in_products and reagent_present
            
        except Exception:
            return False
    
    def detect_deprotection(self, rxn):
        """Detect deacetylation (deprotection reaction)"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Parse molecules
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check if acetyl ester is consumed and alcohol is formed
            acetyl_in_reactants = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.acetyl_ester_pattern)) 
                                    for mol in reactant_mols if mol)
            alcohol_in_products = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.alcohol_pattern)) 
                                    for mol in product_mols if mol)
            
            # Check for deprotection conditions (base or acid hydrolysis)
            deprotection_reagents = ["[OH-]", "[Li+].[OH-]", "[Na+].[OH-]", "[K+].[OH-]"]  # bases for hydrolysis
            reagent_present = any(any(mol.HasSubstructMatch(Chem.MolFromSmarts(reagent)) 
                                    for reagent in deprotection_reagents) 
                                for mol in reactant_mols if mol)
            
            return acetyl_in_reactants and alcohol_in_products and reagent_present
            
        except Exception:
            return False
