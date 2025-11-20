"""Generated evaluation code for: TBS protecting group strategy for primary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBSProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates TBS protecting group strategy for primary alcohols.
    Checks for TBS installation early in the route and removal at specified step.
    """
    
    def __init__(self, config):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.installation_step = config["parameters"]["installation_step"]
        self.removal_step = config["parameters"]["removal_step"]
        
        # SMARTS patterns for TBS group and primary alcohol
        self.tbs_pattern = "[Si](C)(C)C(C)(C)C"  # TBS group
        self.primary_alcohol_pattern = "[CH2][OH]"  # Primary alcohol
        self.tbs_protected_alcohol_pattern = "[CH2]O[Si](C)(C)C(C)(C)C"  # TBS-protected primary alcohol
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_steps = len(reactions)
        
        # Find TBS installation and removal steps
        installation_found = False
        removal_found = False
        installation_depth = -1
        removal_depth = -1
        
        for i, rxn in enumerate(reactions):
            step_number = i + 1
            
            # Check for TBS installation (primary alcohol -> TBS-protected alcohol)
            if self.detect_tbs_installation(rxn):
                installation_found = True
                installation_depth = step_number
            
            # Check for TBS removal (TBS-protected alcohol -> primary alcohol or other)
            if self.detect_tbs_removal(rxn):
                removal_found = True
                removal_depth = step_number
        
        # Evaluate conditions
        early_installation = False
        correct_removal = False
        
        if installation_found:
            if self.installation_step == "early":
                # Consider "early" as first 25% of steps
                early_installation = installation_depth <= max(1, total_steps * 0.25)
            else:
                early_installation = installation_depth == int(self.installation_step)
        
        if removal_found:
            correct_removal = removal_depth == self.removal_step
        
        # Strategy is successful if both installation and removal occur correctly
        condition = installation_found and removal_found and early_installation and correct_removal
        
        return condition, total_steps
    
    def detect_tbs_installation(self, rxn):
        """Detect TBS protection of primary alcohol"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            # Check if reactants contain primary alcohol
            has_primary_alcohol = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.primary_alcohol_pattern))
                for mol in reactant_mols
            )
            
            # Check if products contain TBS-protected alcohol
            has_tbs_protected = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbs_protected_alcohol_pattern))
                for mol in product_mols
            )
            
            # Check if TBS reagent is present in reactants
            has_tbs_reagent = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbs_pattern))
                for mol in reactant_mols
            )
            
            return has_primary_alcohol and has_tbs_protected and has_tbs_reagent
            
        except Exception:
            return False
    
    def detect_tbs_removal(self, rxn):
        """Detect TBS deprotection"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            # Check if reactants contain TBS-protected alcohol
            has_tbs_protected = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbs_protected_alcohol_pattern))
                for mol in reactant_mols
            )
            
            # Check if TBS group is removed (no longer present in main product)
            tbs_removed = True
            for mol in product_mols:
                # Skip small molecules that might be byproducts
                if mol.GetNumAtoms() > 5:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbs_protected_alcohol_pattern)):
                        tbs_removed = False
                        break
            
            return has_tbs_protected and tbs_removed
            
        except Exception:
            return False
