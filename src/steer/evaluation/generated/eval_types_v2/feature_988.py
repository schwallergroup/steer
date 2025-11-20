"""Generated evaluation code for: Benzyl protecting group with allyl ether incompatibility"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylAllylIncompatibility(MultiRxnCondBase):
    """
    Checks for incompatible use of benzyl and allyl protecting groups.
    Penalizes routes that use both benzyl ethers (requiring hydrogenolysis for deprotection)
    and allyl ethers, since hydrogenolysis would reduce both groups simultaneously.
    """
    
    def __init__(self, config):
        self.check_incompatibility = config.get("incompatibility", True)
        self.deprotection_method = config.get("deprotection_method", "hydrogenolysis")
        
        # SMARTS patterns for protecting group detection
        self.benzyl_ether_pattern = "[#6]-[#8]-[CH2]-c1ccccc1"  # R-O-CH2-Ph
        self.allyl_ether_pattern = "[#6]-[#8]-[CH2]-[CH]=[CH2]"  # R-O-CH2-CH=CH2
        self.benzyl_formation_pattern = "[OH,SH]-[CH2]-c1ccccc1>>*"  # Benzyl protection
        self.allyl_formation_pattern = "[OH,SH]-[CH2]-[CH]=[CH2]>>*"  # Allyl protection

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_benzyl_protection = any(self.detect_benzyl_protection(r) for r in reactions)
        has_allyl_protection = any(self.detect_allyl_protection(r) for r in reactions)
        has_benzyl_present = any(self.has_benzyl_ether(r) for r in reactions)
        has_allyl_present = any(self.has_allyl_ether(r) for r in reactions)
        
        # Check for incompatibility: both protecting groups used in the route
        incompatible_groups = (has_benzyl_protection or has_benzyl_present) and \
                             (has_allyl_protection or has_allyl_present)
        
        if self.check_incompatibility:
            # Return True if NO incompatibility (good condition)
            condition = not incompatible_groups
        else:
            # Return True if incompatibility is present
            condition = incompatible_groups
            
        return condition, len(reactions)

    def detect_benzyl_protection(self, rxn):
        """Detect formation of benzyl ether protecting groups"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if benzyl ether is formed (present in product but not reactant)
        return self.protecting_group_formed(reactants, products, self.benzyl_ether_pattern)

    def detect_allyl_protection(self, rxn):
        """Detect formation of allyl ether protecting groups"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        return self.protecting_group_formed(reactants, products, self.allyl_ether_pattern)

    def has_benzyl_ether(self, rxn):
        """Check if benzyl ether is present in any molecule in the reaction"""
        return self.has_substructure_in_reaction(rxn, self.benzyl_ether_pattern)

    def has_allyl_ether(self, rxn):
        """Check if allyl ether is present in any molecule in the reaction"""
        return self.has_substructure_in_reaction(rxn, self.allyl_ether_pattern)

    def protecting_group_formed(self, reactants, products, pattern):
        """Check if a protecting group pattern is formed in the reaction"""
        try:
            # Parse reactants
            reactant_mols = []
            for smi in reactants.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            # Parse products  
            product_mols = []
            for smi in products.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            pattern_mol = Chem.MolFromSmarts(pattern)
            if not pattern_mol:
                return False
            
            # Check if pattern is in products but not in reactants
            in_products = any(mol.HasSubstructMatch(pattern_mol) for mol in product_mols)
            in_reactants = any(mol.HasSubstructMatch(pattern_mol) for mol in reactant_mols)
            
            return in_products and not in_reactants
            
        except:
            return False

    def has_substructure_in_reaction(self, rxn, pattern):
        """Check if substructure pattern exists anywhere in the reaction"""
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if not pattern_mol:
                return False
                
            all_smiles = rxn.replace(">>", ".").split(".")
            
            for smi in all_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol and mol.HasSubstructMatch(pattern_mol):
                    return True
                    
            return False
        except:
            return False

    def route_scoring(self, x):
        """Convert condition result to score"""
        if x < 0:
            return 10  # Maximum penalty for incompatible protecting groups
        else:
            return 0   # No penalty when condition is satisfied
