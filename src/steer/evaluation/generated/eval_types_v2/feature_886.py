"""Generated evaluation code for: Cbz protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzProtectingGroupStrategy(BaseScoring):
    """
    Evaluates the use of Cbz (carboxybenzyl) protecting group strategy for amines.
    Checks if Cbz protection/deprotection reactions occur in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0  # Strategy not used
            return max(0, 1 - abs(x - self.target_depth))  # Closer to target is better
    
    def hit_condition(self, d):
        """Check if this reaction involves Cbz protection/deprotection"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        # Check for Cbz protection (formation of carbamate bond)
        if self._is_cbz_protection(mapped_rxn):
            return True
            
        # Check for Cbz deprotection (cleavage of carbamate bond)
        if self._is_cbz_deprotection(mapped_rxn):
            return True
            
        return False
    
    def _is_cbz_protection(self, mapped_rxn):
        """Check if reaction involves Cbz protection of amine"""
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Look for Cbz chloride or similar reagent in reactants
            cbz_reagent_pattern = Chem.MolFromSmarts("ClC(=O)OCc1ccccc1")  # Cbz-Cl
            cbz_reagent_alt = Chem.MolFromSmarts("O=C(OCc1ccccc1)OCc2ccccc2")  # Cbz2O
            
            has_cbz_reagent = False
            for reactant_smi in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smi)
                if reactant_mol:
                    if (cbz_reagent_pattern and reactant_mol.HasSubstructMatch(cbz_reagent_pattern)) or \
                       (cbz_reagent_alt and reactant_mol.HasSubstructMatch(cbz_reagent_alt)):
                        has_cbz_reagent = True
                        break
            
            # Look for Cbz-protected amine in products
            cbz_protected_pattern = Chem.MolFromSmarts("NC(=O)OCc1ccccc1")  # Cbz-NH
            has_cbz_product = False
            for product_smi in products:
                product_mol = Chem.MolFromSmiles(product_smi)
                if product_mol and cbz_protected_pattern:
                    if product_mol.HasSubstructMatch(cbz_protected_pattern):
                        has_cbz_product = True
                        break
            
            return has_cbz_reagent and has_cbz_product
            
        except:
            return False
    
    def _is_cbz_deprotection(self, mapped_rxn):
        """Check if reaction involves Cbz deprotection to free amine"""
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Look for Cbz-protected amine in reactants
            cbz_protected_pattern = Chem.MolFromSmarts("NC(=O)OCc1ccccc1")
            has_cbz_reactant = False
            for reactant_smi in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smi)
                if reactant_mol and cbz_protected_pattern:
                    if reactant_mol.HasSubstructMatch(cbz_protected_pattern):
                        has_cbz_reactant = True
                        break
            
            # Look for free amine in products and benzyl alcohol/toluene byproduct
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")  # Primary or secondary amine
            benzyl_byproduct_pattern = Chem.MolFromSmarts("OCc1ccccc1")  # Benzyl alcohol
            
            has_free_amine = False
            has_benzyl_byproduct = False
            
            for product_smi in products:
                product_mol = Chem.MolFromSmiles(product_smi)
                if product_mol:
                    if free_amine_pattern and product_mol.HasSubstructMatch(free_amine_pattern):
                        has_free_amine = True
                    if benzyl_byproduct_pattern and product_mol.HasSubstructMatch(benzyl_byproduct_pattern):
                        has_benzyl_byproduct = True
            
            return has_cbz_reactant and has_free_amine
            
        except:
            return False
