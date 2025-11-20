"""Generated evaluation code for: Orthogonal trityl and acetate protecting groups"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroups(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the use of orthogonal trityl and acetate protecting groups.
    Checks for trityl protection of primary alcohols and acetate protection of secondary alcohols
    with selective deprotection conditions.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["trityl", "acetate"])
        self.selectivity = config.get("selectivity", "orthogonal")
        self.alcohol_types = config.get("alcohol_types", ["primary", "secondary"])
        
        # SMARTS patterns for detecting protecting group operations
        self.trityl_protection_pattern = "[OH1][CH2:1]>>[O:1]C(c1ccccc1)(c2ccccc2)c3ccccc3"
        self.acetate_protection_pattern = "[OH1][CH1:1]>>[O:1]C(=O)C"
        self.trityl_deprotection_pattern = "[O:1]C(c1ccccc1)(c2ccccc2)c3ccccc3>>[OH1:1]"
        self.acetate_deprotection_pattern = "[O:1]C(=O)C>>[OH1:1]"
        
        # Substructure patterns
        self.trityl_ether = Chem.MolFromSmarts("[O]C(c1ccccc1)(c2ccccc2)c3ccccc3")
        self.acetate_ester = Chem.MolFromSmarts("[O]C(=O)C")
        self.primary_alcohol = Chem.MolFromSmarts("[OH1][CH2]")
        self.secondary_alcohol = Chem.MolFromSmarts("[OH1][CH1]")

    def condition_depth(self, d):
        reactions = self.get_rxns(d)
        
        trityl_protection_found = False
        acetate_protection_found = False
        trityl_deprotection_found = False
        acetate_deprotection_found = False
        
        for rxn in reactions:
            if self.detect_trityl_protection(rxn):
                trityl_protection_found = True
            if self.detect_acetate_protection(rxn):
                acetate_protection_found = True
            if self.detect_trityl_deprotection(rxn):
                trityl_deprotection_found = True
            if self.detect_acetate_deprotection(rxn):
                acetate_deprotection_found = True
        
        # Check for orthogonal usage - both protecting groups used and selectively removed
        orthogonal_condition = (
            trityl_protection_found and 
            acetate_protection_found and
            (trityl_deprotection_found or acetate_deprotection_found)
        )
        
        return orthogonal_condition, len(reactions)

    def detect_trityl_protection(self, rxn):
        """Detect trityl protection of primary alcohols"""
        reactants, products = self.parse_reaction(rxn)
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(self.primary_alcohol):
                for product in products:
                    if product.HasSubstructMatch(self.trityl_ether):
                        return True
        return False

    def detect_acetate_protection(self, rxn):
        """Detect acetate protection of secondary alcohols"""
        reactants, products = self.parse_reaction(rxn)
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(self.secondary_alcohol):
                for product in products:
                    if product.HasSubstructMatch(self.acetate_ester):
                        return True
        return False

    def detect_trityl_deprotection(self, rxn):
        """Detect selective trityl deprotection"""
        reactants, products = self.parse_reaction(rxn)
        
        trityl_in_reactant = any(r.HasSubstructMatch(self.trityl_ether) for r in reactants)
        alcohol_in_product = any(p.HasSubstructMatch(self.primary_alcohol) for p in products)
        acetate_preserved = any(r.HasSubstructMatch(self.acetate_ester) for r in reactants) and \
                           any(p.HasSubstructMatch(self.acetate_ester) for p in products)
        
        return trityl_in_reactant and alcohol_in_product and acetate_preserved

    def detect_acetate_deprotection(self, rxn):
        """Detect selective acetate deprotection"""
        reactants, products = self.parse_reaction(rxn)
        
        acetate_in_reactant = any(r.HasSubstructMatch(self.acetate_ester) for r in reactants)
        alcohol_in_product = any(p.HasSubstructMatch(self.secondary_alcohol) for p in products)
        trityl_preserved = any(r.HasSubstructMatch(self.trityl_ether) for r in reactants) and \
                          any(p.HasSubstructMatch(self.trityl_ether) for p in products)
        
        return acetate_in_reactant and alcohol_in_product and trityl_preserved

    def parse_reaction(self, rxn_smiles):
        """Parse reaction SMILES into reactant and product molecules"""
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        reactants = []
        if reactants_smiles:
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactants.append(mol)
        
        products = []
        if products_smiles:
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    products.append(mol)
        
        return reactants, products
