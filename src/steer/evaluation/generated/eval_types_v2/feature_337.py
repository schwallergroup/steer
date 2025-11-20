"""Generated evaluation code for: Benzyl protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on benzyl protecting group cycling strategy.
    Checks for the presence of both N-benzyl and O-benzyl protection followed by
    hydrogenation deprotection reactions at different stages of the synthesis.
    """
    
    def __init__(self, config):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.sites = config["parameters"]["sites"]
        self.deprotection_method = config["parameters"]["deprotection_method"]
        
        # SMARTS patterns for benzyl protecting groups
        self.n_benzyl_pattern = "[N:1][CH2:2]c1ccccc1"  # N-benzyl
        self.o_benzyl_pattern = "[O:1][CH2:2]c1ccccc1"  # O-benzyl
        
        # SMARTS patterns for deprotection products
        self.n_deprotected = "[N:1][H]"  # Free amine
        self.o_deprotected = "[O:1][H]"  # Free alcohol
        
        # Hydrogenation reaction indicators
        self.hydrogenation_reagents = ["[H][H]", "Pd", "Pt", "Ni"]

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        n_benzyl_protection = False
        o_benzyl_protection = False
        n_benzyl_deprotection = False
        o_benzyl_deprotection = False
        
        # Check each reaction for protection/deprotection events
        for rxn in reactions:
            if self.detect_n_benzyl_protection(rxn):
                n_benzyl_protection = True
            elif self.detect_o_benzyl_protection(rxn):
                o_benzyl_protection = True
            elif self.detect_n_benzyl_deprotection(rxn):
                n_benzyl_deprotection = True
            elif self.detect_o_benzyl_deprotection(rxn):
                o_benzyl_deprotection = True
        
        # Check if we have the required cycling strategy
        required_sites_protected = all(
            (site == "nitrogen" and n_benzyl_protection) or
            (site == "oxygen" and o_benzyl_protection)
            for site in self.sites
        )
        
        required_sites_deprotected = all(
            (site == "nitrogen" and n_benzyl_deprotection) or
            (site == "oxygen" and o_benzyl_deprotection)
            for site in self.sites
        )
        
        condition = required_sites_protected and required_sites_deprotected
        return condition, len(reactions)

    def detect_n_benzyl_protection(self, rxn):
        """Detect N-benzylation reactions (free amine -> N-benzyl)"""
        reactants, products = self.parse_reaction(rxn)
        
        # Check if we go from free amine to N-benzyl
        has_free_amine_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts("[N:1][H]"))
            for mol in reactants
        )
        
        has_nbenzyl_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.n_benzyl_pattern))
            for mol in products
        )
        
        return has_free_amine_reactant and has_nbenzyl_product

    def detect_o_benzyl_protection(self, rxn):
        """Detect O-benzylation reactions (free alcohol -> O-benzyl)"""
        reactants, products = self.parse_reaction(rxn)
        
        # Check if we go from free alcohol to O-benzyl
        has_free_alcohol_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts("[O:1][H]"))
            for mol in reactants
        )
        
        has_obenzyl_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.o_benzyl_pattern))
            for mol in products
        )
        
        return has_free_alcohol_reactant and has_obenzyl_product

    def detect_n_benzyl_deprotection(self, rxn):
        """Detect N-debenzylation via hydrogenation"""
        reactants, products = self.parse_reaction(rxn)
        
        # Check for N-benzyl -> free amine + hydrogenation conditions
        has_nbenzyl_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.n_benzyl_pattern))
            for mol in reactants
        )
        
        has_free_amine_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.n_deprotected))
            for mol in products
        )
        
        has_hydrogenation = self.detect_hydrogenation_conditions(rxn)
        
        return has_nbenzyl_reactant and has_free_amine_product and has_hydrogenation

    def detect_o_benzyl_deprotection(self, rxn):
        """Detect O-debenzylation via hydrogenation"""
        reactants, products = self.parse_reaction(rxn)
        
        # Check for O-benzyl -> free alcohol + hydrogenation conditions
        has_obenzyl_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.o_benzyl_pattern))
            for mol in reactants
        )
        
        has_free_alcohol_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.o_deprotected))
            for mol in products
        )
        
        has_hydrogenation = self.detect_hydrogenation_conditions(rxn)
        
        return has_obenzyl_reactant and has_free_alcohol_product and has_hydrogenation

    def detect_hydrogenation_conditions(self, rxn):
        """Check for hydrogenation reagents/conditions"""
        rxn_smiles = rxn.lower()
        return any(reagent.lower() in rxn_smiles for reagent in ["h2", "pd", "pt", "ni", "hydrogen"])

    def parse_reaction(self, rxn):
        """Parse reaction SMILES into reactant and product molecules"""
        if ">>" in rxn:
            reactant_smiles, product_smiles = rxn.split(">>")
        else:
            # Handle cases where reaction might be stored differently
            return [], []
        
        reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactant_smiles.split(".") if smi.strip()]
        products = [Chem.MolFromSmiles(smi.strip()) for smi in product_smiles.split(".") if smi.strip()]
        
        # Filter out None values from failed parsing
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        return reactants, products

    def route_scoring(self, x):
        """Score the route based on whether benzyl cycling strategy is used"""
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 10  # Strategy successfully implemented
