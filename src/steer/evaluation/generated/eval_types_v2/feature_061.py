"""Generated evaluation code for: Evans chiral auxiliary attachment and cleavage"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EvansAuxiliaryStrategy(MultiRxnCondBase):
    """
    Evaluates routes for Evans chiral auxiliary attachment and cleavage strategy.
    Checks for presence of both installation and removal of the oxazolidinone auxiliary.
    """
    
    def __init__(self, config):
        self.protecting_group_smarts = config["protecting_group_smarts"]
        self.attachment_present = config.get("attachment_present", True)
        self.cleavage_present = config.get("cleavage_present", True)
        self.auxiliary_pattern = Chem.MolFromSmarts(self.protecting_group_smarts)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        attachment_found = False
        cleavage_found = False
        
        for rxn in reactions:
            if self.detect_auxiliary_attachment(rxn):
                attachment_found = True
            elif self.detect_auxiliary_cleavage(rxn):
                cleavage_found = True
        
        # Check if required conditions are met
        attachment_condition = attachment_found == self.attachment_present
        cleavage_condition = cleavage_found == self.cleavage_present
        
        condition_met = attachment_condition and cleavage_condition
        return condition_met, len(reactions)
    
    def detect_auxiliary_attachment(self, rxn):
        """Detect Evans auxiliary attachment (auxiliary appears in product but not reactants)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if auxiliary is present in products
        auxiliary_in_products = any(
            self.has_auxiliary_substructure(prod) for prod in products
        )
        
        # Check if auxiliary is absent in reactants (or present in fewer molecules)
        reactant_auxiliary_count = sum(
            1 for react in reactants if self.has_auxiliary_substructure(react)
        )
        product_auxiliary_count = sum(
            1 for prod in products if self.has_auxiliary_substructure(prod)
        )
        
        # Attachment: auxiliary count increases from reactants to products
        return auxiliary_in_products and product_auxiliary_count > reactant_auxiliary_count
    
    def detect_auxiliary_cleavage(self, rxn):
        """Detect Evans auxiliary cleavage (auxiliary present in reactants but not products)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if auxiliary is present in reactants
        auxiliary_in_reactants = any(
            self.has_auxiliary_substructure(react) for react in reactants
        )
        
        # Check auxiliary count in reactants vs products
        reactant_auxiliary_count = sum(
            1 for react in reactants if self.has_auxiliary_substructure(react)
        )
        product_auxiliary_count = sum(
            1 for prod in products if self.has_auxiliary_substructure(prod)
        )
        
        # Cleavage: auxiliary count decreases from reactants to products
        return auxiliary_in_reactants and reactant_auxiliary_count > product_auxiliary_count
    
    def has_auxiliary_substructure(self, smiles):
        """Check if a SMILES string contains the Evans auxiliary substructure"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return mol.HasSubstructMatch(self.auxiliary_pattern)
        except:
            return False
