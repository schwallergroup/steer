"""Generated evaluation code for: Boc protecting group strategy for piperazine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocPiperazineStrategy(MultiRxnCondBase):
    """
    Evaluates Boc protecting group strategy for piperazine.
    Checks for presence of Boc-protected piperazine and appropriate deprotection timing.
    """
    
    def __init__(self, config):
        self.deprotection_stage = config["parameters"].get("deprotection_stage", "final")
        self.boc_pattern = Chem.MolFromSmarts("C(=O)OC(C)(C)C")  # Boc group
        self.piperazine_pattern = Chem.MolFromSmarts("C1CNCCN1")  # Piperazine ring
        self.boc_piperazine_pattern = Chem.MolFromSmarts("C1CN(C(=O)OC(C)(C)C)CCN1")  # Boc-protected piperazine
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check if Boc-protected piperazine is used
        has_boc_protection = any(self.detect_boc_piperazine_formation(r) for r in reactions)
        
        # Check deprotection timing
        deprotection_correct = self.check_deprotection_timing(reactions)
        
        # Check if strategy is properly implemented
        condition = has_boc_protection and deprotection_correct
        
        return condition, len(reactions)
    
    def detect_boc_piperazine_formation(self, rxn):
        """Detect formation of Boc-protected piperazine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if reactants contain piperazine and Boc reagent
        has_piperazine_reactant = False
        has_boc_reagent = False
        
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(self.piperazine_pattern):
                    has_piperazine_reactant = True
                if mol and (mol.HasSubstructMatch(self.boc_pattern) or 
                           "Boc2O" in reactant_smiles or "BOC" in reactant_smiles):
                    has_boc_reagent = True
            except:
                continue
                
        # Check if product contains Boc-protected piperazine
        has_boc_piperazine_product = False
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(self.boc_piperazine_pattern):
                    has_boc_piperazine_product = True
            except:
                continue
                
        return has_piperazine_reactant and has_boc_reagent and has_boc_piperazine_product
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if reactant has Boc-protected piperazine
        has_boc_piperazine_reactant = False
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(self.boc_piperazine_pattern):
                    has_boc_piperazine_reactant = True
            except:
                continue
                
        # Check if product has free piperazine
        has_free_piperazine_product = False
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(self.piperazine_pattern):
                    # Make sure it's not still Boc-protected
                    if not mol.HasSubstructMatch(self.boc_piperazine_pattern):
                        has_free_piperazine_product = True
            except:
                continue
                
        return has_boc_piperazine_reactant and has_free_piperazine_product
    
    def check_deprotection_timing(self, reactions):
        """Check if deprotection occurs at the appropriate stage"""
        if self.deprotection_stage == "final":
            # Deprotection should occur in the last few steps
            total_reactions = len(reactions)
            deprotection_indices = []
            
            for i, rxn in enumerate(reactions):
                if self.detect_boc_deprotection(rxn):
                    deprotection_indices.append(i)
            
            if not deprotection_indices:
                return False
                
            # Check if deprotection occurs in final 20% of reactions
            final_stage_threshold = max(1, int(0.8 * total_reactions))
            return any(idx >= final_stage_threshold for idx in deprotection_indices)
        
        return True  # For other deprotection stages, accept any timing
