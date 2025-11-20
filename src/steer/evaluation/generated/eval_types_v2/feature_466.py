"""Generated evaluation code for: Protecting group swap from Cbz to Boc"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates synthesis routes for unnecessary protecting group swaps.
    Detects when a Cbz protecting group is removed and then a Boc group
    is added to the same amine functionality, indicating inefficient strategy.
    """
    
    def __init__(self, config):
        self.initial_group = config.get("initial_group", "Cbz")
        self.final_group = config.get("final_group", "Boc")
        self.functional_group = config.get("functional_group", "amine")
        self.swap_occurs = config.get("swap_occurs", True)
        
        # SMARTS patterns for protected amines
        self.cbz_pattern = "NC(=O)OCc1ccccc1"  # Cbz-protected amine
        self.boc_pattern = "NC(=O)OC(C)(C)C"   # Boc-protected amine
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group changes throughout the route
        cbz_removed = False
        boc_added = False
        cbz_removal_depth = -1
        boc_addition_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_cbz_removal(rxn):
                cbz_removed = True
                cbz_removal_depth = i
                
            if self.detect_boc_addition(rxn):
                boc_added = True
                boc_addition_depth = i
        
        # Check if swap occurs (Cbz removed then Boc added)
        swap_detected = (cbz_removed and boc_added and 
                        cbz_removal_depth < boc_addition_depth)
        
        condition_met = swap_detected == self.swap_occurs
        
        # Return depth as fraction of total reactions where swap completes
        if swap_detected:
            depth_fraction = max(cbz_removal_depth, boc_addition_depth) / len(reactions)
        else:
            depth_fraction = -1  # No swap detected
            
        return condition_met, depth_fraction
    
    def detect_cbz_removal(self, rxn):
        """Detect removal of Cbz protecting group from amine"""
        reactants, products = self.parse_reaction_smiles(rxn)
        
        # Check if Cbz-protected amine in reactants but not in products
        cbz_mol = Chem.MolFromSmarts(self.cbz_pattern)
        
        cbz_in_reactants = any(mol.HasSubstructMatch(cbz_mol) for mol in reactants)
        cbz_in_products = any(mol.HasSubstructMatch(cbz_mol) for mol in products)
        
        return cbz_in_reactants and not cbz_in_products
    
    def detect_boc_addition(self, rxn):
        """Detect addition of Boc protecting group to amine"""
        reactants, products = self.parse_reaction_smiles(rxn)
        
        # Check if Boc-protected amine in products but not in reactants
        boc_mol = Chem.MolFromSmarts(self.boc_pattern)
        
        boc_in_reactants = any(mol.HasSubstructMatch(boc_mol) for mol in reactants)
        boc_in_products = any(mol.HasSubstructMatch(boc_mol) for mol in products)
        
        return not boc_in_reactants and boc_in_products
    
    def parse_reaction_smiles(self, rxn):
        """Parse reaction SMILES into reactant and product molecules"""
        rxn_parts = rxn.split(">>")
        reactant_smiles = rxn_parts[1].split(".")  # Reactants are after >>
        product_smiles = rxn_parts[0].split(".")   # Products are before >>
        
        reactants = [Chem.MolFromSmiles(smi) for smi in reactant_smiles if smi]
        products = [Chem.MolFromSmiles(smi) for smi in product_smiles if smi]
        
        # Filter out None molecules
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        return reactants, products
