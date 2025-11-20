"""Generated evaluation code for: Multi-step alcohol to nitrile conversion sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholToNitrileSequence(MultiRxnCondBase):
    """
    Checks for a 4-step alcohol to nitrile conversion sequence:
    protected_alcohol -> deprotection -> oxidation -> oxime_formation -> dehydration -> nitrile
    """
    
    def __init__(self, config):
        self.reaction_sequence = config["reaction_sequence"]
        self.starting_fg = config["starting_fg"]
        self.ending_fg = config["ending_fg"]
        
        # SMARTS patterns for functional groups
        self.protected_alcohol_pattern = "[CH2][OH1]"  # Simple alcohol pattern
        self.nitrile_pattern = "[C]#[N]"
        
        # Reaction type patterns
        self.deprotection_patterns = ["[OH1]", "[CH2][OH1]"]
        self.oxidation_patterns = ["[CH1]=O", "[CH2]=O"]  # Aldehyde/ketone formation
        self.oxime_patterns = ["[CH1]=[N][OH1]", "[CH2]=[N][OH1]"]  # Oxime formation
        self.dehydration_patterns = ["[C]#[N]"]  # Final nitrile

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check if we can find the complete sequence
        sequence_found = self.detect_sequence(reactions)
        
        return sequence_found, len(reactions)

    def detect_sequence(self, reactions) -> bool:
        """Detect if the 4-step sequence is present in the reactions"""
        if len(reactions) < 4:
            return False
            
        # Track functional group transformations through the sequence
        sequence_steps = {
            "deprotection": False,
            "oxidation": False, 
            "oxime_formation": False,
            "dehydration": False
        }
        
        for rxn in reactions:
            rxn_smiles = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                continue
                
            if self.detect_deprotection(rxn_smiles):
                sequence_steps["deprotection"] = True
            elif self.detect_oxidation(rxn_smiles):
                sequence_steps["oxidation"] = True
            elif self.detect_oxime_formation(rxn_smiles):
                sequence_steps["oxime_formation"] = True
            elif self.detect_dehydration(rxn_smiles):
                sequence_steps["dehydration"] = True
        
        # All steps must be present
        return all(sequence_steps.values())

    def detect_deprotection(self, rxn_smiles) -> bool:
        """Detect deprotection reactions leading to free alcohol"""
        if ">>" not in rxn_smiles:
            return False
            
        reactants, products = rxn_smiles.split(">>")
        
        try:
            prod_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            prod_mols = [m for m in prod_mols if m is not None]
            
            # Look for alcohol formation in products
            alcohol_pattern = Chem.MolFromSmarts("[CH2,CH1][OH1]")
            return any(mol.HasSubstructMatch(alcohol_pattern) for mol in prod_mols)
        except:
            return False

    def detect_oxidation(self, rxn_smiles) -> bool:
        """Detect oxidation of alcohol to aldehyde/ketone"""
        if ">>" not in rxn_smiles:
            return False
            
        reactants, products = rxn_smiles.split(">>")
        
        try:
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            prod_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            react_mols = [m for m in react_mols if m is not None]
            prod_mols = [m for m in prod_mols if m is not None]
            
            # Check for alcohol in reactants and carbonyl in products
            alcohol_pattern = Chem.MolFromSmarts("[CH2,CH1][OH1]")
            carbonyl_pattern = Chem.MolFromSmarts("[CH1,CH2]=O")
            
            has_alcohol_reactant = any(mol.HasSubstructMatch(alcohol_pattern) for mol in react_mols)
            has_carbonyl_product = any(mol.HasSubstructMatch(carbonyl_pattern) for mol in prod_mols)
            
            return has_alcohol_reactant and has_carbonyl_product
        except:
            return False

    def detect_oxime_formation(self, rxn_smiles) -> bool:
        """Detect oxime formation from carbonyl"""
        if ">>" not in rxn_smiles:
            return False
            
        reactants, products = rxn_smiles.split(">>")
        
        try:
            prod_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            prod_mols = [m for m in prod_mols if m is not None]
            
            # Look for oxime pattern in products
            oxime_pattern = Chem.MolFromSmarts("[CH1,CH2]=[N][OH1]")
            return any(mol.HasSubstructMatch(oxime_pattern) for mol in prod_mols)
        except:
            return False

    def detect_dehydration(self, rxn_smiles) -> bool:
        """Detect dehydration of oxime to nitrile"""
        if ">>" not in rxn_smiles:
            return False
            
        reactants, products = rxn_smiles.split(">>")
        
        try:
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            prod_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            react_mols = [m for m in react_mols if m is not None]
            prod_mols = [m for m in prod_mols if m is not None]
            
            # Check for oxime in reactants and nitrile in products
            oxime_pattern = Chem.MolFromSmarts("[CH1,CH2]=[N][OH1]")
            nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
            
            has_oxime_reactant = any(mol.HasSubstructMatch(oxime_pattern) for mol in react_mols)
            has_nitrile_product = any(mol.HasSubstructMatch(nitrile_pattern) for mol in prod_mols)
            
            return has_oxime_reactant and has_nitrile_product
        except:
            return False
