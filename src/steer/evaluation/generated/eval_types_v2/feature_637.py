"""Generated evaluation code for: Sequential electrophile preparation then alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialElectrophileAlkylation(MultiRxnCondBase):
    """
    Checks for sequential electrophile preparation followed by alkylation.
    First detects alcohol to halide conversion, then alkylation forming C-C bond.
    """
    
    def __init__(self, config):
        self.reaction_sequence = config.get("reaction_sequence", ["alcohol_to_halide", "alkylation"])
        self.bond_formed = config.get("bond_formed", "C-C")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find alcohol to halide conversions
        alcohol_to_halide_steps = []
        for i, rxn in enumerate(reactions):
            if self.detect_alcohol_to_halide(rxn):
                alcohol_to_halide_steps.append(i)
        
        # Find alkylation reactions forming C-C bonds
        alkylation_steps = []
        for i, rxn in enumerate(reactions):
            if self.detect_cc_alkylation(rxn):
                alkylation_steps.append(i)
        
        # Check if we have both reaction types
        has_alcohol_to_halide = len(alcohol_to_halide_steps) > 0
        has_alkylation = len(alkylation_steps) > 0
        
        # For sequential requirement, alkylation should occur after electrophile prep
        sequential = False
        if has_alcohol_to_halide and has_alkylation:
            # Check if any alkylation step comes after any alcohol->halide step
            for alc_step in alcohol_to_halide_steps:
                for alk_step in alkylation_steps:
                    if alk_step > alc_step:  # alkylation after electrophile prep
                        sequential = True
                        break
                if sequential:
                    break
        
        condition = has_alcohol_to_halide and has_alkylation and sequential
        return condition, len(reactions)
    
    def detect_alcohol_to_halide(self, rxn):
        """Detect conversion of alcohol (-OH) to halide (-X)"""
        try:
            reactants, products = rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Look for alcohol pattern in reactants
            alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
            has_alcohol = any(mol and mol.HasSubstructMatch(alcohol_pattern) for mol in reactant_mols)
            
            # Look for halide pattern in products
            halide_patterns = [
                Chem.MolFromSmarts("[C][Cl]"),
                Chem.MolFromSmarts("[C][Br]"),
                Chem.MolFromSmarts("[C][I]")
            ]
            has_halide = any(
                mol and any(mol.HasSubstructMatch(pattern) for pattern in halide_patterns)
                for mol in product_mols
            )
            
            return has_alcohol and has_halide
            
        except:
            return False
    
    def detect_cc_alkylation(self, rxn):
        """Detect alkylation reaction forming C-C bond"""
        try:
            reactants, products = rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Count C-C bonds in reactants vs products
            reactant_cc_bonds = sum(self.count_cc_bonds(mol) for mol in reactant_mols if mol)
            product_cc_bonds = sum(self.count_cc_bonds(mol) for mol in product_mols if mol)
            
            # Check for presence of electrophilic carbon (halide or similar leaving group)
            electrophile_patterns = [
                Chem.MolFromSmarts("[C][Cl,Br,I]"),
                Chem.MolFromSmarts("[C][O][S](=O)(=O)"),  # tosylate, mesylate
            ]
            has_electrophile = any(
                mol and any(mol.HasSubstructMatch(pattern) for pattern in electrophile_patterns)
                for mol in reactant_mols
            )
            
            # Alkylation: C-C bond formation with electrophilic carbon
            return (product_cc_bonds > reactant_cc_bonds) and has_electrophile
            
        except:
            return False
    
    def count_cc_bonds(self, mol):
        """Count C-C bonds in a molecule"""
        if not mol:
            return 0
        count = 0
        for bond in mol.GetBonds():
            if (bond.GetBeginAtom().GetSymbol() == 'C' and 
                bond.GetEndAtom().GetSymbol() == 'C'):
                count += 1
        return count
