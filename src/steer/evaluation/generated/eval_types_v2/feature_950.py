"""Generated evaluation code for: Convergent synthesis via amide and C-N coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCNCoupling(MultiRxnCondBase):
    """
    Evaluates convergent synthesis routes that use both amide coupling and C-N coupling reactions.
    Checks for the presence of amide bond formation and aryl amine coupling (e.g., Buchwald-Hartwig)
    reactions in the synthesis tree.
    """
    
    def __init__(self, config):
        self.coupling_points = config.get("coupling_points", ["amide", "aryl_amine"])
        self.fragment_count = config.get("fragment_count", 2)
        self.require_amide = "amide" in self.coupling_points
        self.require_aryl_amine = "aryl_amine" in self.coupling_points
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_amide_coupling = False
        has_aryl_amine_coupling = False
        
        for rxn in reactions:
            if self.require_amide and self.detect_amide_coupling(rxn):
                has_amide_coupling = True
            if self.require_aryl_amine and self.detect_aryl_amine_coupling(rxn):
                has_aryl_amine_coupling = True
        
        # Check if all required coupling types are present
        condition_met = True
        if self.require_amide and not has_amide_coupling:
            condition_met = False
        if self.require_aryl_amine and not has_aryl_amine_coupling:
            condition_met = False
            
        return condition_met, len(reactions)
    
    def detect_amide_coupling(self, rxn):
        """Detects amide bond formation reactions"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[1]
        reactants = rxn_parts[0]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check for amide bond formation (C(=O)N pattern in product but not in all reactants)
            amide_pattern = Chem.MolFromSmarts("[C](=O)[N]")
            
            if prod_mol.HasSubstructMatch(amide_pattern):
                # Check if amide bond is newly formed (not present in all reactants)
                reactant_has_amide = any(mol.HasSubstructMatch(amide_pattern) for mol in react_mols)
                if not reactant_has_amide:
                    return True
                    
                # Or check for coupling of carboxylic acid/ester with amine
                carboxyl_pattern = Chem.MolFromSmarts("[C](=O)[OH,O]")
                amine_pattern = Chem.MolFromSmarts("[N;!$(N=*);!$(N#*)]")
                
                has_carboxyl = any(mol.HasSubstructMatch(carboxyl_pattern) for mol in react_mols)
                has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in react_mols)
                
                return has_carboxyl and has_amine
                
        except Exception:
            return False
            
        return False
    
    def detect_aryl_amine_coupling(self, rxn):
        """Detects aryl amine coupling reactions (e.g., Buchwald-Hartwig)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[1]
        reactants = rxn_parts[0]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check for aryl-nitrogen bond formation
            aryl_amine_pattern = Chem.MolFromSmarts("[c][N;!$(N=*);!$(N#*)]")
            
            if prod_mol.HasSubstructMatch(aryl_amine_pattern):
                # Check if this is a new C-N bond (coupling reaction)
                aryl_halide_pattern = Chem.MolFromSmarts("[c][Cl,Br,I]")
                amine_pattern = Chem.MolFromSmarts("[N;!$(N=*);!$(N#*);!$(N-c)]")
                
                has_aryl_halide = any(mol.HasSubstructMatch(aryl_halide_pattern) for mol in react_mols)
                has_free_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in react_mols)
                
                if has_aryl_halide and has_free_amine:
                    return True
                    
                # Alternative: check for boronic acid/ester coupling
                boronic_pattern = Chem.MolFromSmarts("[c][B]([OH,O])[OH,O]")
                has_boronic = any(mol.HasSubstructMatch(boronic_pattern) for mol in react_mols)
                
                if has_boronic and has_free_amine:
                    return True
                    
        except Exception:
            return False
            
        return False
