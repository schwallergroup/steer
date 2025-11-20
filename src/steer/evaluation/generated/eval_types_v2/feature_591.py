"""Generated evaluation code for: Late stage Sonogashira coupling for pyridine attachment"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSonogashiraPyridine(BaseScoring):
    """
    Evaluates whether a specific C-pyridine bond is formed via late-stage Sonogashira coupling.
    
    Checks for the disconnection of a C-pyridine bond where the reaction involves
    Sonogashira coupling chemistry (alkyne + halide coupling).
    """
    
    def __init__(self, config):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.timing = config["parameters"]["timing"]
        self.reaction_type = config["parameters"]["reaction_type"]
        
        # Compile the bond pattern for substructure matching
        self.bond_pattern = Chem.MolFromSmarts(self.bond_smarts)
        
        # Sonogashira reaction patterns
        self.alkyne_pattern = Chem.MolFromSmarts("[C]#[C]")
        self.halide_pattern = Chem.MolFromSmarts("[Cl,Br,I]")
        self.pyridine_pattern = Chem.MolFromSmarts("c1ccncc1")
    
    def route_scoring(self, x):
        if x < 0:
            return 0  # Bond disconnection doesn't happen
        
        if self.timing == "late":
            # For late-stage reactions, lower depth fractions are better
            return 1 - x
        else:
            # For early-stage reactions, higher depth fractions are better  
            return x
    
    def hit_condition(self, d):
        """Check if this reaction breaks the target C-pyridine bond via Sonogashira coupling"""
        
        # Get mapped reaction SMILES
        mapped_rxn = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check if product contains the target bond pattern
            if not prod_mol.HasSubstructMatch(self.bond_pattern):
                return False
            
            # Check if this is a Sonogashira-type disconnection
            has_alkyne = any(mol.HasSubstructMatch(self.alkyne_pattern) for mol in react_mols)
            has_halide_or_pyridine = any(
                mol.HasSubstructMatch(self.halide_pattern) or mol.HasSubstructMatch(self.pyridine_pattern) 
                for mol in react_mols
            )
            
            # Verify the bond is actually broken (C-pyridine bond exists in product but fragments are separated)
            pyridine_in_product = prod_mol.HasSubstructMatch(self.pyridine_pattern)
            pyridine_fragments = [mol for mol in react_mols if mol.HasSubstructMatch(self.pyridine_pattern)]
            
            # Check that we have the right disconnection pattern:
            # 1. Product has both alkyne-like carbon and pyridine
            # 2. Reactants separate these components
            # 3. Evidence of Sonogashira coupling (alkyne + halide/pyridine fragments)
            
            if pyridine_in_product and len(pyridine_fragments) > 0 and has_alkyne and has_halide_or_pyridine:
                return True
                
        except Exception:
            return False
            
        return False
