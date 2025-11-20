"""Generated evaluation code for: Late stage Sonogashira coupling for aryl-alkyne bond"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSonogashira(BaseScoring):
    """
    Evaluates whether a Sonogashira coupling reaction occurs at late stage
    for forming aryl-alkyne bonds. Checks for the presence of the specified
    bond pattern and Sonogashira-like transformations.
    """
    
    def __init__(self, config):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.reaction_type = config["parameters"]["reaction_type"]
        self.timing = config["parameters"]["timing"]
        
        # Compile the SMARTS pattern for the target bond
        self.bond_pattern = Chem.MolFromSmarts(self.bond_smarts)
        
        # SMARTS patterns for Sonogashira coupling detection
        # Terminal alkyne pattern
        self.terminal_alkyne = Chem.MolFromSmarts("[#6]#[#6][H]")
        # Aryl halide pattern (typically iodide, bromide)
        self.aryl_halide = Chem.MolFromSmarts("c[I,Br,Cl]")
        # Product aryl-alkyne pattern
        self.aryl_alkyne = Chem.MolFromSmarts("c-[#6]#[#6]")

    def route_scoring(self, x):
        """Convert depth fraction to score (0-10), favoring late stage"""
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Late stage is better, so higher depth fraction gives higher score
            return 10 * x

    def hit_condition(self, d):
        """Check if this reaction node represents a Sonogashira coupling"""
        metadata = d.get("metadata", {})
        
        # Check if mapped reaction SMILES is available
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        if not mapped_rxn:
            return False
            
        try:
            # Parse reaction SMILES: products >> reactants
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse molecules
            products = [Chem.MolFromSmiles(products_smiles)]
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            # Filter out None molecules
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Check if product contains the target aryl-alkyne bond pattern
            product_has_target = any(mol.HasSubstructMatch(self.bond_pattern) for mol in products)
            if not product_has_target:
                return False
            
            # Check if reactants contain terminal alkyne and aryl halide
            has_terminal_alkyne = any(mol.HasSubstructMatch(self.terminal_alkyne) for mol in reactants)
            has_aryl_halide = any(mol.HasSubstructMatch(self.aryl_halide) for mol in reactants)
            
            # Check if product has aryl-alkyne bond that wasn't in reactants
            product_has_aryl_alkyne = any(mol.HasSubstructMatch(self.aryl_alkyne) for mol in products)
            reactant_has_aryl_alkyne = any(mol.HasSubstructMatch(self.aryl_alkyne) for mol in reactants)
            
            # Sonogashira conditions:
            # 1. Reactants have terminal alkyne and aryl halide
            # 2. Product has aryl-alkyne bond
            # 3. Aryl-alkyne bond is newly formed (not present in reactants)
            return (has_terminal_alkyne and has_aryl_halide and 
                    product_has_aryl_alkyne and not reactant_has_aryl_alkyne)
                    
        except Exception:
            return False
