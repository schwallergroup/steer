"""Generated evaluation code for: Benzyl protecting group with halogenated substrate"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupHalide(BaseScoring):
    """
    Checks if benzyl protecting group is used with halogenated substrates.
    This is problematic because hydrogenolysis conditions for benzyl deprotection
    can also reduce aryl halides, leading to undesired side reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
        
        # SMARTS patterns for detection
        self.benzyl_ether_pattern = "[#6]-[#8]-[CH2]-c1ccccc1"  # Benzyl ether
        self.benzyl_ester_pattern = "[#6]-[#6](=O)-[#8]-[CH2]-c1ccccc1"  # Benzyl ester
        self.aryl_halide_pattern = "c[F,Cl,Br,I]"  # Aryl halide
        self.alkyl_halide_pattern = "[CH3,CH2,CH][F,Cl,Br,I]"  # Alkyl halide (also reducible)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if x < 0:
                return 0  # Condition not met
            else:
                return 1  # Problematic combination found
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """
        Check if this reaction involves benzyl protection/deprotection with halogenated substrates.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check for benzyl protection formation
            benzyl_protection = self._detect_benzyl_protection(reactants, products)
            
            # Check for benzyl deprotection
            benzyl_deprotection = self._detect_benzyl_deprotection(reactants, products)
            
            # Check if any molecule contains reducible halides
            all_mols = reactants + products
            has_reducible_halide = any(self._has_reducible_halide(mol) for mol in all_mols)
            
            # Problematic if benzyl group is used (protection or deprotection) with halogenated substrates
            return (benzyl_protection or benzyl_deprotection) and has_reducible_halide
            
        except Exception:
            return False
    
    def _detect_benzyl_protection(self, reactants, products):
        """Check if benzyl protecting group is being installed."""
        benzyl_ether_smarts = Chem.MolFromSmarts(self.benzyl_ether_pattern)
        benzyl_ester_smarts = Chem.MolFromSmarts(self.benzyl_ester_pattern)
        
        # Count benzyl groups in reactants vs products
        reactant_benzyl_count = sum(
            len(mol.GetSubstructMatches(benzyl_ether_smarts)) + 
            len(mol.GetSubstructMatches(benzyl_ester_smarts))
            for mol in reactants
        )
        
        product_benzyl_count = sum(
            len(mol.GetSubstructMatches(benzyl_ether_smarts)) + 
            len(mol.GetSubstructMatches(benzyl_ester_smarts))
            for mol in products
        )
        
        return product_benzyl_count > reactant_benzyl_count
    
    def _detect_benzyl_deprotection(self, reactants, products):
        """Check if benzyl protecting group is being removed."""
        benzyl_ether_smarts = Chem.MolFromSmarts(self.benzyl_ether_pattern)
        benzyl_ester_smarts = Chem.MolFromSmarts(self.benzyl_ester_pattern)
        
        # Count benzyl groups in reactants vs products
        reactant_benzyl_count = sum(
            len(mol.GetSubstructMatches(benzyl_ether_smarts)) + 
            len(mol.GetSubstructMatches(benzyl_ester_smarts))
            for mol in reactants
        )
        
        product_benzyl_count = sum(
            len(mol.GetSubstructMatches(benzyl_ether_smarts)) + 
            len(mol.GetSubstructMatches(benzyl_ester_smarts))
            for mol in products
        )
        
        return reactant_benzyl_count > product_benzyl_count
    
    def _has_reducible_halide(self, mol):
        """Check if molecule contains halides that are reducible under hydrogenolysis conditions."""
        aryl_halide_smarts = Chem.MolFromSmarts(self.aryl_halide_pattern)
        alkyl_halide_smarts = Chem.MolFromSmarts(self.alkyl_halide_pattern)
        
        return (mol.HasSubstructMatch(aryl_halide_smarts) or 
                mol.HasSubstructMatch(alkyl_halide_smarts))
