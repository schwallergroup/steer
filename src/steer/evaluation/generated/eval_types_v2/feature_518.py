"""Generated evaluation code for: Early stereocenter establishment via ketone reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyStereocenterKetoneReduction(BaseScoring):
    """
    Evaluates routes for early stereocenter establishment via ketone reduction.
    
    Checks if ketone reduction reactions occur early in the synthesis route
    (within specified depth threshold) to establish stereocenters that guide
    subsequent transformations.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 7)
        self.timing = config.get("timing", "early")
        
        # SMARTS patterns for ketone reduction detection
        self.ketone_pattern = Chem.MolFromSmarts("[C]=[O]")
        self.alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
        self.chiral_carbon_pattern = Chem.MolFromSmarts("[C@,C@@]")
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10)"""
        if x < 0:
            return 0  # Condition not met
        
        if self.timing == "early":
            # Reward earlier occurrence, penalize later
            if x <= (self.depth_threshold / 10.0):  # Convert to fraction
                return 10 * (1 - x)  # Higher score for earlier depths
            else:
                return 0  # Too late
        else:
            # For other timing preferences
            return max(0, 10 * (1 - abs(x - 0.5)))
    
    def hit_condition(self, d) -> bool:
        """Check if reaction is a ketone reduction creating a stereocenter"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check for ketone to alcohol conversion
            ketone_in_reactants = any(mol.HasSubstructMatch(self.ketone_pattern) for mol in reactants)
            alcohol_in_products = any(mol.HasSubstructMatch(self.alcohol_pattern) for mol in products)
            
            if not (ketone_in_reactants and alcohol_in_products):
                return False
            
            # Check if new stereocenter is created
            reactant_stereocenters = sum(len([atom for atom in mol.GetAtoms() 
                                           if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]) 
                                       for mol in reactants)
            
            product_stereocenters = sum(len([atom for atom in mol.GetAtoms() 
                                          if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]) 
                                      for mol in products)
            
            # Additional check using SMARTS for chiral carbons
            reactant_chiral_matches = sum(len(mol.GetSubstructMatches(self.chiral_carbon_pattern)) 
                                        for mol in reactants)
            product_chiral_matches = sum(len(mol.GetSubstructMatches(self.chiral_carbon_pattern)) 
                                       for mol in products)
            
            # Return True if stereocenter count increases
            return (product_stereocenters > reactant_stereocenters or 
                   product_chiral_matches > reactant_chiral_matches)
            
        except Exception:
            return False
