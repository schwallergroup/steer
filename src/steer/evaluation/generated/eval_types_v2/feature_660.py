"""Generated evaluation code for: Late stage reductive oxindole cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageReductiveOxindoleCyclization(BaseScoring):
    """
    Evaluates late-stage reductive oxindole cyclization in synthesis routes.
    Checks for the formation of oxindole rings (C1C(=O)Nc2ccccc21) via reductive 
    cyclization, typically involving nitro reduction followed by intramolecular 
    cyclization with an ester group.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.oxindole_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # Patterns for detecting reductive cyclization components
        self.nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
        self.ester_pattern = Chem.MolFromSmarts("C(=O)O[C,c]")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        else:
            # For late-stage preference, lower depth fraction is better
            if self.timing == "late":
                return 1 - x  # Later is better (higher score for lower depth)
            else:
                return x  # Earlier is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents reductive oxindole cyclization.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactants.append(mol)
            
            products = []            
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check if oxindole ring is formed (present in products but not reactants)
            oxindole_in_products = any(mol.HasSubstructMatch(self.oxindole_pattern) for mol in products)
            oxindole_in_reactants = any(mol.HasSubstructMatch(self.oxindole_pattern) for mol in reactants)
            
            if not (oxindole_in_products and not oxindole_in_reactants):
                return False
            
            # Check for reductive cyclization characteristics
            if self.formation_method == "reductive_cyclization":
                return self._is_reductive_cyclization(reactants, products)
                
            return True
            
        except Exception:
            return False
    
    def _is_reductive_cyclization(self, reactants, products) -> bool:
        """
        Check if the reaction involves reductive cyclization by looking for:
        1. Nitro group in reactants that's reduced/cyclized in products
        2. Ester group that participates in cyclization
        """
        # Check for nitro group in reactants
        has_nitro_reactant = any(mol.HasSubstructMatch(self.nitro_pattern) for mol in reactants)
        
        # Check for ester group in reactants
        has_ester_reactant = any(mol.HasSubstructMatch(self.ester_pattern) for mol in reactants)
        
        # Nitro should be consumed (reduced presence in products)
        nitro_count_reactants = sum(len(mol.GetSubstructMatches(self.nitro_pattern)) for mol in reactants)
        nitro_count_products = sum(len(mol.GetSubstructMatches(self.nitro_pattern)) for mol in products)
        
        # Should have nitro in reactants, and reduced nitro in products
        nitro_reduced = has_nitro_reactant and (nitro_count_products < nitro_count_reactants)
        
        # For reductive cyclization, we expect either nitro reduction or presence of ester
        return nitro_reduced or has_ester_reactant
