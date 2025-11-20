"""Generated evaluation code for: Early indole ring formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IndoleRingFormation(BaseScoring):
    """
    Evaluates early indole ring formation via cyclization.
    Checks if an indole ring is formed early in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "early":
            return 1 - x  # Early formation is better (lower depth fraction)
        elif self.timing == "late":
            return x  # Late formation is better (higher depth fraction)
        else:
            return 1 if x >= 0 else 0  # Just check if it happens
    
    def hit_condition(self, d) -> bool:
        """Check if indole ring formation occurs in this reaction step."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        try:
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    product_mols.append(mol)
                    
        except:
            return False
        
        if not reactant_mols or not product_mols:
            return False
            
        # Check for ring formation: indole present in products but not in reactants
        if self.direction == "formation":
            # Count indole rings in reactants
            reactant_indole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) 
                for mol in reactant_mols
            )
            
            # Count indole rings in products
            product_indole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) 
                for mol in product_mols
            )
            
            # Ring formation occurred if product has more indole rings than reactants
            return product_indole_count > reactant_indole_count
            
        elif self.direction == "breaking":
            # Count indole rings in reactants
            reactant_indole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) 
                for mol in reactant_mols
            )
            
            # Count indole rings in products
            product_indole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) 
                for mol in product_mols
            )
            
            # Ring breaking occurred if reactants have more indole rings than products
            return reactant_indole_count > product_indole_count
        
        return False
