"""Generated evaluation code for: Early indole core annulation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyIndoleAnnulation(BaseScoring):
    """
    Evaluates whether indole core formation occurs early in the synthesis route.
    Detects indole ring formation and scores based on timing preference.
    """
    
    def __init__(self, config: Dict):
        self.indole_smarts = config["parameters"]["ring_smarts"]  # "c1c[nH]c2ccccc12"
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Indole formation doesn't happen
        
        if self.timing == "early":
            return 1 - x  # Earlier formation gets higher score
        elif self.timing == "late":
            return x  # Later formation gets higher score
        else:
            return 0.5  # Neutral if timing not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves indole core formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            # Parse products and reactants
            products_mol = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products_mol or not reactants:
                return False
                
            # Create indole pattern
            indole_pattern = Chem.MolFromSmarts(self.indole_smarts)
            if not indole_pattern:
                return False
            
            # Check for indole formation
            if self.direction == "formation":
                # Indole should be present in products but absent in all reactants
                products_has_indole = products_mol.HasSubstructMatch(indole_pattern)
                reactants_have_indole = any(mol.HasSubstructMatch(indole_pattern) for mol in reactants)
                
                return products_has_indole and not reactants_have_indole
                
            elif self.direction == "breaking":
                # Indole should be present in reactants but absent in products
                products_has_indole = products_mol.HasSubstructMatch(indole_pattern)
                reactants_have_indole = any(mol.HasSubstructMatch(indole_pattern) for mol in reactants)
                
                return not products_has_indole and reactants_have_indole
                
        except Exception:
            return False
            
        return False
