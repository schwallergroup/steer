"""Generated evaluation code for: Late stage triazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTriazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage triazole ring formation.
    
    Detects when a triazole ring (c1nncn1) is formed in a reaction and 
    scores based on how late in the synthesis this occurs. Later formation
    is scored higher as it represents a more strategic late-stage approach.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Triazole formation doesn't happen
        else:
            # Late-stage formation is better (higher score)
            # x is depth fraction, so 1-x gives higher scores for later reactions
            return 1 - x
    
    def hit_condition(self, d):
        """
        Check if this reaction step involves triazole ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            # Products (left side) and reactants (right side)
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            product_mols = [Chem.MolFromSmiles(products)]
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not all(mol for mol in product_mols + reactant_mols):
                return False
            
            # Count triazole rings in products vs reactants
            product_triazoles = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                  for mol in product_mols if mol)
            reactant_triazoles = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                   for mol in reactant_mols if mol)
            
            # Check for ring formation (more triazoles in products than reactants)
            if self.direction == "formation":
                return product_triazoles > reactant_triazoles
            else:
                return product_triazoles < reactant_triazoles
                
        except Exception:
            return False
