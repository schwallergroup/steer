"""Generated evaluation code for: Late stage imidazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageImidazoleFormation(BaseScoring):
    """
    Evaluates whether imidazole ring formation occurs late in the synthesis route.
    Rewards routes where the specified imidazole ring is formed in later stages
    of the synthesis, indicating late-stage cyclization strategy.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage formation, higher depth fractions get better scores.
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Later formation is better - reward higher depth fractions
            return x * 10
        elif self.timing == "early":
            # Earlier formation is better - reward lower depth fractions
            return (1 - x) * 10
        else:
            # Default case - moderate preference for later stages
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves imidazole ring formation.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Count imidazole rings in reactants and products
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactants)
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in products)
            
            # Check for ring formation (more rings in products than reactants)
            if self.direction == "formation":
                return product_rings > reactant_rings
            elif self.direction == "breaking":
                return reactant_rings > product_rings
            else:
                # Default to formation
                return product_rings > reactant_rings
                
        except Exception:
            # Handle any parsing errors
            return False
