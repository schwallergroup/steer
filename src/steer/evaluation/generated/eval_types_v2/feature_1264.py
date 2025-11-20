"""Generated evaluation code for: Late pyrrolidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrrolidineFormation(BaseScoring):
    """
    Evaluates whether pyrrolidine ring formation occurs late in the synthesis route.
    Returns higher scores when pyrrolidine ring formation happens closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "[#7]1[#6][#6][#6][#6]1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        elif self.timing == "early":
            return x  # Earlier formation gets higher score
        else:
            return 0.5  # Neutral if timing preference not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves pyrrolidine ring formation.
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
            
            if None in reactants or None in products:
                return False
            
            # Create SMARTS pattern for pyrrolidine
            pyrrolidine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if pyrrolidine_pattern is None:
                return False
            
            # Count pyrrolidine rings in reactants and products
            reactant_pyrrolidine_count = sum(
                len(mol.GetSubstructMatches(pyrrolidine_pattern)) 
                for mol in reactants if mol is not None
            )
            
            product_pyrrolidine_count = sum(
                len(mol.GetSubstructMatches(pyrrolidine_pattern))
                for mol in products if mol is not None
            )
            
            if self.direction == "formation":
                # Ring formation: more pyrrolidine rings in products than reactants
                return product_pyrrolidine_count > reactant_pyrrolidine_count
            elif self.direction == "breaking":
                # Ring breaking: fewer pyrrolidine rings in products than reactants
                return reactant_pyrrolidine_count > product_pyrrolidine_count
            else:
                # Any change in pyrrolidine ring count
                return reactant_pyrrolidine_count != product_pyrrolidine_count
                
        except Exception:
            return False
