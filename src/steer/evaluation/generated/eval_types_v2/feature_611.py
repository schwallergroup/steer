"""Generated evaluation code for: Late stage lactam ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageLactamFormation(BaseScoring):
    """
    Evaluates late stage lactam ring formation in synthesis routes.
    Detects when a specific lactam ring pattern is formed and scores based on timing.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation is better, score decreases with earlier timing
        elif self.timing == "early":
            return x  # Earlier formation is better, score increases with later timing
        else:
            return 0.5  # Neutral scoring if timing preference not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction involves formation of the target lactam ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains the target lactam ring
            product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_ring:
                return False
                
            # Check reactants - none should have the complete ring if this is formation
            if self.direction == "formation":
                reactant_mols = []
                for reactant_smiles in reactants_smiles.split("."):
                    reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                    if reactant_mol:
                        reactant_mols.append(reactant_mol)
                
                # If any reactant already has the complete ring, this isn't ring formation
                for reactant_mol in reactant_mols:
                    if reactant_mol.HasSubstructMatch(self.ring_pattern):
                        return False
                        
                return True
                
            elif self.direction == "break":
                # For ring breaking, check if reactants have the ring
                for reactant_smiles in reactants_smiles.split("."):
                    reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                    if reactant_mol and reactant_mol.HasSubstructMatch(self.ring_pattern):
                        return True
                return False
                
        except Exception:
            return False
            
        return False
