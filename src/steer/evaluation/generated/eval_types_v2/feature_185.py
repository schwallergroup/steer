"""Generated evaluation code for: Early thiazole ring formation via Hantzsch synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyThiazoleHantzsch(BaseScoring):
    """
    Evaluates synthesis routes for early thiazole ring formation via Hantzsch synthesis.
    
    This class checks if a thiazole ring (c1scnc1) is formed early in the synthesis
    using Hantzsch condensation, which typically involves the reaction of an 
    α-haloketone with a thioamide.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.method = config["parameters"]["method"]  # "hantzsch_synthesis"
        
        # Hantzsch synthesis pattern: α-haloketone + thioamide
        self.haloketone_pattern = "[CX4][CX3](=[OX1])[CX4][ClX1,BrX1,IX1]"  # α-haloketone
        self.thioamide_pattern = "[NX3][CX3]([#1,CX4])=[SX1]"  # thioamide
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole formation doesn't happen
        else:
            # Early formation is better - convert depth fraction to score
            if self.timing == "early":
                return (1 - x) * 10  # Earlier = higher score
            else:
                return x * 10  # Later = higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a thiazole ring via Hantzsch synthesis.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        # Parse molecules
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
        except:
            return False
        
        # Check if thiazole ring is formed in product
        thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if not product.HasSubstructMatch(thiazole_pattern):
            return False
        
        # Check if thiazole ring was absent in reactants
        thiazole_in_reactants = any(r.HasSubstructMatch(thiazole_pattern) for r in reactants)
        if thiazole_in_reactants:
            return False  # Ring was already present, not formed in this step
        
        # Check for Hantzsch synthesis pattern in reactants
        haloketone_pattern = Chem.MolFromSmarts(self.haloketone_pattern)
        thioamide_pattern = Chem.MolFromSmarts(self.thioamide_pattern)
        
        has_haloketone = any(r.HasSubstructMatch(haloketone_pattern) for r in reactants)
        has_thioamide = any(r.HasSubstructMatch(thioamide_pattern) for r in reactants)
        
        # Hantzsch synthesis requires both α-haloketone and thioamide
        return has_haloketone and has_thioamide
