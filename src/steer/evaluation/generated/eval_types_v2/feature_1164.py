"""Generated evaluation code for: Late indole ring formation via Fischer synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIndoleFischerSynthesis(BaseScoring):
    """
    Evaluates synthesis routes for late-stage indole ring formation via Fischer indole synthesis.
    
    This class checks if the indole ring (c1ccc2[nH]ccc2c1) is formed late in the synthesis
    through Fischer indole synthesis, which typically involves the reaction of a phenylhydrazine
    with a ketone or aldehyde to form the indole ring system.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.indole_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # Fischer indole synthesis patterns
        self.phenylhydrazine_pattern = Chem.MolFromSmarts("c1ccccc1NN")
        self.ketone_pattern = Chem.MolFromSmarts("[CX3](=O)[CH2,CH3]")
        self.aldehyde_pattern = Chem.MolFromSmarts("[CX3H1](=O)")
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage formation, later depths get higher scores.
        """
        if x < 0:
            return 0  # Fischer indole synthesis doesn't occur
        
        if self.timing == "late":
            # Later formation is better, x is depth fraction (0=root, 1=leaves)
            return x * 10
        else:
            # Earlier formation is better
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents Fischer indole synthesis forming an indole ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains indole ring
            if not product.HasSubstructMatch(self.indole_pattern):
                return False
            
            # Check if indole ring is formed (not present in any reactant)
            indole_formed = True
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.indole_pattern):
                    indole_formed = False
                    break
            
            if not indole_formed:
                return False
            
            # Check for Fischer indole synthesis pattern
            # Look for phenylhydrazine and carbonyl compound in reactants
            has_phenylhydrazine = False
            has_carbonyl = False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.phenylhydrazine_pattern):
                    has_phenylhydrazine = True
                if (reactant.HasSubstructMatch(self.ketone_pattern) or 
                    reactant.HasSubstructMatch(self.aldehyde_pattern)):
                    has_carbonyl = True
            
            # Fischer indole synthesis requires both components
            return has_phenylhydrazine and has_carbonyl
            
        except Exception:
            return False
