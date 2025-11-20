"""Generated evaluation code for: Late thiazole ring formation via Hantzsch synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleHantzschFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiazole ring formation via Hantzsch synthesis.
    Rewards routes where thiazole rings are formed in later stages of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.formation_method = config["parameters"]["formation_method"]  # "hantzsch"
        self.timing = config["parameters"]["timing"]  # "late"
        
        # Compile SMARTS pattern for thiazole
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # Hantzsch synthesis components - typical reactants for thiazole formation
        self.hantzsch_patterns = [
            Chem.MolFromSmarts("[C;X3]([O,S])[C;X4]"),  # α-halo ketone pattern
            Chem.MolFromSmarts("[N;X3][C;X3]=[S;X2]"),   # thioamide pattern
            Chem.MolFromSmarts("[S;X2][C;X4]"),          # α-mercapto compound
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole formation doesn't happen
        else:
            # Late-stage formation gets higher score (closer to 1.0 gets score closer to 10)
            return 10 * x  # x is depth fraction, so later = higher score

    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a thiazole ring via Hantzsch synthesis"""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        product_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles)
            
            if not all(reactants) or not product:
                return False
            
            # Check if product has thiazole ring
            if not product.HasSubstructMatch(self.thiazole_pattern):
                return False
            
            # Check if any reactant already has thiazole ring (if so, not formation)
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.thiazole_pattern):
                    return False
            
            # Check for Hantzsch synthesis pattern
            return self._is_hantzsch_synthesis(reactants, product)
            
        except:
            return False

    def _is_hantzsch_synthesis(self, reactants, product) -> bool:
        """Check if the reaction matches Hantzsch thiazole synthesis pattern"""
        if len(reactants) < 2:
            return False
        
        # Count thiazole rings formed
        product_thiazole_count = len(product.GetSubstructMatches(self.thiazole_pattern))
        reactant_thiazole_count = sum(len(r.GetSubstructMatches(self.thiazole_pattern)) for r in reactants)
        
        # Must form at least one new thiazole ring
        if product_thiazole_count <= reactant_thiazole_count:
            return False
        
        # Look for characteristic Hantzsch reactant patterns
        pattern_matches = 0
        for reactant in reactants:
            for pattern in self.hantzsch_patterns:
                if reactant.HasSubstructMatch(pattern):
                    pattern_matches += 1
                    break
        
        # Need at least 2 different Hantzsch-type reactants
        return pattern_matches >= 2
