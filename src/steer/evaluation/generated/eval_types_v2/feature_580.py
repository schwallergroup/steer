"""Generated evaluation code for: Late isoxazole ring formation via cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIsoxazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage isoxazole ring formation via cycloaddition.
    Checks if an isoxazole ring (c1oncc1) is formed through cycloaddition reactions
    and scores based on how late in the synthesis this occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.reaction_type = config["parameters"]["reaction_type"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Isoxazole formation not found
        
        if self.timing == "late":
            # Late-stage formation is preferred, so lower depth fractions are better
            return 1 - x  # Convert to 0-1 scale where 1 is best (latest)
        else:
            # If early timing was preferred (though not in this case)
            return x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents isoxazole ring formation via cycloaddition.
        """
        metadata = d.get("metadata", {})
        
        # Check if this is identified as a cycloaddition reaction
        if not self._is_cycloaddition(metadata):
            return False
        
        # Get the mapped reaction SMILES
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains isoxazole ring
            if not product_mol.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if isoxazole ring is NOT present in any reactants (ring formation)
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already exists, not formation
            
            return True
            
        except Exception:
            return False
    
    def _is_cycloaddition(self, metadata: Dict) -> bool:
        """
        Check if the reaction is identified as a cycloaddition.
        """
        # Check policy name for cycloaddition indicators
        policy_name = metadata.get("policy_name", "").lower()
        if "cycloadd" in policy_name or "dipolar" in policy_name:
            return True
        
        # Check reaction template or classification if available
        template = metadata.get("template", "").lower()
        if "cycloadd" in template or "[3+2]" in template:
            return True
        
        reaction_class = metadata.get("reaction_class", "").lower()
        if "cycloadd" in reaction_class or "dipolar" in reaction_class:
            return True
        
        return False
