"""Generated evaluation code for: Late stage morpholine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageMorpholineFormation(BaseScoring):
    """
    Evaluates routes based on late-stage morpholine ring formation.
    
    Checks if a morpholine ring (C1COCCN1) is formed late in the synthesis,
    preferring formation in the final steps of the route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1COCCN1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10), preferring late formation."""
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation gets higher score
            # x=1 (final step) -> score=1, x=0 (early) -> score=0
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """Check if morpholine ring formation occurs in this reaction step."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        # Product side (left) and reactant side (right)
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
                
            reactant_mols = []
            for smi in reactant_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None:
                    reactant_mols.append(mol)
                    
            if not reactant_mols:
                return False
                
            # Create morpholine pattern
            morpholine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if morpholine_pattern is None:
                return False
                
            # Check for ring formation: morpholine present in product but not in any reactant
            product_has_morpholine = product_mol.HasSubstructMatch(morpholine_pattern)
            
            if not product_has_morpholine:
                return False
                
            # Check that morpholine is not present in reactants (indicating formation)
            for reactant_mol in reactant_mols:
                if reactant_mol.HasSubstructMatch(morpholine_pattern):
                    return False  # Ring already exists in reactant
                    
            return True  # Morpholine formed in this step
            
        except Exception:
            return False
