"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclopropaneFormation(BaseScoring):
    """
    Evaluates synthesis routes based on late-stage cyclopropane ring formation.
    Rewards routes where cyclopropane rings are formed in the final steps,
    typically via reactions like Corey-Chaykovsky or cyclopropanation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CC1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        self.cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't happen
        else:
            # Late-stage formation is better (closer to 1.0 depth fraction)
            if self.timing == "late":
                return 10 * x  # Higher score for later formation
            else:
                return 10 * (1 - x)  # Higher score for earlier formation
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves cyclopropane ring formation
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
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
            
            # Count cyclopropane rings in product
            product_cyclopropanes = len(product_mol.GetSubstructMatches(self.cyclopropane_pattern))
            
            # Count cyclopropane rings in all reactants
            reactant_cyclopropanes = sum(
                len(mol.GetSubstructMatches(self.cyclopropane_pattern)) 
                for mol in reactant_mols
            )
            
            # Check for ring formation (more rings in product than reactants)
            if self.direction == "formation":
                return product_cyclopropanes > reactant_cyclopropanes
            elif self.direction == "breaking":
                return reactant_cyclopropanes > product_cyclopropanes
            else:
                return product_cyclopropanes != reactant_cyclopropanes
                
        except Exception:
            return False
