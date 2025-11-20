"""Generated evaluation code for: Late stage pyridine ring formation via annulation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates late-stage formation of a specific ring system via annulation reactions.
    Checks if a target ring (defined by SMARTS) is formed late in the synthesis
    through an annulation process.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.stage = config["parameters"]["stage"]  # "late", "early", "any"
        self.method = config["parameters"]["method"]  # "annulation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.stage == "late":
            # Late-stage formation preferred - higher score for later depths
            return 10 * x  # x is depth fraction, so later = higher score
        elif self.stage == "early":
            # Early-stage formation preferred
            return 10 * (1 - x)
        else:  # "any" stage
            return 10  # Full score if ring formation occurs at any stage
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents annulation ring formation
        of the target ring system.
        """
        # Check if reaction has mapped SMILES
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check if product contains the target ring
            if not prod_mol.HasSubstructMatch(self.ring_pattern):
                return False
                
            # Check if any reactant already contains the complete target ring
            for react_mol in react_mols:
                if react_mol.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already exists, not formation
            
            # Check for annulation pattern - ring formation from acyclic precursors
            # Count ring systems in reactants vs product
            prod_ring_count = self._count_ring_systems(prod_mol)
            total_react_ring_count = sum(self._count_ring_systems(mol) for mol in react_mols)
            
            # Ring formation should increase ring count
            if prod_ring_count <= total_react_ring_count:
                return False
                
            # Additional check for annulation-like transformation
            # Look for increase in ring size or new ring formation
            return self._is_annulation_pattern(react_mols, prod_mol)
            
        except Exception:
            return False
    
    def _count_ring_systems(self, mol) -> int:
        """Count the number of separate ring systems in a molecule."""
        if not mol:
            return 0
        return mol.GetRingInfo().NumRings()
    
    def _is_annulation_pattern(self, reactants, product) -> bool:
        """
        Check if the transformation matches annulation pattern.
        Annulation typically involves cyclization with incorporation of 
        additional atoms/bonds to form a new ring.
        """
        # Simple heuristic: check if we have fewer molecules in reactants
        # and the product has the target ring that wasn't in any single reactant
        if len(reactants) < 2:
            return False
            
        # Verify none of the reactants contain the complete target ring
        for reactant in reactants:
            if reactant.HasSubstructMatch(self.ring_pattern):
                return False
                
        # Product should contain the target ring
        return product.HasSubstructMatch(self.ring_pattern)
