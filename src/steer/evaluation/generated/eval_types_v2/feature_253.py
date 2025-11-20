"""Generated evaluation code for: Late piperidine ring closure via amine-ketone cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperidineRingClosure(BaseScoring):
    """
    Evaluates synthesis routes for late-stage piperidine ring formation via amine-ketone cyclization.
    Checks if a piperidine ring (C1CCNCC1) is formed at a specific depth in the synthesis tree.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.formation_step = config["parameters"]["formation_step"]
        self.total_steps = config["parameters"]["total_steps"]
        self.timing = config["parameters"]["timing"]
        self.target_depth_fraction = self.formation_step / self.total_steps
        
        # Compile the SMARTS pattern for piperidine ring
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # SMARTS patterns for detecting amine-ketone cyclization
        self.ketone_pattern = Chem.MolFromSmarts("[C](=O)")
        self.amine_pattern = Chem.MolFromSmarts("[N]")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        # Convert depth fraction to score (0-10)
        # For late-stage timing, prefer formation closer to target depth
        target_score = 10
        if self.timing == "late":
            # Penalize early formation, reward formation at target depth
            if x < self.target_depth_fraction:
                # Too early - linear penalty
                penalty = (self.target_depth_fraction - x) * 15
                return max(0, target_score - penalty)
            else:
                # At or after target - small penalty for being too late
                penalty = (x - self.target_depth_fraction) * 5
                return max(0, target_score - penalty)
        
        return target_score * (1 - abs(x - self.target_depth_fraction))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves piperidine ring formation via amine-ketone cyclization.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
        
        rxn_smiles = metadata["mapped_reaction_smiles"]
        try:
            # Parse reaction: products >> reactants
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            products_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            # Check if piperidine ring is formed (present in products but not in reactants)
            has_piperidine_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            has_piperidine_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            
            if has_piperidine_in_products and not has_piperidine_in_reactants:
                # Additional check for amine-ketone cyclization pattern
                return self._is_amine_ketone_cyclization(reactants, products)
            
            return False
            
        except Exception:
            return False
    
    def _is_amine_ketone_cyclization(self, reactants, products):
        """
        Check if the reaction involves cyclization between amine and ketone functionalities.
        """
        # Look for reactants containing both amine and ketone groups
        for reactant in reactants:
            has_ketone = reactant.HasSubstructMatch(self.ketone_pattern)
            has_amine = reactant.HasSubstructMatch(self.amine_pattern)
            
            if has_ketone and has_amine:
                # Check if this reactant forms a piperidine ring in products
                # This is a simplified check - in practice, you might want more sophisticated
                # atom mapping analysis
                return True
        
        # Alternative: separate amine and ketone reactants that combine
        ketone_reactants = [mol for mol in reactants if mol.HasSubstructMatch(self.ketone_pattern)]
        amine_reactants = [mol for mol in reactants if mol.HasSubstructMatch(self.amine_pattern)]
        
        return len(ketone_reactants) > 0 and len(amine_reactants) > 0
