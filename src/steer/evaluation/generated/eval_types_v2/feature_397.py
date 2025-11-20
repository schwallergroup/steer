"""Generated evaluation code for: Late purine ring formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePurineRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage purine ring formation via cyclization.
    Detects when a purine ring system is formed through cyclization reactions,
    favoring routes where this occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_step = config["parameters"]["formation_step"]
        self.purine_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Later formation is better for late-stage cyclization.
        """
        if x < 0:
            return 0  # Purine ring formation not detected
        
        if self.timing == "late":
            # Higher score for later formation (closer to 1.0)
            return x * 10
        else:
            # Higher score for earlier formation (closer to 0.0)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves purine ring formation.
        Returns True if purine ring is formed in this step.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Count purine rings in reactants vs products
            reactant_purine_count = sum(
                len(mol.GetSubstructMatches(self.purine_pattern)) 
                for mol in reactants
            )
            
            product_purine_count = sum(
                len(mol.GetSubstructMatches(self.purine_pattern)) 
                for mol in products
            )
            
            # Check if purine ring was formed (more purine rings in products than reactants)
            purine_formed = product_purine_count > reactant_purine_count
            
            # Additional check for cyclization: ensure it's actually a ring-forming reaction
            if purine_formed:
                # Check if this is likely a cyclization by looking for intramolecular bond formation
                # This is a simplified heuristic - could be made more sophisticated
                return self._is_cyclization_reaction(reactants, products)
            
            return False
            
        except Exception:
            return False
    
    def _is_cyclization_reaction(self, reactants, products):
        """
        Heuristic to determine if this is a cyclization reaction.
        Checks if number of molecules decreases (intramolecular reaction).
        """
        # Simple heuristic: cyclization often reduces molecule count
        # or involves reactions with small molecules (water, etc.)
        if len(products) < len(reactants):
            return True
        
        # Check for typical cyclization pattern: one main reactant + small molecules
        large_reactants = [mol for mol in reactants if mol.GetNumAtoms() > 5]
        if len(large_reactants) == 1:
            return True
            
        return False
