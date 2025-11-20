"""Generated evaluation code for: Late piperidine ring formation via intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PiperidineRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage piperidine ring formation via intramolecular cyclization.
    
    Checks if a piperidine ring (C1CCNCC1) is formed at a specific depth in the synthesis route
    through intramolecular N-alkylation mechanism.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CCNCC1"
        self.target_formation_step = config["parameters"]["formation_step"]  # 6
        self.total_steps = config["parameters"]["total_steps"]  # 10
        self.mechanism = config["parameters"]["mechanism"]  # "intramolecular_alkylation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Score based on how close the ring formation occurs to the target step.
        Late-stage formation (closer to target) gets higher score.
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        # Convert depth fraction to actual step number
        actual_step = x * self.total_steps
        step_difference = abs(actual_step - self.target_formation_step)
        
        # Score inversely proportional to step difference, scaled 0-10
        max_difference = self.total_steps
        score = max(0, 10 * (1 - step_difference / max_difference))
        return score
        
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves piperidine ring formation via intramolecular cyclization.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
        except:
            return False
        
        # Check if piperidine ring is formed (absent in reactants, present in products)
        reactant_has_piperidine = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
        product_has_piperidine = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
        
        # Ring must be formed (not present in reactants but present in products)
        ring_formed = not reactant_has_piperidine and product_has_piperidine
        
        if not ring_formed:
            return False
            
        # Check for intramolecular mechanism - single reactant should contain both N and alkyl chain
        if len(reactants) == 1 and self._is_intramolecular_alkylation(reactants[0], products):
            return True
            
        return False
        
    def _is_intramolecular_alkylation(self, reactant, products) -> bool:
        """
        Check if the reaction represents intramolecular N-alkylation to form piperidine.
        """
        # Look for nitrogen and leaving group pattern in reactant
        # This is a simplified check - in practice, you might want more sophisticated pattern matching
        n_alkylation_pattern = Chem.MolFromSmarts("[N;H1,H2][CH2][CH2][CH2][CH2][CH2][Cl,Br,I,OTs,OMs]")
        
        if reactant.HasSubstructMatch(n_alkylation_pattern):
            return True
            
        # Alternative pattern for different chain lengths or leaving groups
        alt_pattern = Chem.MolFromSmarts("[N][CH2][CH2][CH2][CH2][CH2][C]")
        
        return reactant.HasSubstructMatch(alt_pattern)
