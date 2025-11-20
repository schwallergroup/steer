"""Generated evaluation code for: Early Schmidt reaction ring expansion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySchmidtRingExpansion(BaseScoring):
    """
    Evaluates whether a Schmidt reaction for ring expansion occurs early in the synthesis route.
    Schmidt reactions typically convert ketones to lactams with ring expansion.
    """
    
    def __init__(self, config: Dict):
        # Early means we want it to happen at low depth (close to target)
        self.target_depth = 0.3  # Prefer within first 30% of route
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Schmidt reaction doesn't happen
        else:
            # Reward early occurrence, penalize late occurrence
            if x <= self.target_depth:
                return 10  # Perfect score for early Schmidt reaction
            else:
                # Linear decay as depth increases beyond target
                return max(0, 10 * (1 - (x - self.target_depth) / (1 - self.target_depth)))
    
    def hit_condition(self, d) -> bool:
        """
        Detects Schmidt reaction by looking for characteristic patterns:
        - Ketone to lactam conversion with ring expansion
        - Presence of azide or similar nitrogen source
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
                
            # Check for Schmidt reaction patterns
            return self._detect_schmidt_ring_expansion(reactants, products)
            
        except Exception:
            return False
    
    def _detect_schmidt_ring_expansion(self, reactants, products):
        """
        Detect Schmidt ring expansion by checking:
        1. Presence of ketone in reactants
        2. Presence of lactam in products  
        3. Ring size increase
        4. Nitrogen source (azide or similar)
        """
        # Define SMARTS patterns
        ketone_pattern = Chem.MolFromSmarts("[#6]C(=O)[#6]")  # Ketone
        cyclic_ketone_pattern = Chem.MolFromSmarts("[#6]1[#6][#6]C(=O)[#6][#6]1")  # 6-membered cyclic ketone
        lactam_pattern = Chem.MolFromSmarts("[#6]1[#6][#6]C(=O)N[#6][#6]1")  # 7-membered lactam
        azide_pattern = Chem.MolFromSmarts("N=[N+]=[N-]")  # Azide
        
        if not all([ketone_pattern, cyclic_ketone_pattern, lactam_pattern, azide_pattern]):
            return False
        
        # Check reactants for ketone and nitrogen source
        has_cyclic_ketone = False
        has_nitrogen_source = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(cyclic_ketone_pattern):
                has_cyclic_ketone = True
            if reactant.HasSubstructMatch(azide_pattern):
                has_nitrogen_source = True
        
        # Check products for lactam
        has_lactam = False
        for product in products:
            if product.HasSubstructMatch(lactam_pattern):
                has_lactam = True
        
        # Schmidt ring expansion: cyclic ketone + nitrogen source -> lactam
        return has_cyclic_ketone and has_nitrogen_source and has_lactam
