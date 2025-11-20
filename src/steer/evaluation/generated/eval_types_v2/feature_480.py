"""Generated evaluation code for: Early piperazine ring formation via double N-alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PiperazineRingFormation(BaseScoring):
    """
    Evaluates early piperazine ring formation via double N-alkylation cyclization.
    Detects when a piperazine ring (N1CCNCC1) is formed through cyclization reactions
    and scores based on how early this occurs in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "N1CCNCC1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.formation_method = config["parameters"]["formation_method"]  # "cyclization"
        
    def route_scoring(self, x) -> float:
        """
        Score based on depth fraction where piperazine formation occurs.
        For early timing, lower depth fractions get higher scores.
        """
        if x < 0:
            return 0  # Ring formation not detected
        
        if self.timing == "early":
            # Early formation preferred - higher score for lower depth
            return 1 - x
        else:
            # Could extend for other timing preferences
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step forms a piperazine ring via cyclization.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check if piperazine ring is formed (present in products but not reactants)
            piperazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if piperazine_pattern is None:
                return False
            
            # Check if any product contains piperazine
            piperazine_in_products = any(
                mol.HasSubstructMatch(piperazine_pattern) for mol in products
            )
            
            # Check if piperazine is already present in reactants
            piperazine_in_reactants = any(
                mol.HasSubstructMatch(piperazine_pattern) for mol in reactants
            )
            
            # Ring formation occurs if piperazine is in products but not in reactants
            if piperazine_in_products and not piperazine_in_reactants:
                # Additional check for cyclization: count nitrogen atoms
                # Piperazine formation should involve connecting two nitrogens
                return self._is_cyclization_reaction(reactants, products, piperazine_pattern)
            
            return False
            
        except Exception:
            return False
    
    def _is_cyclization_reaction(self, reactants, products, piperazine_pattern):
        """
        Additional validation that this is a cyclization reaction forming piperazine.
        Checks for reduction in molecular count and presence of suitable precursors.
        """
        # Simple heuristic: cyclization typically reduces molecule count
        if len(products) >= len(reactants):
            return False
            
        # Check if reactants contain suitable nitrogen-containing precursors
        nitrogen_pattern = Chem.MolFromSmarts("[N]")
        reactant_nitrogens = sum(
            len(mol.GetSubstructMatches(nitrogen_pattern)) for mol in reactants
        )
        
        # Piperazine formation should involve at least 2 nitrogens
        if reactant_nitrogens < 2:
            return False
            
        return True
