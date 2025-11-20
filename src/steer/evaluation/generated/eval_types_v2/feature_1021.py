"""Generated evaluation code for: Late morpholine ring formation via intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MorpholineRingFormation(BaseScoring):
    """
    Evaluates late morpholine ring formation via intramolecular cyclization.
    
    Checks if a morpholine ring (C1COCCN1) is formed through intramolecular
    cyclization and rewards later formation in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.mechanism = config["parameters"]["mechanism"]
        self.morpholine_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is better for morpholine rings
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves morpholine ring formation via intramolecular cyclization"""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        if len(rxn) != 2:
            return False
            
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn[0].split(".") if p.strip()]
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn[1].split(".") if r.strip()]
        
        # Check if morpholine ring is formed (present in products but not in individual reactants)
        morpholine_in_products = any(mol and mol.HasSubstructMatch(self.morpholine_pattern) 
                                   for mol in products)
        
        if not morpholine_in_products:
            return False
        
        # Check if this is intramolecular cyclization
        # For intramolecular cyclization, we expect:
        # 1. One main reactant that doesn't contain the morpholine ring
        # 2. The reactant should contain the precursor atoms that form the ring
        return self._is_intramolecular_cyclization(reactants, products)
    
    def _is_intramolecular_cyclization(self, reactants, products):
        """Check if the reaction represents intramolecular cyclization to form morpholine"""
        # Filter out small molecules (likely reagents/catalysts)
        main_reactants = [mol for mol in reactants if mol and mol.GetNumAtoms() > 5]
        
        if len(main_reactants) != 1:
            return False
            
        main_reactant = main_reactants[0]
        
        # The main reactant should not already contain the morpholine ring
        if main_reactant.HasSubstructMatch(self.morpholine_pattern):
            return False
        
        # Check for precursor patterns that could cyclize to morpholine
        # Look for linear precursors with N and O separated by appropriate carbon chains
        precursor_patterns = [
            "NCCOC",  # Basic N-C-C-O-C chain that could cyclize
            "OCCNC",  # O-C-C-N-C chain
            "NCCO",   # N-C-C-O chain (partial)
        ]
        
        for pattern_smarts in precursor_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and main_reactant.HasSubstructMatch(pattern):
                return True
        
        return False
