"""Generated evaluation code for: Double bond isomerization cycling Δ³→Δ²→Δ³"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DoubleBondIsomerizationCycling(MultiRxnCondBase):
    """
    Detects double bond isomerization cycling between Δ³ and Δ² positions in cephalosporin structures.
    Penalizes routes that involve converting stable Δ³ to unstable Δ² and back to Δ³, creating synthetic detours.
    """
    
    def __init__(self, config):
        self.cycle_count = config.get("cycle_count", 2)
        # SMARTS patterns for cephalosporin Δ³ and Δ² isomers
        self.delta3_pattern = "[#6]1[#6][#6](=[#6][#16]1)[#6](=[#8])[#7]"  # Δ³ cephalosporin core
        self.delta2_pattern = "[#6]1=[#6][#6]([#6][#16]1)[#6](=[#8])[#7]"   # Δ² cephalosporin core
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track isomerization sequence
        isomerization_sequence = []
        
        for rxn in reactions:
            if self.detect_delta3_to_delta2(rxn):
                isomerization_sequence.append("3to2")
            elif self.detect_delta2_to_delta3(rxn):
                isomerization_sequence.append("2to3")
        
        # Check for cycling pattern (Δ³→Δ²→Δ³)
        cycling_detected = self.detect_cycling_pattern(isomerization_sequence)
        
        return cycling_detected, len(reactions)
    
    def detect_delta3_to_delta2(self, rxn):
        """Detect Δ³ to Δ² isomerization reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check if reactant has Δ³ and product has Δ² pattern
        has_delta3_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.delta3_pattern)) 
                                 for mol in reactants)
        has_delta2_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.delta2_pattern)) 
                                for mol in products)
        
        return has_delta3_reactant and has_delta2_product
    
    def detect_delta2_to_delta3(self, rxn):
        """Detect Δ² to Δ³ isomerization reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check if reactant has Δ² and product has Δ³ pattern
        has_delta2_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.delta2_pattern)) 
                                 for mol in reactants)
        has_delta3_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.delta3_pattern)) 
                                for mol in products)
        
        return has_delta2_reactant and has_delta3_product
    
    def detect_cycling_pattern(self, sequence):
        """Detect if the sequence contains the specified number of complete cycles"""
        if len(sequence) < self.cycle_count * 2:
            return False
        
        # Look for alternating pattern: 3to2, 2to3, 3to2, 2to3, ...
        cycle_pattern = ["3to2", "2to3"] * self.cycle_count
        
        # Check if the sequence contains the cycling pattern
        for i in range(len(sequence) - len(cycle_pattern) + 1):
            if sequence[i:i + len(cycle_pattern)] == cycle_pattern:
                return True
        
        return False
    
    def route_scoring(self, x):
        """Score the route based on presence of cycling"""
        if x < 0:
            return 0  # No cycling detected
        else:
            return 10  # Heavy penalty for cycling
