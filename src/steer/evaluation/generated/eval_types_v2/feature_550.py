"""Generated evaluation code for: Unstable carbamic acid intermediate strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class UnstableCarbamicAcidStrategy(MultiRxnCondBase):
    """
    Evaluates whether the route uses an unstable carbamic acid intermediate 
    that persists through multiple synthetic steps.
    """
    
    def __init__(self, config):
        self.steps_carried_through = config.get("steps_carried_through", 5)
        # SMARTS pattern for carbamic acid: N-C(=O)-OH
        self.carbamic_acid_pattern = "[NH,NH2]-C(=O)-[OH]"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track carbamic acid presence through the route
        carbamic_persistence = 0
        has_carbamic_acid = False
        
        for i, rxn in enumerate(reactions):
            if self.has_carbamic_acid_intermediate(rxn):
                has_carbamic_acid = True
                # Count consecutive steps where carbamic acid persists
                consecutive_steps = self.count_persistence_from_step(reactions, i)
                carbamic_persistence = max(carbamic_persistence, consecutive_steps)
        
        # Condition met if carbamic acid persists for target number of steps
        condition = has_carbamic_acid and carbamic_persistence >= self.steps_carried_through
        
        return condition, len(reactions)
    
    def has_carbamic_acid_intermediate(self, rxn):
        """Check if reaction involves carbamic acid intermediate"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if carbamic acid appears in reactants or products
        all_molecules = reactants + products
        
        for mol_smiles in all_molecules:
            try:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol is not None:
                    pattern = Chem.MolFromSmarts(self.carbamic_acid_pattern)
                    if pattern is not None and mol.HasSubstructMatch(pattern):
                        return True
            except:
                continue
                
        return False
    
    def count_persistence_from_step(self, reactions, start_idx):
        """Count how many consecutive steps the carbamic acid persists"""
        count = 1  # Count the initial step
        
        for i in range(start_idx + 1, len(reactions)):
            if self.has_carbamic_acid_intermediate(reactions[i]):
                count += 1
            else:
                break
                
        return count
    
    def route_scoring(self, x):
        """Score based on persistence length relative to target"""
        if x < 0:
            return 0  # No carbamic acid strategy detected
        
        # x represents the persistence length
        target = self.steps_carried_through
        
        if x >= target:
            return 10  # Perfect score if meets or exceeds target persistence
        else:
            # Partial score based on how close to target
            return (x / target) * 8  # Scale to 0-8 range for partial credit
