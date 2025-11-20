"""Generated evaluation code for: Late stage intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageIntramolecularCyclization(BaseScoring):
    """
    Evaluates routes based on when intramolecular cyclization occurs to form rings.
    Rewards late-stage ring formation reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better (depth closer to 1 gives higher score)
        else:  # early
            return x  # Earlier is better (depth closer to 0 gives higher score)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves intramolecular ring formation"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Skip if we can't parse molecules
            if not product or not all(reactants):
                return False
            
            # Check if product has the target ring pattern
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # For intramolecular cyclization, we expect:
            # 1. Only one main reactant (intramolecular)
            # 2. Ring is formed in this step (not present in reactant)
            main_reactant = max(reactants, key=lambda x: x.GetNumAtoms())
            
            # Check if main reactant lacks the ring pattern
            if main_reactant.HasSubstructMatch(self.ring_pattern):
                return False  # Ring already exists, not a formation reaction
            
            # Verify intramolecular nature by checking atom mapping
            # The atoms forming the new ring should be present in the same reactant
            product_atoms = set(atom.GetAtomMapNum() for atom in product.GetAtoms() 
                              if atom.GetAtomMapNum() > 0)
            main_reactant_atoms = set(atom.GetAtomMapNum() for atom in main_reactant.GetAtoms() 
                                    if atom.GetAtomMapNum() > 0)
            
            # Most product atoms should come from the main reactant (intramolecular)
            overlap_ratio = len(product_atoms.intersection(main_reactant_atoms)) / len(product_atoms)
            
            return overlap_ratio > 0.7  # At least 70% of atoms from same reactant
            
        except Exception:
            return False
