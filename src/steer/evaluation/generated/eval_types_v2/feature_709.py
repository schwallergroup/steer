"""Generated evaluation code for: Late stage diaminopyrimidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DiaminopyrimidineFormation(BaseScoring):
    """
    Evaluates synthesis routes based on when diaminopyrimidine ring formation occurs.
    Rewards late-stage formation of the diaminopyrimidine core structure.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        
        # Create RDKit pattern for diaminopyrimidine
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if self.ring_pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {self.ring_smarts}")
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        For late-stage formation: higher depth (closer to 1) gives better score.
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Late stage preferred: score increases with depth
            return 10 * x  # Linear scaling from 0-10
        elif self.timing == "early":
            # Early stage preferred: score decreases with depth  
            return 10 * (1 - x)
        else:
            # Default to late timing
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node involves diaminopyrimidine ring formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
                
            # Check if product contains diaminopyrimidine ring
            product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_ring:
                return False
                
            # Check if any reactant lacks the diaminopyrimidine ring (formation)
            if self.direction == "formation":
                reactant_mols = []
                for r_smiles in reactants_smiles.split("."):
                    r_mol = Chem.MolFromSmiles(r_smiles)
                    if r_mol is not None:
                        reactant_mols.append(r_mol)
                
                # Ring formation: product has ring but at least one reactant doesn't
                reactants_have_ring = [mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols]
                return not all(reactants_have_ring)
                
            elif self.direction == "breaking":
                # Ring breaking: check if reactants lack the ring
                reactant_mols = []
                for r_smiles in reactants_smiles.split("."):
                    r_mol = Chem.MolFromSmiles(r_smiles)
                    if r_mol is not None:
                        reactant_mols.append(r_mol)
                
                reactants_have_ring = [mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols]
                return not any(reactants_have_ring)
                
        except Exception:
            return False
            
        return False
