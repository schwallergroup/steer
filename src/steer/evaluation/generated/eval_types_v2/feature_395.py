"""Generated evaluation code for: Late piperazinone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperazinoneFormation(BaseScoring):
    """
    Evaluates whether piperazinone ring formation occurs late in the synthesis route.
    
    Detects formation of piperazinone rings (6-membered rings containing two nitrogens
    and one ketone) and scores based on timing preference for late-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.formation_step = config["parameters"]["formation_step"]
        self.total_steps = config["parameters"]["total_steps"]
        self.timing = config["parameters"]["timing"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Prefer formation closer to the end (higher depth fraction)
            # Score increases as depth fraction approaches 1
            return x * 10
        else:
            # For early timing, prefer lower depth fractions
            return (1 - x) * 10
    
    def hit_condition(self, d):
        """
        Check if this reaction involves piperazinone ring formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            # Product (left side) and reactants (right side)
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains piperazinone ring
            product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_ring:
                return False
                
            # Check if any reactant already has the piperazinone ring
            reactant_mols = []
            for r_smiles in reactant_smiles.split("."):
                r_mol = Chem.MolFromSmiles(r_smiles)
                if r_mol:
                    reactant_mols.append(r_mol)
                    
            # If any reactant already has the ring, this is not ring formation
            for r_mol in reactant_mols:
                if r_mol.HasSubstructMatch(self.ring_pattern):
                    return False
                    
            # Product has ring but reactants don't - this is ring formation
            return True
            
        except Exception:
            return False
