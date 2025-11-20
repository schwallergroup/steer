"""Generated evaluation code for: Early thiazolidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyThiazolidineFormation(BaseScoring):
    """
    Evaluates synthesis routes based on early thiazolidine ring formation.
    Checks for the formation of thiazolidine rings (C1CSCN1) and rewards
    routes where this occurs early in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CSCN1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For early timing, lower depth (earlier) gets higher score.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "early":
            # Early formation preferred - higher score for lower depth
            return 1 - x
        else:
            # Late formation preferred - higher score for higher depth
            return x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves thiazolidine ring formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) 
                           for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check for ring formation
            if self.direction == "formation":
                # Product should have thiazolidine ring
                product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
                
                # Reactants should not have the complete ring
                reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) 
                                        for mol in reactant_mols)
                
                # Ring formation occurs if product has ring but reactants don't
                return product_has_ring and not reactants_have_ring
            
            else:  # direction == "breaking"
                # Product should not have thiazolidine ring
                product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
                
                # At least one reactant should have the ring
                reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) 
                                        for mol in reactant_mols)
                
                # Ring breaking occurs if reactants have ring but product doesn't
                return not product_has_ring and reactants_have_ring
                
        except Exception:
            return False
