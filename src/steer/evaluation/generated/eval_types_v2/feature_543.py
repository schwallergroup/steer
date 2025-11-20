"""Generated evaluation code for: Late stage pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation of specified ring systems.
    Detects when a target ring structure is formed (appears in product but not in reactants)
    and scores based on how late in the synthesis this formation occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score (closer to 1)
        else:  # early timing
            return x  # Earlier formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents ring formation/breaking of the target ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            if product is None:
                return False
                
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                r_mol = Chem.MolFromSmiles(r_smiles)
                if r_mol is not None:
                    reactant_mols.append(r_mol)
            
            if not reactant_mols:
                return False
            
            # Check for ring pattern in product and reactants
            product_has_ring = product.HasSubstructMatch(self.ring_pattern)
            reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
            
            if self.direction == "formation":
                # Ring formation: product has ring, but reactants don't (or fewer instances)
                if product_has_ring and not reactants_have_ring:
                    return True
                elif product_has_ring and reactants_have_ring:
                    # Check if number of ring instances increased
                    product_matches = len(product.GetSubstructMatches(self.ring_pattern))
                    reactant_matches = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactant_mols)
                    return product_matches > reactant_matches
            else:  # direction == "break"
                # Ring breaking: reactants have ring, but product doesn't (or fewer instances)
                if reactants_have_ring and not product_has_ring:
                    return True
                elif reactants_have_ring and product_has_ring:
                    # Check if number of ring instances decreased
                    product_matches = len(product.GetSubstructMatches(self.ring_pattern))
                    reactant_matches = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactant_mols)
                    return reactant_matches > product_matches
                    
        except Exception:
            return False
            
        return False
