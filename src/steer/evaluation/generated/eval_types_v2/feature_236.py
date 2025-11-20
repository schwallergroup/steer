"""Generated evaluation code for: Late intramolecular piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperidineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage intramolecular piperidine ring formation.
    Checks if a piperidine ring (C1CCNCC1) is formed via intramolecular cyclization
    and penalizes early formation while rewarding late-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # C1CCNCC1
        self.timing = config["parameters"]["timing"]  # late
        self.formation_type = config["parameters"]["formation_type"]  # intramolecular_cyclization
        self.piperidine_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late timing: higher depth (closer to 1.0) gets better score.
        x = -1 means condition not met (no intramolecular piperidine formation).
        """
        if x < 0:
            return 0  # No intramolecular piperidine formation found
        
        if self.timing == "late":
            # Reward late-stage formation: depth closer to 1.0 gets higher score
            return x * 10
        else:
            # For early timing, invert the score
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves intramolecular piperidine ring formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Check if product has piperidine ring
            if not prod_mol.HasSubstructMatch(self.piperidine_pattern):
                return False
            
            # Check if none of the reactants have the piperidine ring (ring formation)
            reactants_have_piperidine = any(
                r.HasSubstructMatch(self.piperidine_pattern) for r in reactant_mols if r
            )
            
            if reactants_have_piperidine:
                return False  # Ring already exists, not formation
            
            # Check for intramolecular cyclization:
            # Should be single reactant -> single product for intramolecular
            if len(reactant_mols) != 1:
                return False  # Multiple reactants suggests intermolecular
                
            reactant_mol = reactant_mols[0]
            
            # Verify the reactant has the necessary atoms to form piperidine
            # (should have linear chain that can cyclize)
            reactant_atoms = reactant_mol.GetNumAtoms()
            product_atoms = prod_mol.GetNumAtoms()
            
            # For intramolecular cyclization, atom count should be the same
            # (just forming new bond, not adding atoms)
            return abs(reactant_atoms - product_atoms) <= 1  # Allow for small differences due to H handling
            
        except Exception:
            return False
