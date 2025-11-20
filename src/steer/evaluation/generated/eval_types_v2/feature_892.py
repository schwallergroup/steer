"""Generated evaluation code for: Early triazolone ring formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriazoloneFormationTiming(BaseScoring):
    """
    Evaluates the timing of triazolone ring formation via intramolecular cyclization.
    Favors early formation of the triazolone core structure in the synthesis route.
    """
    
    def __init__(self, config):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "[nH]1nc(=O)nn1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.formation_method = config["parameters"]["formation_method"]  # "intramolecular_cyclization"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x):
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Earlier formation is better (lower depth fraction = higher score)
            else:
                return x  # Later formation is better (higher depth fraction = higher score)
    
    def hit_condition(self, d):
        """Check if this reaction forms a triazolone ring via intramolecular cyclization"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains triazolone ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant already contains the triazolone ring
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already formed, this is not the formation step
            
            # Check if this is intramolecular cyclization (single reactant forms ring)
            if self.formation_method == "intramolecular_cyclization":
                # For intramolecular cyclization, we expect one main reactant
                # Filter out small molecules (catalysts, reagents)
                main_reactants = [r for r in reactants if r.GetNumAtoms() > 5]
                
                if len(main_reactants) == 1:
                    # Check if the reactant contains the atoms needed for triazolone formation
                    # Look for nitrogen atoms and carbonyl that could cyclize
                    reactant = main_reactants[0]
                    n_count = sum(1 for atom in reactant.GetAtoms() if atom.GetSymbol() == 'N')
                    has_carbonyl = reactant.HasSubstructMatch(Chem.MolFromSmarts("[C,c](=O)"))
                    
                    # Basic check: needs multiple nitrogens and carbonyl for triazolone formation
                    return n_count >= 3 and has_carbonyl
            
            return True
            
        except Exception:
            return False
