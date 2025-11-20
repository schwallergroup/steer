"""Generated evaluation code for: Late quinazoline ring formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateQuinazolineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage quinazoline ring formation via cyclization.
    Checks for the formation of benzouracil core (quinazoline with oxo group) through
    intramolecular cyclization, typically from anthranilate derivatives with cyanic acid.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.formation_method = config["parameters"]["formation_method"]  # "cyclization"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # SMARTS pattern for anthranilate-like precursor (ortho-amino benzoic acid derivative)
        self.precursor_pattern = Chem.MolFromSmarts("c1ccc(N)c(C(=O)[OH,O-,OC])c1")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Late-stage formation is better, so higher depth fractions get higher scores
            return 10 * x  # Scale depth fraction to 0-10
        else:
            # Early-stage formation preference
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step forms the quinazoline ring via cyclization.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains the quinazoline ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant contains the quinazoline ring (if so, it's not formation)
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False
            
            # Check for cyclization pattern: look for anthranilate-like precursor
            has_precursor = any(reactant.HasSubstructMatch(self.precursor_pattern) for reactant in reactants)
            
            # Additional check for intramolecular cyclization:
            # The precursor should have both the aniline and carboxyl groups that form the ring
            if has_precursor:
                # Verify this is indeed a cyclization by checking atom count changes
                reactant_heavy_atoms = sum(r.GetNumHeavyAtoms() for r in reactants)
                product_heavy_atoms = product.GetNumHeavyAtoms()
                
                # In cyclization with cyanic acid addition, we expect minimal atom count change
                # or slight increase (adding C, N, O from cyanic acid)
                atom_diff = product_heavy_atoms - reactant_heavy_atoms
                if -2 <= atom_diff <= 4:  # Allow for water loss and small additions
                    return True
            
            return False
            
        except Exception:
            return False
