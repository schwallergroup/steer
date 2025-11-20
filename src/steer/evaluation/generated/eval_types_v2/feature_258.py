"""Generated evaluation code for: Late coumarin ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateCoumarinRingFormation(BaseScoring):
    """
    Evaluates routes based on late-stage coumarin ring formation.
    
    Checks if a coumarin ring (c1ccc2cc(=O)oc2c1) is formed late in the synthesis,
    typically through condensation between ortho-hydroxy acetophenone and aromatic ester.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.coumarin_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Late formation is better - higher depth gives better score
            return 10 * x  # x is depth fraction (0-1)
        elif self.timing == "early":
            # Early formation is better - lower depth gives better score
            return 10 * (1 - x)
        else:
            # Just presence of formation
            return 10
    
    def hit_condition(self, d):
        """
        Check if coumarin ring formation occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            # Product (left side) and reactants (right side)
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if Chem.MolFromSmiles(r)]
            
            if not product_mol or not reactant_mols:
                return False
            
            # Check if product has coumarin ring
            product_has_coumarin = product_mol.HasSubstructMatch(self.coumarin_pattern)
            
            # Check if any reactant has coumarin ring
            reactants_have_coumarin = any(mol.HasSubstructMatch(self.coumarin_pattern) for mol in reactant_mols)
            
            if self.direction == "formation":
                # Ring formation: product has coumarin but reactants don't
                return product_has_coumarin and not reactants_have_coumarin
            elif self.direction == "breaking":
                # Ring breaking: reactants have coumarin but product doesn't
                return not product_has_coumarin and reactants_have_coumarin
            else:
                # Just presence in reaction
                return product_has_coumarin or reactants_have_coumarin
                
        except Exception:
            return False
