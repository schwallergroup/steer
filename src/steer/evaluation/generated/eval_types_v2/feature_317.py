"""Generated evaluation code for: Late stage C-S bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageBondFormation(BaseScoring):
    """
    Evaluates late-stage formation of specific bonds in synthesis routes.
    Detects when a target bond type is formed and rewards later formation.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.bond_pattern = Chem.MolFromSmarts(self.bond_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Bond formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (closer to 1.0)
            else:  # early
                return x  # Earlier formation is better
                
    def hit_condition(self, d) -> bool:
        """Check if the target bond is formed in this reaction step."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if product contains the target bond
            product_has_bond = product_mol.HasSubstructMatch(self.bond_pattern)
            
            if self.direction == "formation":
                # For bond formation: product should have bond, reactants should not all have it
                if not product_has_bond:
                    return False
                    
                # Check if any reactant already has the complete bond pattern
                reactants_have_bond = any(mol.HasSubstructMatch(self.bond_pattern) for mol in reactant_mols)
                
                # True if bond is formed (present in product but not in reactants)
                return not reactants_have_bond
                
            else:  # direction == "breaking"
                # For bond breaking: reactants should have bond, product should not
                reactants_have_bond = any(mol.HasSubstructMatch(self.bond_pattern) for mol in reactant_mols)
                return reactants_have_bond and not product_has_bond
                
        except (KeyError, ValueError, AttributeError):
            return False
