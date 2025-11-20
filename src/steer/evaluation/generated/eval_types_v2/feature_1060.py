"""Generated evaluation code for: Early ester hydrolysis before ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyEsterHydrolysis(BaseScoring):
    """
    Evaluates routes for early ester hydrolysis before ether formation.
    
    Checks if an ester bond (C(=O)OC) is broken early in the synthesis route,
    which could complicate subsequent ether formation reactions.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ester_pattern = Chem.MolFromSmarts(self.bond_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For early timing, lower depth (earlier occurrence) gives higher score.
        """
        if x < 0:
            return 0  # Ester hydrolysis doesn't occur
        
        if self.timing == "early":
            # Early hydrolysis is rewarded with higher scores
            return (1 - x) * 10
        else:
            # Late hydrolysis would be penalized
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step breaks an ester bond.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains ester pattern
            has_ester_in_product = product_mol.HasSubstructMatch(self.ester_pattern)
            
            # Check if any reactant lacks the ester pattern (indicating bond breaking)
            ester_in_reactants = any(mol.HasSubstructMatch(self.ester_pattern) for mol in reactant_mols)
            
            # Ester hydrolysis: ester present in product but broken in reactants
            if self.direction == "break":
                return has_ester_in_product and not ester_in_reactants
            else:
                # Formation case
                return not has_ester_in_product and ester_in_reactants
                
        except Exception:
            return False
