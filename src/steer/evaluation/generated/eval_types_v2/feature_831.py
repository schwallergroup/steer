"""Generated evaluation code for: Late stage ozonolysis for aldehyde formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageOzonolysis(BaseScoring):
    """
    Evaluates if ozonolysis reaction occurs at late stage for aldehyde formation.
    Checks for cleavage of C=C bonds to form aldehydes, rewarding later occurrence.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ozonolysis doesn't happen
        else:
            # Reward late-stage ozonolysis (higher depth fraction is better)
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Score increases as depth approaches target (late stage)
                return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Detect ozonolysis reaction by checking for:
        1. C=C bond breaking in reactant
        2. Formation of aldehyde (C=O) in products
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactant_smiles, product_smiles = rxn_smiles.split(">>")
            reactant = Chem.MolFromSmiles(reactant_smiles)
            products = [Chem.MolFromSmiles(p) for p in product_smiles.split(".")]
            
            if not reactant or not all(products):
                return False
            
            # Check for alkene pattern in reactant
            alkene_pattern = Chem.MolFromSmarts("[C:1]=[C:2]")
            if not reactant.HasSubstructMatch(alkene_pattern):
                return False
            
            # Check for aldehyde formation in products
            aldehyde_pattern = Chem.MolFromSmarts("[C:1]=[O:2]")
            aldehyde_formed = any(prod.HasSubstructMatch(aldehyde_pattern) for prod in products)
            
            if not aldehyde_formed:
                return False
            
            # Verify bond breaking: mapped atoms from C=C should be in different products
            matches = reactant.GetSubstructMatches(alkene_pattern)
            for match in matches:
                atom1_map = reactant.GetAtomWithIdx(match[0]).GetAtomMapNum()
                atom2_map = reactant.GetAtomWithIdx(match[1]).GetAtomMapNum()
                
                if atom1_map > 0 and atom2_map > 0:
                    # Check if these mapped atoms end up in different products
                    atom1_products = []
                    atom2_products = []
                    
                    for i, prod in enumerate(products):
                        if any(atom.GetAtomMapNum() == atom1_map for atom in prod.GetAtoms()):
                            atom1_products.append(i)
                        if any(atom.GetAtomMapNum() == atom2_map for atom in prod.GetAtoms()):
                            atom2_products.append(i)
                    
                    # If atoms are in different products, bond was broken
                    if atom1_products and atom2_products and not set(atom1_products).intersection(set(atom2_products)):
                        return True
            
            return False
            
        except Exception:
            return False
