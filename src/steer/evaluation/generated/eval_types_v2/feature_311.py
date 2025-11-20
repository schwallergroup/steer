"""Generated evaluation code for: Sandmeyer reaction for aryl bromide formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SandmeyerReaction(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Sandmeyer reaction converting
    aromatic amine to aryl bromide via diazonium intermediate.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth) / 10)
    
    def hit_condition(self, d):
        """
        Check if a reaction represents a Sandmeyer reaction:
        - Reactant contains aromatic amine (aniline-like)
        - Product contains aryl bromide
        - Conversion from C-N to C-Br on aromatic ring
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [r for r in reactants if r is not None]
            products = [p for p in products if p is not None]
            
            if not reactants or not products:
                return False
            
            # Check for aromatic amine pattern in reactants
            aromatic_amine_pattern = Chem.MolFromSmarts("[c:1][NH2]")
            has_aromatic_amine = any(r.HasSubstructMatch(aromatic_amine_pattern) for r in reactants)
            
            # Check for aryl bromide pattern in products
            aryl_bromide_pattern = Chem.MolFromSmarts("[c:1][Br]")
            has_aryl_bromide = any(p.HasSubstructMatch(aryl_bromide_pattern) for p in products)
            
            if not (has_aromatic_amine and has_aryl_bromide):
                return False
            
            # Check for atom mapping to confirm C-N to C-Br transformation
            for reactant in reactants:
                if reactant.HasSubstructMatch(aromatic_amine_pattern):
                    matches = reactant.GetSubstructMatches(aromatic_amine_pattern)
                    for match in matches:
                        carbon_map = reactant.GetAtomWithIdx(match[0]).GetAtomMapNum()
                        if carbon_map > 0:
                            # Check if this mapped carbon has bromine in products
                            for product in products:
                                for atom in product.GetAtoms():
                                    if (atom.GetAtomMapNum() == carbon_map and 
                                        atom.GetSymbol() == 'C'):
                                        # Check if this carbon is bonded to bromine
                                        for neighbor in atom.GetNeighbors():
                                            if neighbor.GetSymbol() == 'Br':
                                                return True
            
            return False
            
        except Exception:
            return False
