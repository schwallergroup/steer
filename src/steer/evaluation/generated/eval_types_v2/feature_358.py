"""Generated evaluation code for: Halide activation via Finkelstein reaction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FinkelsteinReaction(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Finkelstein reactions.
    
    The Finkelstein reaction involves halide exchange, typically converting 
    bromide or chloride to iodide for electrophile activation. This improves
    coupling efficiency in subsequent reactions.
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
            return 1 - abs(x - self.target_depth) / 10  # Earlier is better for activation
    
    def hit_condition(self, d):
        """
        Detects Finkelstein reaction by checking for halide exchange patterns.
        Looks for conversion of Cl/Br to I or vice versa.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Get halogen counts for product and reactants
            prod_halogens = self._count_halogens(prod_mol)
            react_halogens = {"Cl": 0, "Br": 0, "I": 0}
            
            for mol in react_mols:
                mol_halogens = self._count_halogens(mol)
                for halogen in ["Cl", "Br", "I"]:
                    react_halogens[halogen] += mol_halogens[halogen]
            
            # Check for halide exchange (different halogen types, same total count)
            total_prod = sum(prod_halogens.values())
            total_react = sum(react_halogens.values())
            
            if total_prod != total_react or total_prod == 0:
                return False
            
            # Check if halogen types have changed (indicating exchange)
            halogen_change = False
            for halogen in ["Cl", "Br", "I"]:
                if prod_halogens[halogen] != react_halogens[halogen]:
                    halogen_change = True
                    break
            
            return halogen_change
            
        except Exception:
            return False
    
    def _count_halogens(self, mol):
        """Count halogen atoms in a molecule."""
        halogen_counts = {"Cl": 0, "Br": 0, "I": 0}
        
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in halogen_counts:
                halogen_counts[symbol] += 1
        
        return halogen_counts
