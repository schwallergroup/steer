"""Generated evaluation code for: Late stage halogen exchange bromide to iodide"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class HalogenExchange(BaseScoring):
    """
    Evaluates whether a halogen exchange reaction (bromide to iodide) occurs at the specified timing.
    Late-stage exchanges are preferred for synthetic efficiency.
    """
    
    def __init__(self, config: Dict):
        self.from_halogen = config["parameters"]["from_halogen"]
        self.to_halogen = config["parameters"]["to_halogen"]
        self.timing = config["parameters"]["timing"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Exchange doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better, return higher score for smaller depth fraction
        elif self.timing == "early":
            return x  # Earlier is better, return higher score for larger depth fraction
        else:
            return 1  # Any timing is acceptable
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a halogen exchange from Br to I"""
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
            
            # Check for halogen exchange pattern
            return self._detect_halogen_exchange(reactants, products)
            
        except Exception:
            return False
    
    def _detect_halogen_exchange(self, reactants, products) -> bool:
        """Detect if halogen exchange from Br to I occurred"""
        # Get atom map numbers for halogens in reactants and products
        reactant_br_maps = set()
        reactant_i_maps = set()
        product_br_maps = set()
        product_i_maps = set()
        
        # Collect halogen atom map numbers from reactants
        for mol in reactants:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "Br":
                    reactant_br_maps.add(atom.GetAtomMapNum())
                elif atom.GetSymbol() == "I":
                    reactant_i_maps.add(atom.GetAtomMapNum())
        
        # Collect halogen atom map numbers from products
        for mol in products:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "Br":
                    product_br_maps.add(atom.GetAtomMapNum())
                elif atom.GetSymbol() == "I":
                    product_i_maps.add(atom.GetAtomMapNum())
        
        # Check for Br to I exchange: atom that was Br in reactants is I in products
        if self.from_halogen == "Br" and self.to_halogen == "I":
            exchanged_atoms = reactant_br_maps.intersection(product_i_maps)
            return len(exchanged_atoms) > 0
        
        # Check for I to Br exchange: atom that was I in reactants is Br in products
        elif self.from_halogen == "I" and self.to_halogen == "Br":
            exchanged_atoms = reactant_i_maps.intersection(product_br_maps)
            return len(exchanged_atoms) > 0
        
        return False
