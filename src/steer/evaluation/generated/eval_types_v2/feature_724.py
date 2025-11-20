"""Generated evaluation code for: Late stage halogen exchange reaction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageHalogenExchange(BaseScoring):
    """
    Evaluates synthesis routes for late-stage halogen exchange reactions.
    Detects Finkelstein-type halogen exchange (e.g., Br to I conversion) and
    rewards routes where this occurs in the final steps.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        self.timing = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No halogen exchange found
        
        if self.timing == "late":
            # Reward later stage reactions more highly
            return 1 - x  # Early stages get lower scores
        else:
            # Standard depth-based scoring
            return 1 if x <= (self.depth_threshold / 10.0) else 0
    
    def hit_condition(self, d) -> bool:
        """
        Detects halogen exchange reactions by looking for:
        1. A halogen (F, Cl, Br, I) in the product
        2. A different halogen in the same position in reactants
        3. Mapped atoms to confirm the exchange
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            return self._detect_halogen_exchange(product, reactants)
            
        except:
            return False
    
    def _detect_halogen_exchange(self, product, reactants):
        """
        Detects halogen exchange by comparing mapped halogen atoms
        between product and reactants.
        """
        halogens = {'F', 'Cl', 'Br', 'I'}
        
        # Get halogen atoms with map numbers from product
        product_halogens = {}
        for atom in product.GetAtoms():
            if atom.GetSymbol() in halogens and atom.GetAtomMapNum() > 0:
                product_halogens[atom.GetAtomMapNum()] = atom.GetSymbol()
        
        if not product_halogens:
            return False
            
        # Check reactants for different halogens at same map positions
        for reactant in reactants:
            reactant_halogens = {}
            for atom in reactant.GetAtoms():
                if atom.GetSymbol() in halogens and atom.GetAtomMapNum() > 0:
                    reactant_halogens[atom.GetAtomMapNum()] = atom.GetSymbol()
            
            # Look for halogen exchange at mapped positions
            for map_num, prod_halogen in product_halogens.items():
                if map_num in reactant_halogens:
                    react_halogen = reactant_halogens[map_num]
                    if prod_halogen != react_halogen:
                        # Different halogens at same mapped position = exchange
                        return True
        
        return False
