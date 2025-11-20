"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when multiple fragments
    are coupled together at a specific depth in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_step = config.get("coupling_step", "final")  # "final" or "any"
        self.min_fragment_size = config.get("min_fragment_size", 5)  # minimum atoms per fragment
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling not found
        
        if self.coupling_step == "final":
            # Reward convergent coupling happening late in synthesis (closer to target)
            return (1 - x) * 10
        else:
            # For "any" coupling step, reward finding convergent coupling anywhere
            return 10 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """
        Detect if this reaction represents a convergent coupling step
        by checking if multiple significant fragments are being joined.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all([r is not None for r in reactants]) or product is None:
                return False
                
            # Filter out small molecules (reagents/catalysts)
            significant_reactants = [
                r for r in reactants 
                if r.GetNumHeavyAtoms() >= self.min_fragment_size
            ]
            
            # Check if we have the required number of significant fragments
            if len(significant_reactants) < self.fragment_count:
                return False
                
            # Verify that the fragments are actually being coupled
            # (not just mixed together without bond formation)
            return self._verify_coupling(significant_reactants, product)
            
        except Exception:
            return False
    
    def _verify_coupling(self, reactants, product) -> bool:
        """
        Verify that reactant fragments are actually coupled together
        by checking atom mapping and connectivity.
        """
        # Get mapped atoms from reactants
        reactant_maps = []
        for reactant in reactants:
            maps = set()
            for atom in reactant.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    maps.add(atom.GetAtomMapNum())
            if maps:
                reactant_maps.append(maps)
        
        if len(reactant_maps) < self.fragment_count:
            return False
            
        # Check if atoms from different reactants are bonded in product
        product_bonds = {}
        for bond in product.GetBonds():
            atom1_map = bond.GetBeginAtom().GetAtomMapNum()
            atom2_map = bond.GetEndAtom().GetAtomMapNum()
            if atom1_map > 0 and atom2_map > 0:
                product_bonds[(atom1_map, atom2_map)] = True
                product_bonds[(atom2_map, atom1_map)] = True
        
        # Look for bonds between atoms from different reactant fragments
        for i, reactant_map1 in enumerate(reactant_maps):
            for j, reactant_map2 in enumerate(reactant_maps[i+1:], i+1):
                # Check if any atom from reactant i is bonded to any atom from reactant j
                for atom1 in reactant_map1:
                    for atom2 in reactant_map2:
                        if (atom1, atom2) in product_bonds:
                            return True
        
        return False
