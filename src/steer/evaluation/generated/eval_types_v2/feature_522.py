"""Generated evaluation code for: Sandmeyer halogenation for cross-coupling setup"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SandmeyerHalogenation(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Sandmeyer halogenation reactions
    that convert anilines to aryl halides for cross-coupling setup.
    
    Detects C-N bond breaking where an aniline (aryl-NH2) is converted to an aryl halide,
    typically for subsequent palladium-catalyzed cross-coupling reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sandmeyer reaction doesn't occur
        else:
            # Earlier Sandmeyer reaction is better for synthetic planning
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a Sandmeyer halogenation by detecting:
        1. C-N bond break where N is part of aniline (aryl-NH2)
        2. Formation of aryl halide (Br, Cl, I)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for aniline pattern in reactants and aryl halide in product
            aniline_pattern = Chem.MolFromSmarts("[cH1,c]1[cH1][cH1][cH1][cH1][cH1]1-[NH2]")
            aryl_halide_pattern = Chem.MolFromSmarts("[cH1,c]1[cH1][cH1][cH1][cH1][cH1]1-[Br,Cl,I]")
            
            if not aniline_pattern or not aryl_halide_pattern:
                return False
            
            # Check if any reactant contains aniline
            has_aniline = any(r.HasSubstructMatch(aniline_pattern) for r in reactants)
            
            # Check if product contains aryl halide
            has_aryl_halide = product.HasSubstructMatch(aryl_halide_pattern)
            
            # Additional check: ensure C-N bond is actually broken
            if has_aniline and has_aryl_halide:
                return self._verify_cn_bond_break(product, reactants)
                
        except Exception:
            return False
            
        return False
    
    def _verify_cn_bond_break(self, product, reactants) -> bool:
        """
        Verify that a C-N bond was actually broken by checking atom mapping.
        """
        try:
            # Get mapped atoms from product
            product_mapped_atoms = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms() 
                                  if atom.GetAtomMapNum() > 0}
            
            # Get mapped atoms from reactants
            reactant_mapped_atoms = {}
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        reactant_mapped_atoms[atom.GetAtomMapNum()] = atom
            
            # Look for carbon atoms that were bonded to nitrogen in reactants
            # but are now bonded to halogen in product
            for map_num, prod_atom in product_mapped_atoms.items():
                if prod_atom.GetSymbol() == 'C' and map_num in reactant_mapped_atoms:
                    react_atom = reactant_mapped_atoms[map_num]
                    
                    # Check if this carbon is now bonded to halogen in product
                    prod_neighbors = [n.GetSymbol() for n in prod_atom.GetNeighbors()]
                    if any(symbol in ['Br', 'Cl', 'I'] for symbol in prod_neighbors):
                        
                        # Check if it was bonded to nitrogen in reactant
                        react_neighbors = [n.GetSymbol() for n in react_atom.GetNeighbors()]
                        if 'N' in react_neighbors and not any(symbol in ['Br', 'Cl', 'I'] for symbol in react_neighbors):
                            return True
            
            return False
            
        except Exception:
            return False
