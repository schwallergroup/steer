"""Generated evaluation code for: Chemoselective bromo-chloro linker strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChemoselectiveBromoChloroLinker(BaseScoring):
    """
    Detects chemoselective bromo-chloro linker strategy where C-Br bond is broken
    preferentially over C-Cl bond due to differential leaving group ability.
    """
    
    def __init__(self, config: Dict):
        self.linker_pattern = config.get("linker_pattern", "BrCCCCl")
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not used
        else:
            # Earlier use of chemoselective strategy is better
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction involves chemoselective C-Br break while preserving C-Cl
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            prod = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not prod or not all(reactants):
                return False
            
            # Check if product has the bromo-chloro linker pattern
            linker_mol = Chem.MolFromSmarts(self._get_linker_smarts())
            if not prod.HasSubstructMatch(linker_mol):
                return False
            
            # Check if any reactant has C-Br broken but C-Cl preserved
            return self._detect_chemoselective_break(prod, reactants)
            
        except Exception:
            return False
    
    def _get_linker_smarts(self) -> str:
        """Convert linker pattern to SMARTS for substructure matching"""
        # Pattern for bromo-chloro alkyl linker
        return "[Br][CH2][CH2][CH2][Cl]"
    
    def _detect_chemoselective_break(self, product, reactants) -> bool:
        """
        Detects if C-Br bond was broken while C-Cl bond remains intact
        """
        # Get atoms with map numbers in product
        prod_br_maps = set()
        prod_cl_maps = set()
        
        for atom in product.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                if atom.GetSymbol() == 'Br':
                    prod_br_maps.add(map_num)
                elif atom.GetSymbol() == 'Cl':
                    prod_cl_maps.add(map_num)
        
        # Check reactants for broken C-Br bonds
        for reactant in reactants:
            react_br_maps = set()
            react_cl_maps = set()
            react_c_maps = set()
            
            for atom in reactant.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    if atom.GetSymbol() == 'Br':
                        react_br_maps.add(map_num)
                    elif atom.GetSymbol() == 'Cl':
                        react_cl_maps.add(map_num)
                    elif atom.GetSymbol() == 'C':
                        react_c_maps.add(map_num)
            
            # Check if Br from linker is in reactant but not bonded to original carbon
            if react_br_maps and not (react_br_maps & prod_br_maps):
                # Br was cleaved - check if corresponding C-Cl bond is preserved
                if react_cl_maps & prod_cl_maps:
                    # Cl remains in same position - chemoselective break detected
                    return self._verify_linker_connectivity(reactant, react_br_maps, react_cl_maps)
        
        return False
    
    def _verify_linker_connectivity(self, mol, br_maps, cl_maps) -> bool:
        """
        Verify the molecule contains the expected linker connectivity pattern
        """
        if not br_maps or not cl_maps:
            return False
        
        # Simple check for alkyl chain connectivity between Br and Cl
        chain_pattern = Chem.MolFromSmarts("[Br][CH2][CH2][CH2][Cl]")
        return mol.HasSubstructMatch(chain_pattern)
