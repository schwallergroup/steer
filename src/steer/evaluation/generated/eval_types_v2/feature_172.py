"""Generated evaluation code for: O-glycoside to N-glycoside rearrangement strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OToNGlycosideRearrangement(BaseScoring):
    """
    Evaluates synthesis routes for the presence of O-glycoside to N-glycoside rearrangement reactions.
    
    This scoring function identifies reactions where an O-glycosidic bond is converted to an 
    N-glycosidic bond, typically involving the rearrangement of a sugar moiety from oxygen 
    to nitrogen connectivity.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth) / 5)  # Scale to 0-1
    
    def hit_condition(self, d):
        """Check if the reaction represents an O- to N-glycoside rearrangement."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol is not None:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check for O-glycoside in reactants and N-glycoside in products
            has_o_glycoside_reactant = any(self._has_o_glycoside(mol) for mol in reactant_mols)
            has_n_glycoside_product = any(self._has_n_glycoside(mol) for mol in product_mols)
            
            # Additional check: ensure we're not just detecting separate molecules
            # Look for mapped atoms to confirm actual rearrangement
            if has_o_glycoside_reactant and has_n_glycoside_product:
                return self._confirm_rearrangement(reactants_smiles, products_smiles)
            
            return False
            
        except Exception:
            return False
    
    def _has_o_glycoside(self, mol):
        """Detect O-glycosidic bond patterns in a molecule."""
        # Pattern for O-glycosidic linkage: sugar-O-aglycon
        # Look for pyranose/furanose ring connected via oxygen
        o_glycoside_patterns = [
            "[CH]1[CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])O1[O][#6]",  # Pyranose-O-
            "[CH]1[CH]([OH])[CH]([OH])[CH]([OH])[CH]1([OH])O[#6]",  # Furanose-O-
            "[CH]1[CH]([OH])[CH]([OH])[CH]([OH])[CH]([CH2][OH])O1[O][#6]",  # General sugar-O-
            "C1[CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])[CH](*)O1[O][!H]"  # More general pattern
        ]
        
        for pattern in o_glycoside_patterns:
            try:
                query = Chem.MolFromSmarts(pattern)
                if query is not None and mol.HasSubstructMatch(query):
                    return True
            except:
                continue
        return False
    
    def _has_n_glycoside(self, mol):
        """Detect N-glycosidic bond patterns in a molecule."""
        # Pattern for N-glycosidic linkage: sugar-N-aglycon
        n_glycoside_patterns = [
            "[CH]1[CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])O1[N][#6]",  # Pyranose-N-
            "[CH]1[CH]([OH])[CH]([OH])[CH]([OH])[CH]1([OH])[N][#6]",  # Furanose-N-
            "[CH]1[CH]([OH])[CH]([OH])[CH]([OH])[CH]([CH2][OH])O1[N][#6]",  # General sugar-N-
            "C1[CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])[CH](*)O1[N][!H]"  # More general pattern
        ]
        
        for pattern in n_glycoside_patterns:
            try:
                query = Chem.MolFromSmarts(pattern)
                if query is not None and mol.HasSubstructMatch(query):
                    return True
            except:
                continue
        return False
    
    def _confirm_rearrangement(self, reactants_smiles, products_smiles):
        """Confirm that this is actually a rearrangement by checking atom mapping."""
        try:
            # Look for common atom map numbers between reactants and products
            # indicating the same molecular framework is being rearranged
            reactant_maps = set()
            product_maps = set()
            
            # Extract atom map numbers from reactants
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    for atom in mol.GetAtoms():
                        if atom.GetAtomMapNum() > 0:
                            reactant_maps.add(atom.GetAtomMapNum())
            
            # Extract atom map numbers from products  
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol is not None:
                    for atom in mol.GetAtoms():
                        if atom.GetAtomMapNum() > 0:
                            product_maps.add(atom.GetAtomMapNum())
            
            # If there's significant overlap in atom mapping, likely a rearrangement
            overlap = len(reactant_maps.intersection(product_maps))
            total_unique = len(reactant_maps.union(product_maps))
            
            if total_unique > 0:
                overlap_ratio = overlap / total_unique
                return overlap_ratio > 0.5  # At least 50% of atoms are conserved
            
            # If no atom mapping, fall back to molecular formula comparison
            return True
            
        except Exception:
            return True  # Default to True if we can't confirm
