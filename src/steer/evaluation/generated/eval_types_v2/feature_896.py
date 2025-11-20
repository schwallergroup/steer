"""Generated evaluation code for: Sandmeyer reaction for aryl halide synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SandmeyerReaction(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Sandmeyer reactions.
    
    Detects aryl halide formation from anilines via diazotization followed by
    halide substitution, specifically looking for conversion of aniline substrates
    to aryl iodides.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sandmeyer reaction not found
        else:
            # Earlier Sandmeyer reactions are generally preferred
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Detects Sandmeyer reaction by checking for:
        1. Aniline substrate (aromatic amine)
        2. Aryl halide product (specifically iodide)
        3. Loss of amino group and gain of halogen
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for aniline pattern in reactants
            aniline_pattern = Chem.MolFromSmarts("c[NH2,NH3+]")  # Aromatic amine
            has_aniline = any(mol.HasSubstructMatch(aniline_pattern) for mol in reactants)
            
            if not has_aniline:
                return False
            
            # Check for aryl iodide pattern in products
            aryl_iodide_pattern = Chem.MolFromSmarts("cI")  # Aromatic carbon bonded to iodine
            has_aryl_iodide = any(mol.HasSubstructMatch(aryl_iodide_pattern) for mol in products)
            
            if not has_aryl_iodide:
                return False
            
            # Additional check: verify amino group is replaced by iodine
            # by checking atom mapping if available
            return self._verify_amino_to_iodine_conversion(reactants_smiles, products_smiles)
            
        except Exception:
            return False
    
    def _verify_amino_to_iodine_conversion(self, reactants_smiles: str, products_smiles: str) -> bool:
        """
        Verify that an amino group is specifically converted to an iodine atom
        by checking atom mappings if present, or structural comparison if not.
        """
        try:
            # Parse molecules with atom mapping
            reactant_mol = Chem.MolFromSmiles(reactants_smiles.split(".")[0])
            product_mol = Chem.MolFromSmiles(products_smiles.split(".")[0])
            
            if not reactant_mol or not product_mol:
                return True  # Default to true if parsing fails
            
            # Get atom mappings
            reactant_maps = {atom.GetAtomMapNum(): atom for atom in reactant_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            product_maps = {atom.GetAtomMapNum(): atom for atom in product_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            if not reactant_maps or not product_maps:
                return True  # No mapping available, assume correct
            
            # Check if any carbon that was bonded to nitrogen is now bonded to iodine
            for map_num in reactant_maps:
                if map_num in product_maps:
                    reactant_atom = reactant_maps[map_num]
                    product_atom = product_maps[map_num]
                    
                    # Check if this carbon was bonded to nitrogen in reactant
                    reactant_neighbors = [neighbor.GetSymbol() for neighbor in reactant_atom.GetNeighbors()]
                    product_neighbors = [neighbor.GetSymbol() for neighbor in product_atom.GetNeighbors()]
                    
                    if 'N' in reactant_neighbors and 'I' in product_neighbors:
                        return True
            
            return False
            
        except Exception:
            return True  # Default to true if verification fails
