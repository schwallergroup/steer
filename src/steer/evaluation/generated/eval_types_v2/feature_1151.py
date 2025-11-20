"""Generated evaluation code for: Halodesilylation for aryl bromide installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class HalodesilylationDepth(BaseScoring):
    """
    Evaluates synthesis routes for the presence of halodesilylation reactions,
    specifically targeting the conversion of aryl silanes to aryl halides.
    Detects ipso-halodesilylation where a trimethylsilyl group is replaced with a halogen.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Halodesilylation doesn't occur
        else:
            if self.condition_type == "bool":
                return 1  # Reaction occurs
            else:
                # Earlier halodesilylation (lower depth) is typically better
                return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Detects halodesilylation by checking for:
        1. Loss of trimethylsilyl group from aromatic carbon
        2. Gain of halogen on the same aromatic carbon
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Define patterns for aryl silanes and aryl halides
            aryl_silane_pattern = Chem.MolFromSmarts("[c:1][Si](C)(C)C")  # Aromatic C bonded to trimethylsilyl
            aryl_halide_pattern = Chem.MolFromSmarts("[c:1][Cl,Br,I]")    # Aromatic C bonded to halogen
            
            if not aryl_silane_pattern or not aryl_halide_pattern:
                return False
            
            # Check for aryl silane in reactants
            reactant_silane_atoms = set()
            for reactant in reactants:
                matches = reactant.GetSubstructMatches(aryl_silane_pattern)
                for match in matches:
                    aryl_carbon_map = None
                    for atom_idx in match:
                        atom = reactant.GetAtomWithIdx(atom_idx)
                        if atom.GetAtomMapNum() > 0:
                            aryl_carbon_map = atom.GetAtomMapNum()
                            break
                    if aryl_carbon_map:
                        reactant_silane_atoms.add(aryl_carbon_map)
            
            # Check for corresponding aryl halide in products
            product_halide_atoms = set()
            for product in products:
                matches = product.GetSubstructMatches(aryl_halide_pattern)
                for match in matches:
                    aryl_carbon_map = None
                    for atom_idx in match:
                        atom = product.GetAtomWithIdx(atom_idx)
                        if atom.GetAtomMapNum() > 0:
                            aryl_carbon_map = atom.GetAtomMapNum()
                            break
                    if aryl_carbon_map:
                        product_halide_atoms.add(aryl_carbon_map)
            
            # Check if the same aromatic carbon has silyl in reactants and halogen in products
            return bool(reactant_silane_atoms.intersection(product_halide_atoms))
            
        except Exception:
            return False
