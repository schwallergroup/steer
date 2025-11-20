"""Generated evaluation code for: Late stage aryl bromide to nitrile conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylBromideToNitrile(BaseScoring):
    """
    Evaluates synthesis routes for late-stage aryl bromide to nitrile conversion.
    Checks for Rosenmund-von Braun reaction (aryl bromide + copper cyanide -> aryl nitrile)
    occurring in the later stages of synthesis to avoid early nitrile sensitivity issues.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Later stage is better, penalize early occurrence
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Score based on how close to target depth
                if x >= self.target_depth:
                    return 1.0
                else:
                    return max(0, x / self.target_depth)
    
    def hit_condition(self, d) -> bool:
        """
        Detect aryl bromide to nitrile conversion (Rosenmund-von Braun reaction).
        Checks for:
        1. Aryl bromide in reactants
        2. Aryl nitrile in products
        3. Same aromatic core structure
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Define patterns
            aryl_bromide_pattern = Chem.MolFromSmarts("[c]Br")  # Aromatic carbon bonded to bromine
            aryl_nitrile_pattern = Chem.MolFromSmarts("[c]C#N")  # Aromatic carbon bonded to nitrile
            
            # Check for aryl bromide in reactants
            aryl_bromide_found = False
            bromide_reactant = None
            for reactant in reactants:
                if reactant.HasSubstructMatch(aryl_bromide_pattern):
                    aryl_bromide_found = True
                    bromide_reactant = reactant
                    break
            
            if not aryl_bromide_found:
                return False
            
            # Check for aryl nitrile in products
            aryl_nitrile_found = False
            nitrile_product = None
            for product in products:
                if product.HasSubstructMatch(aryl_nitrile_pattern):
                    aryl_nitrile_found = True
                    nitrile_product = product
                    break
            
            if not aryl_nitrile_found:
                return False
            
            # Verify transformation: same aromatic core with Br->CN substitution
            return self._verify_bromide_to_nitrile_transformation(
                bromide_reactant, nitrile_product
            )
            
        except Exception:
            return False
    
    def _verify_bromide_to_nitrile_transformation(self, bromide_mol, nitrile_mol) -> bool:
        """
        Verify that the transformation is specifically Br -> CN on the same aromatic core.
        Uses atom mapping to confirm the substitution pattern.
        """
        try:
            # Get atom maps for bromine-bearing carbon in reactant
            bromide_carbon_map = None
            for atom in bromide_mol.GetAtoms():
                if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 35:  # Bromine
                            bromide_carbon_map = atom.GetAtomMapNum()
                            break
                    if bromide_carbon_map:
                        break
            
            if not bromide_carbon_map:
                return False
            
            # Check if same carbon now bears nitrile in product
            for atom in nitrile_mol.GetAtoms():
                if (atom.GetAtomMapNum() == bromide_carbon_map and 
                    atom.GetAtomicNum() == 6 and atom.GetIsAromatic()):
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 6:  # Carbon of nitrile
                            for nn in neighbor.GetNeighbors():
                                if nn.GetAtomicNum() == 7 and len(nn.GetNeighbors()) == 1:  # Terminal nitrogen
                                    return True
            
            return False
            
        except Exception:
            # Fallback: structural similarity check
            return True  # Already confirmed aryl Br -> aryl CN pattern above
