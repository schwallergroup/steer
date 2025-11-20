"""Generated evaluation code for: Convergent synthesis via two complex fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if the route assembles
    two separately synthesized complex fragments in a final coupling step.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_step_position = config.get("coupling_step_position", "final")
        self.min_fragment_complexity = config.get("min_fragment_complexity", 5)  # atoms
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Better score for earlier convergent coupling (but still penalize very early)
            if self.coupling_step_position == "final":
                return 10 * (1 - x) if x > 0.8 else 5 * (1 - x)
            else:
                return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of complex fragments
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles")
        
        if not rxn_smiles:
            return False
            
        try:
            # Parse reaction SMILES
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            # Filter out small molecules (reagents/catalysts)
            complex_reactants = []
            for r_smiles in reactants_smiles:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and mol.GetNumAtoms() >= self.min_fragment_complexity:
                    complex_reactants.append(mol)
            
            # Check if we have the right number of complex fragments
            if len(complex_reactants) != self.fragment_count:
                return False
                
            # Verify this is a coupling reaction (C-C, C-N, C-O bond formation)
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if fragments are being coupled together
            return self._is_coupling_reaction(product_mol, complex_reactants)
            
        except Exception:
            return False
    
    def _is_coupling_reaction(self, product, reactants) -> bool:
        """
        Check if the reaction represents coupling of fragments by looking for
        new bonds between atoms that were in different reactants
        """
        if len(reactants) != 2:
            return False
            
        # Get atom map numbers for each reactant
        reactant_maps = []
        for reactant in reactants:
            maps = set()
            for atom in reactant.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    maps.add(atom.GetAtomMapNum())
            reactant_maps.append(maps)
        
        if len(reactant_maps) != 2 or not all(reactant_maps):
            return False
            
        # Check product for bonds between atoms from different reactants
        for bond in product.GetBonds():
            atom1_map = bond.GetBeginAtom().GetAtomMapNum()
            atom2_map = bond.GetEndAtom().GetAtomMapNum()
            
            if atom1_map > 0 and atom2_map > 0:
                # Check if these atoms were in different reactants
                in_reactant1 = (atom1_map in reactant_maps[0], atom2_map in reactant_maps[0])
                in_reactant2 = (atom1_map in reactant_maps[1], atom2_map in reactant_maps[1])
                
                # If one atom from each reactant, this is a coupling bond
                if (in_reactant1[0] and in_reactant2[1]) or (in_reactant1[1] and in_reactant2[0]):
                    return True
                    
        return False
