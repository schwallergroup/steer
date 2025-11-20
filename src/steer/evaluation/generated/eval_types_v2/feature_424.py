"""Generated evaluation code for: Late stage benzylic alcohol chlorination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAlcoholChlorination(BaseScoring):
    """
    Evaluates routes for late-stage benzylic alcohol to chloride conversion.
    Checks if a benzylic alcohol is converted to a chloride, with preference
    for this occurring in later stages of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.benzylic_alcohol_pattern = "[CH2:1][OH:2].[cH]"  # Benzylic alcohol pattern
        self.chloride_pattern = "[CH2:1][Cl:2]"  # Corresponding chloride pattern
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        else:
            return 1 - x  # Later stage is better, so higher score for smaller depth fraction
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents benzylic alcohol chlorination
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants, products = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mol = Chem.MolFromSmiles(products.strip())
            
            if not all(reactant_mols) or not product_mol:
                return False
            
            # Check if any reactant has benzylic alcohol pattern
            benzylic_alcohol_smarts = Chem.MolFromSmarts("C[OH]c1ccccc1")  # Benzylic alcohol
            has_benzylic_alcohol = any(
                mol.HasSubstructMatch(benzylic_alcohol_smarts) for mol in reactant_mols
            )
            
            if not has_benzylic_alcohol:
                return False
            
            # Check if product has corresponding benzylic chloride
            benzylic_chloride_smarts = Chem.MolFromSmarts("C[Cl]c1ccccc1")  # Benzylic chloride
            has_benzylic_chloride = product_mol.HasSubstructMatch(benzylic_chloride_smarts)
            
            if not has_benzylic_chloride:
                return False
            
            # Verify transformation: alcohol -> chloride using atom mapping
            return self._verify_alcohol_to_chloride_transformation(rxn_smiles)
            
        except Exception:
            return False
    
    def _verify_alcohol_to_chloride_transformation(self, rxn_smiles: str) -> bool:
        """
        Verify that OH group is specifically converted to Cl using atom mapping
        """
        try:
            reactants, products = rxn_smiles.split(">>")
            
            # Parse molecules with atom mapping
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mol = Chem.MolFromSmiles(products.strip())
            
            # Find atoms with mapping numbers in reactants and products
            reactant_map_to_atom = {}
            for mol in reactant_mols:
                for atom in mol.GetAtoms():
                    map_num = atom.GetAtomMapNum()
                    if map_num > 0:
                        reactant_map_to_atom[map_num] = atom.GetSymbol()
            
            product_map_to_atom = {}
            for atom in product_mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    product_map_to_atom[map_num] = atom.GetSymbol()
            
            # Look for O -> Cl transformation
            for map_num in reactant_map_to_atom:
                if (reactant_map_to_atom[map_num] == 'O' and 
                    map_num in product_map_to_atom and 
                    product_map_to_atom[map_num] == 'Cl'):
                    
                    # Verify the carbon this is attached to is benzylic
                    return self._is_benzylic_position(reactant_mols, product_mol, map_num)
            
            return False
            
        except Exception:
            return False
    
    def _is_benzylic_position(self, reactant_mols, product_mol, oxygen_map_num) -> bool:
        """
        Check if the carbon attached to the OH/Cl is benzylic (adjacent to aromatic ring)
        """
        try:
            # Find the carbon connected to the oxygen in reactants
            for mol in reactant_mols:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() == oxygen_map_num:
                        # This is the oxygen, find its carbon neighbor
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetSymbol() == 'C':
                                # Check if this carbon is connected to an aromatic carbon
                                for carbon_neighbor in neighbor.GetNeighbors():
                                    if carbon_neighbor.GetIsAromatic():
                                        return True
            return False
            
        except Exception:
            return False
