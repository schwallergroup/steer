"""Generated evaluation code for: Late stage dehalogenation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageDehalogenation(BaseScoring):
    """
    Evaluates whether a dehalogenation reaction (specifically C-Br bond breaking)
    occurs at the final step of the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.target_timing = config["parameters"]["timing"]  # "final_step"
        self.bond_type = config["parameters"]["bond_type"]  # "C-Br"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Dehalogenation doesn't happen
        elif self.target_timing == "final_step":
            # For final step, we want x to be close to 1 (late in the route)
            if x >= 0.9:  # Very late stage (final step)
                return 10
            elif x >= 0.7:  # Reasonably late stage
                return 5
            else:  # Too early
                return 1
        else:
            return 1 - x  # General preference for late-stage disconnection
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves C-Br dehalogenation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Check for C-Br bond breaking (Br present in reactants but not products)
            prod_atoms = {atom.GetAtomMapNum(): atom.GetSymbol() 
                         for atom in prod_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            # Look for bromine atoms that are in reactants but not in products
            for reactant in reactant_mols:
                react_atoms = {atom.GetAtomMapNum(): atom.GetSymbol() 
                              for atom in reactant.GetAtoms() if atom.GetAtomMapNum() > 0}
                
                # Find Br atoms in reactant
                for map_num, symbol in react_atoms.items():
                    if symbol == 'Br':
                        # Check if this Br is connected to carbon in reactant
                        br_atom = None
                        for atom in reactant.GetAtoms():
                            if atom.GetAtomMapNum() == map_num:
                                br_atom = atom
                                break
                        
                        if br_atom:
                            # Check if Br is bonded to carbon
                            for neighbor in br_atom.GetNeighbors():
                                if neighbor.GetSymbol() == 'C':
                                    # Check if this Br is absent in products
                                    if map_num not in prod_atoms or prod_atoms[map_num] != 'Br':
                                        return True
            
            return False
            
        except Exception:
            return False
