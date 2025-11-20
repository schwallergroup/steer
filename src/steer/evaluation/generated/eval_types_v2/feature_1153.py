"""Generated evaluation code for: Regioselective bromination via TMS group placeholder"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TMSBrominationRegioselectivity(BaseScoring):
    """
    Evaluates regioselective bromination via TMS group placeholder strategy.
    Detects when a Si-C bond is broken and replaced with Br-C bond in the same position.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]  # "[Si]-c"
        self.replacement_atom = config["parameters"]["replacement_atom"]  # "Br"
        self.condition_type = config.get("condition_type", "depth")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not used
        else:
            return 1 - x  # Earlier use is better for regioselectivity planning
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves TMS group removal and bromination.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
            
            # Look for TMS pattern in products and Br in corresponding position in reactants
            tms_pattern = Chem.MolFromSmarts(self.bond_smarts)
            if not tms_pattern:
                return False
            
            for product in products:
                # Find TMS groups in product
                matches = product.GetSubstructMatches(tms_pattern)
                if matches:
                    # Check if any reactant has Br in place of the Si
                    for match in matches:
                        si_atom_map = None
                        c_atom_map = None
                        
                        # Get atom map numbers for the matched atoms
                        for atom_idx in match:
                            atom = product.GetAtomWithIdx(atom_idx)
                            if atom.GetSymbol() == "Si":
                                si_atom_map = atom.GetAtomMapNum()
                            elif atom.GetSymbol() == "C":
                                c_atom_map = atom.GetAtomMapNum()
                        
                        if si_atom_map and c_atom_map:
                            # Check if in reactants, the Si is gone and Br is bonded to the C
                            if self._check_tms_to_br_replacement(reactants, si_atom_map, c_atom_map):
                                return True
            
            return False
            
        except Exception:
            return False
    
    def _check_tms_to_br_replacement(self, reactants, si_atom_map, c_atom_map):
        """
        Check if Si atom is absent and Br is bonded to the carbon in reactants.
        """
        for reactant in reactants:
            # Check if this reactant has the carbon but not the silicon
            c_atom = None
            has_si = False
            
            for atom in reactant.GetAtoms():
                if atom.GetAtomMapNum() == c_atom_map:
                    c_atom = atom
                elif atom.GetAtomMapNum() == si_atom_map:
                    has_si = True
            
            # If we found the carbon but no silicon in this reactant
            if c_atom and not has_si:
                # Check if carbon is bonded to bromine
                for neighbor in c_atom.GetNeighbors():
                    if neighbor.GetSymbol() == self.replacement_atom:
                        return True
        
        return False
