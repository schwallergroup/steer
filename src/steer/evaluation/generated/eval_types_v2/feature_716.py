"""Generated evaluation code for: Early stage diketone construction strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DiketoneConstructionStrategy(BaseScoring):
    """
    Evaluates synthesis routes for early-stage diketone construction via malonate condensation.
    Checks if diketone scaffolds are assembled early in the synthesis through C-C bond formation
    involving malonate or similar activated methylene chemistry.
    """
    
    def __init__(self, config: Dict):
        self.target_timing = config.get("timing", "early")  # early, middle, late
        self.construction_method = config.get("construction_method", "malonate_condensation")
        
        # SMARTS patterns for diketone motifs
        self.diketone_patterns = [
            "[#6](=[#8])-[#6]-[#6](=[#8])",  # 1,3-diketone
            "[#6](=[#8])-[#6]-[#6]-[#6](=[#8])",  # 1,4-diketone
            "[#6](=[#8])-[#6]=[#6]-[#6](=[#8])"  # α,β-unsaturated diketone
        ]
        
        # SMARTS patterns for malonate and related activated methylene compounds
        self.malonate_patterns = [
            "[#6](=[#8])-[#8]-[#6](-[#6](=[#8])-[#8])",  # diethyl malonate core
            "[#6](=[#8])-[#6](-[#6](=[#8]))",  # β-diketone/malonate anion equivalent
            "[#6](-[#6](=[#8])-[#8])-[#6](=[#8])-[#8]"  # malonate ester
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Diketone construction doesn't occur
        
        if self.target_timing == "early":
            # Reward early construction (lower depth fraction)
            return (1 - x) * 10
        elif self.target_timing == "middle":
            # Optimal around 0.5 depth
            return (1 - abs(x - 0.5)) * 10
        else:  # late
            # Reward late construction (higher depth fraction)
            return x * 10

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents diketone construction via malonate condensation
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            
            if not product or not reactants:
                return False
            
            # Check if product contains diketone motif
            has_diketone = any(product.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                             for pattern in self.diketone_patterns)
            
            if not has_diketone:
                return False
            
            # Check if reaction involves malonate or activated methylene chemistry
            has_malonate_reactant = any(
                any(reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                    for pattern in self.malonate_patterns)
                for reactant in reactants if reactant
            )
            
            # Additional check for C-C bond formation pattern
            # Look for carbon atoms that are connected in product but not in reactants
            if has_malonate_reactant:
                return self._check_cc_bond_formation(reactants, product)
            
            return False
            
        except Exception:
            return False

    def _check_cc_bond_formation(self, reactants, product) -> bool:
        """
        Verify that C-C bond formation occurred between malonate-type carbon
        and electrophilic carbon to form diketone scaffold
        """
        try:
            # Look for mapped atoms to track bond formation
            product_atoms = {atom.GetAtomMapNum(): atom.GetIdx() 
                           for atom in product.GetAtoms() if atom.GetAtomMapNum()}
            
            reactant_atom_maps = set()
            for reactant in reactants:
                if reactant:
                    reactant_atom_maps.update(atom.GetAtomMapNum() 
                                            for atom in reactant.GetAtoms() 
                                            if atom.GetAtomMapNum())
            
            # Check for new C-C bonds in product involving mapped atoms
            for bond in product.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                
                if (begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'C' and
                    begin_atom.GetAtomMapNum() in reactant_atom_maps and
                    end_atom.GetAtomMapNum() in reactant_atom_maps):
                    
                    # Check if this bond wasn't present in reactants
                    bond_in_reactants = any(
                        self._bond_exists_in_mol(reactant, begin_atom.GetAtomMapNum(), 
                                               end_atom.GetAtomMapNum())
                        for reactant in reactants if reactant
                    )
                    
                    if not bond_in_reactants:
                        return True
            
            return False
            
        except Exception:
            return False

    def _bond_exists_in_mol(self, mol, map_num1, map_num2) -> bool:
        """Check if bond exists between atoms with given map numbers in molecule"""
        try:
            atom_idx_map = {atom.GetAtomMapNum(): atom.GetIdx() 
                          for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
            
            if map_num1 not in atom_idx_map or map_num2 not in atom_idx_map:
                return False
                
            idx1, idx2 = atom_idx_map[map_num1], atom_idx_map[map_num2]
            return mol.GetBondBetweenAtoms(idx1, idx2) is not None
            
        except Exception:
            return False
