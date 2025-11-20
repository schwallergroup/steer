"""Generated evaluation code for: Late stage aryl ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylEtherFormation(BaseScoring):
    """
    Evaluates routes based on late-stage aryl ether formation.
    Detects when an aryl ether bond (Ar-O-R) is formed in a reaction,
    with preference for later stages in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Aryl ether formation doesn't occur
        else:
            # Late-stage formation is better (higher depth fraction)
            # Score increases as we get closer to target late-stage timing
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Penalize deviation from target depth, with preference for later
                if x >= self.target_depth:
                    return 10 - abs(x - self.target_depth) * 5
                else:
                    return max(0, 10 - (self.target_depth - x) * 10)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves aryl ether formation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            products = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not products or not all(reactants):
                return False
            
            # Find aryl ether patterns in products
            aryl_ether_patterns = [
                "[cH1,cH0:1]-[OH0:2]-[CH2,CH3,c:3]",  # Aromatic C-O-C
                "[c:1]-[OH0:2]-[!c:3]",               # Aromatic C-O-aliphatic
                "[c:1]-[OH0:2]-[c:3]"                 # Aromatic C-O-aromatic
            ]
            
            product_aryl_ethers = []
            for pattern in aryl_ether_patterns:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol and products.HasSubstructMatch(patt_mol):
                    matches = products.GetSubstructMatches(patt_mol)
                    for match in matches:
                        # Get atom map numbers for the C-O-C linkage
                        aryl_c_map = None
                        o_map = None
                        other_c_map = None
                        
                        for atom_idx in match:
                            atom = products.GetAtomWithIdx(atom_idx)
                            map_num = atom.GetAtomMapNum()
                            if map_num > 0:
                                if atom.GetSymbol() == 'C' and atom.GetIsAromatic():
                                    aryl_c_map = map_num
                                elif atom.GetSymbol() == 'O':
                                    o_map = map_num
                                elif atom.GetSymbol() == 'C':
                                    other_c_map = map_num
                        
                        if aryl_c_map and o_map and other_c_map:
                            product_aryl_ethers.append((aryl_c_map, o_map, other_c_map))
            
            if not product_aryl_ethers:
                return False
            
            # Check if this aryl ether linkage is formed in this step
            # by verifying the C-O bond doesn't exist in reactants
            for aryl_c_map, o_map, other_c_map in product_aryl_ethers:
                # Check if the C-O-C linkage exists in any reactant
                linkage_exists_in_reactants = False
                
                for reactant in reactants:
                    aryl_c_atom = None
                    o_atom = None
                    other_c_atom = None
                    
                    # Find atoms with matching map numbers
                    for atom in reactant.GetAtoms():
                        if atom.GetAtomMapNum() == aryl_c_map:
                            aryl_c_atom = atom
                        elif atom.GetAtomMapNum() == o_map:
                            o_atom = atom
                        elif atom.GetAtomMapNum() == other_c_map:
                            other_c_atom = atom
                    
                    # If all three atoms are in the same reactant, check connectivity
                    if aryl_c_atom and o_atom and other_c_atom:
                        # Check if C-O and O-C bonds exist
                        aryl_c_idx = aryl_c_atom.GetIdx()
                        o_idx = o_atom.GetIdx()
                        other_c_idx = other_c_atom.GetIdx()
                        
                        bond1 = reactant.GetBondBetweenAtoms(aryl_c_idx, o_idx)
                        bond2 = reactant.GetBondBetweenAtoms(o_idx, other_c_idx)
                        
                        if bond1 and bond2:
                            linkage_exists_in_reactants = True
                            break
                
                # If linkage doesn't exist in reactants but exists in products,
                # this is aryl ether formation
                if not linkage_exists_in_reactants:
                    return True
            
            return False
            
        except Exception:
            return False
