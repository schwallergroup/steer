"""Generated evaluation code for: Early stage SNAr bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySNArBondFormation(BaseScoring):
    """
    Evaluates early stage nucleophilic aromatic substitution (SNAr) bond formation.
    Checks for C-O bond formation on activated aromatic rings at early stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Early stage default
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr reaction doesn't happen
        else:
            # Early stage is better - lower depth fraction gives higher score
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                # Penalize reactions that occur too late
                if x <= self.target_depth:
                    return 1.0
                else:
                    return max(0, 1.0 - (x - self.target_depth) * 5)
    
    def hit_condition(self, d) -> bool:
        """
        Detects nucleophilic aromatic substitution reactions involving C-O bond formation.
        Looks for aromatic carbon-oxygen bond formation on activated rings.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        try:
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".") if r.strip()]
            
            if not product or not all(reactants):
                return False
            
            return self._detect_snar_co_formation(product, reactants)
            
        except Exception:
            return False
    
    def _detect_snar_co_formation(self, product, reactants) -> bool:
        """
        Detects if a C-O bond is formed via SNAr on an activated aromatic ring.
        """
        # Get all aromatic C-O bonds in product
        product_co_bonds = self._get_aromatic_co_bonds(product)
        
        if not product_co_bonds:
            return False
        
        # Check if any of these C-O bonds are newly formed
        for reactant in reactants:
            reactant_co_bonds = self._get_aromatic_co_bonds(reactant)
            
            # Look for C-O bonds present in product but not in this reactant
            for prod_bond in product_co_bonds:
                prod_c_map, prod_o_map = prod_bond
                
                # Check if this bond exists in reactant
                bond_exists_in_reactant = any(
                    (c_map == prod_c_map and o_map == prod_o_map) or 
                    (c_map == prod_o_map and o_map == prod_c_map)
                    for c_map, o_map in reactant_co_bonds
                )
                
                if not bond_exists_in_reactant:
                    # New C-O bond found, check if carbon is on activated aromatic ring
                    if self._is_activated_aromatic_carbon(product, prod_c_map):
                        return True
        
        return False
    
    def _get_aromatic_co_bonds(self, mol) -> List[Tuple[int, int]]:
        """
        Get all aromatic carbon-oxygen bonds with atom map numbers.
        """
        co_bonds = []
        
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Check for aromatic C-O bond
            if ((atom1.GetSymbol() == 'C' and atom1.GetIsAromatic() and atom2.GetSymbol() == 'O') or
                (atom2.GetSymbol() == 'C' and atom2.GetIsAromatic() and atom1.GetSymbol() == 'O')):
                
                map1 = atom1.GetAtomMapNum()
                map2 = atom2.GetAtomMapNum()
                
                if map1 > 0 and map2 > 0:
                    if atom1.GetSymbol() == 'C':
                        co_bonds.append((map1, map2))
                    else:
                        co_bonds.append((map2, map1))
        
        return co_bonds
    
    def _is_activated_aromatic_carbon(self, mol, carbon_map_num) -> bool:
        """
        Check if aromatic carbon is activated for nucleophilic substitution.
        Looks for electron-withdrawing groups like nitro, cyano, or heteroaromatics.
        """
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == carbon_map_num:
                # Check if carbon is in an electron-deficient aromatic system
                
                # Check for pyridine-like nitrogen in ring
                if self._has_heteroaromatic_activation(atom):
                    return True
                
                # Check for electron-withdrawing substituents
                if self._has_ewg_activation(atom):
                    return True
                
                break
        
        return False
    
    def _has_heteroaromatic_activation(self, carbon_atom) -> bool:
        """
        Check if carbon is in a heteroaromatic ring (e.g., pyridine).
        """
        mol = carbon_atom.GetOwningMol()
        
        # Check if any ring containing this carbon has aromatic nitrogen
        for ring in mol.GetRingInfo().AtomRings():
            if carbon_atom.GetIdx() in ring:
                for atom_idx in ring:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                        return True
        
        return False
    
    def _has_ewg_activation(self, carbon_atom) -> bool:
        """
        Check for electron-withdrawing groups that activate aromatic substitution.
        """
        mol = carbon_atom.GetOwningMol()
        
        # Define EWG patterns
        ewg_patterns = [
            "[N+](=O)[O-]",  # Nitro group
            "C#N",           # Cyano group
            "C(=O)",         # Carbonyl groups
            "S(=O)(=O)",     # Sulfonyl groups
        ]
        
        # Check for EWG patterns in the molecule
        for pattern in ewg_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        
        return False
