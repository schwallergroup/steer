"""Generated evaluation code for: Late stage N-arylation via cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNArylation(BaseScoring):
    """
    Evaluates whether N-arylation via cross-coupling (Chan-Lam, Buchwald-Hartwig, or general N-arylation) 
    occurs at a late stage in the synthesis route. Checks for formation of C-N bonds between aromatic 
    carbons and nitrogens.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # N-arylation doesn't happen
        else:
            # Later stage is better, so higher depth fraction gets higher score
            if self.condition_type == "bool":
                return 10 if x >= self.target_depth else 0
            else:
                # Score based on how close to target depth (late stage preferred)
                return max(0, 10 - abs(x - self.target_depth) * 10)
    
    def hit_condition(self, d):
        """Check if this reaction involves N-arylation cross-coupling"""
        metadata = d.get("metadata", {})
        
        # Check if reaction SMILES exists
        rxn_smiles = metadata.get("mapped_reaction_smiles")
        if not rxn_smiles:
            return False
            
        try:
            # Parse reaction SMILES
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check for C-N bond formation between aromatic carbon and nitrogen
            return self._detects_n_arylation(prod_mol, react_mols)
            
        except Exception:
            return False
    
    def _detects_n_arylation(self, product, reactants):
        """Detect if C-N bond formation occurs between aromatic C and N"""
        
        # Get all atom map numbers in product
        prod_atom_maps = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms() 
                         if atom.GetAtomMapNum() > 0}
        
        # Get all atom map numbers in reactants
        react_atom_maps = {}
        for react in reactants:
            for atom in react.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    react_atom_maps[atom.GetAtomMapNum()] = atom
        
        # Look for C-N bonds in product
        for bond in product.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            
            begin_map = begin_atom.GetAtomMapNum()
            end_map = end_atom.GetAtomMapNum()
            
            if begin_map == 0 or end_map == 0:
                continue
                
            # Check if this is a C-N bond where C is aromatic
            is_c_n_bond = False
            aromatic_carbon_map = None
            nitrogen_map = None
            
            if (begin_atom.GetSymbol() == 'C' and begin_atom.GetIsAromatic() and 
                end_atom.GetSymbol() == 'N'):
                is_c_n_bond = True
                aromatic_carbon_map = begin_map
                nitrogen_map = end_map
            elif (end_atom.GetSymbol() == 'C' and end_atom.GetIsAromatic() and 
                  begin_atom.GetSymbol() == 'N'):
                is_c_n_bond = True
                aromatic_carbon_map = end_map
                nitrogen_map = begin_map
            
            if not is_c_n_bond:
                continue
                
            # Check if this bond was formed (atoms were in different reactant molecules)
            if self._bond_was_formed(aromatic_carbon_map, nitrogen_map, reactants):
                # Additional check for typical N-arylation patterns
                if self._matches_n_arylation_pattern(product, reactants, aromatic_carbon_map, nitrogen_map):
                    return True
                    
        return False
    
    def _bond_was_formed(self, map1, map2, reactants):
        """Check if two mapped atoms were in different reactant molecules"""
        mol1_idx = None
        mol2_idx = None
        
        for i, react in enumerate(reactants):
            maps_in_mol = [atom.GetAtomMapNum() for atom in react.GetAtoms()]
            if map1 in maps_in_mol:
                mol1_idx = i
            if map2 in maps_in_mol:
                mol2_idx = i
                
        return mol1_idx is not None and mol2_idx is not None and mol1_idx != mol2_idx
    
    def _matches_n_arylation_pattern(self, product, reactants, carbon_map, nitrogen_map):
        """Check for typical N-arylation coupling patterns"""
        
        # Look for boronic acid/ester patterns (Chan-Lam, Suzuki-type N-arylation)
        boronic_patterns = [
            Chem.MolFromSmarts("[C:1][B](O)O"),  # Boronic acid
            Chem.MolFromSmarts("[C:1][B]1OC(C)(C)C(C)(C)O1"),  # Pinacol boronate
            Chem.MolFromSmarts("[C:1][B](OC)OC")  # Boronic ester
        ]
        
        # Look for haloarene patterns (Buchwald-Hartwig)
        haloarene_patterns = [
            Chem.MolFromSmarts("[c:1][Cl,Br,I]"),  # Aryl halides
        ]
        
        # Check reactants for these patterns
        has_coupling_partner = False
        for react in reactants:
            # Check for boronic acid derivatives
            for pattern in boronic_patterns:
                if react.HasSubstructMatch(pattern):
                    matches = react.GetSubstructMatches(pattern)
                    for match in matches:
                        if react.GetAtomWithIdx(match[0]).GetAtomMapNum() == carbon_map:
                            has_coupling_partner = True
                            break
            
            # Check for aryl halides
            for pattern in haloarene_patterns:
                if react.HasSubstructMatch(pattern):
                    matches = react.GetSubstructMatches(pattern)
                    for match in matches:
                        if react.GetAtomWithIdx(match[0]).GetAtomMapNum() == carbon_map:
                            has_coupling_partner = True
                            break
        
        return has_coupling_partner
