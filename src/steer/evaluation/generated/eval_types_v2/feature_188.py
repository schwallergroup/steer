"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates routes for late-stage nucleophilic aromatic substitution (SNAr) reactions.
    Specifically looks for C-N bond formation through SNAr displacement of leaving groups
    (like chloride) by anilines in the final stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "normalized")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Late stage preferred
    
    def route_scoring(self, x) -> float:
        """
        Scoring function that rewards late-stage SNAr reactions.
        Lower depth values (closer to final product) get higher scores.
        """
        if x < 0:
            return 0  # SNAr reaction doesn't happen
        
        if self.condition_type == "bool":
            return 1  # Just presence/absence
        else:
            # Reward late-stage reactions (lower depth values)
            return max(0, 1 - x)  # Linear decay from 1 to 0 as depth increases
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction node represents a nucleophilic aromatic substitution
        forming a C-N bond.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            return self._is_snar_cn_formation(prod_mol, react_mols)
            
        except Exception:
            return False
    
    def _is_snar_cn_formation(self, product, reactants) -> bool:
        """
        Determines if the reaction represents SNAr C-N bond formation.
        Looks for:
        1. Aromatic C-N bond in product
        2. Corresponding aromatic C-LG bond in reactant
        3. Aniline or amine nucleophile in reactants
        """
        # Find aromatic C-N bonds in product using atom mapping
        aromatic_cn_bonds = []
        for bond in product.GetBonds():
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or
                (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C')):
                # Check if carbon is aromatic
                c_atom = atom1 if atom1.GetSymbol() == 'C' else atom2
                n_atom = atom2 if atom1.GetSymbol() == 'C' else atom1
                if c_atom.GetIsAromatic():
                    aromatic_cn_bonds.append((c_atom.GetAtomMapNum(), n_atom.GetAtomMapNum()))
        
        if not aromatic_cn_bonds:
            return False
        
        # Check reactants for SNAr pattern
        for c_map, n_map in aromatic_cn_bonds:
            if self._check_snar_pattern_in_reactants(reactants, c_map, n_map):
                return True
        
        return False
    
    def _check_snar_pattern_in_reactants(self, reactants, c_map, n_map) -> bool:
        """
        Verify SNAr pattern in reactants:
        - Carbon should be bonded to leaving group (Cl, Br, F, NO2, etc.)
        - Nitrogen should be part of aniline or amine nucleophile
        """
        c_reactant = None
        n_reactant = None
        
        # Find which reactants contain the mapped atoms
        for reactant in reactants:
            c_found = any(atom.GetAtomMapNum() == c_map for atom in reactant.GetAtoms())
            n_found = any(atom.GetAtomMapNum() == n_map for atom in reactant.GetAtoms())
            
            if c_found:
                c_reactant = reactant
            if n_found:
                n_reactant = reactant
        
        if not c_reactant or not n_reactant or c_reactant == n_reactant:
            return False
        
        # Check for leaving group on aromatic carbon
        c_atom = None
        for atom in c_reactant.GetAtoms():
            if atom.GetAtomMapNum() == c_map:
                c_atom = atom
                break
        
        if not c_atom or not c_atom.GetIsAromatic():
            return False
        
        # Look for typical leaving groups bonded to the carbon
        leaving_groups = {'Cl', 'Br', 'F', 'I'}
        has_leaving_group = False
        
        for neighbor in c_atom.GetNeighbors():
            if neighbor.GetSymbol() in leaving_groups:
                has_leaving_group = True
                break
            # Check for nitro group or other electron-withdrawing groups
            if neighbor.GetSymbol() == 'N':
                # Could be NO2 group
                n_neighbors = [n.GetSymbol() for n in neighbor.GetNeighbors()]
                if n_neighbors.count('O') >= 2:
                    has_leaving_group = True
                    break
        
        # Check if nitrogen reactant is a suitable nucleophile (amine/aniline)
        n_atom = None
        for atom in n_reactant.GetAtoms():
            if atom.GetAtomMapNum() == n_map:
                n_atom = atom
                break
        
        if not n_atom or n_atom.GetSymbol() != 'N':
            return False
        
        # Simple check for aniline pattern (aromatic ring with NH2)
        aniline_pattern = Chem.MolFromSmarts('c1ccccc1N')
        has_aniline = n_reactant.HasSubstructMatch(aniline_pattern)
        
        return has_leaving_group and (has_aniline or n_atom.GetTotalDegree() <= 2)
