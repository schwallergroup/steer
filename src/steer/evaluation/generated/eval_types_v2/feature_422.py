"""Generated evaluation code for: Multiple protecting group swaps on nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleProtectingGroupSwaps(MultiRxnCondBase):
    """
    Checks for multiple protecting group swaps on nitrogen atoms.
    Tracks sequential protecting group changes on the same nitrogen atom.
    """
    
    def __init__(self, config):
        self.atom_type = config["parameters"]["atom_type"]
        self.swap_count = config["parameters"]["swap_count"]
        self.groups = config["parameters"]["groups"]
        
        # SMARTS patterns for common protecting groups
        self.pg_patterns = {
            "Bn": "[NX3:1][CH2][c]",  # Benzyl
            "Boc": "[NX3:1]C(=O)OC(C)(C)C",  # tert-Butoxycarbonyl
            "Cbz": "[NX3:1]C(=O)O[CH2][c]",  # Carbobenzyloxy
            "Fmoc": "[NX3:1]C(=O)OCC1c2ccccc2-c2ccccc21",  # Fluorenylmethoxycarbonyl
            "Ts": "[NX3:1]S(=O)(=O)c1ccc(C)cc1",  # Tosyl
            "Ms": "[NX3:1]S(=O)(=O)C"  # Mesyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group changes on each nitrogen by atom map number
        nitrogen_pg_history = {}
        
        for rxn in reactions:
            pg_changes = self.detect_pg_changes(rxn)
            for atom_map, (old_pg, new_pg) in pg_changes.items():
                if atom_map not in nitrogen_pg_history:
                    nitrogen_pg_history[atom_map] = []
                nitrogen_pg_history[atom_map].append((old_pg, new_pg))
        
        # Count swaps for each nitrogen
        max_swaps = 0
        for atom_map, changes in nitrogen_pg_history.items():
            swap_count = len(changes)
            max_swaps = max(max_swaps, swap_count)
        
        condition = max_swaps >= self.swap_count
        return condition, len(reactions)
    
    def detect_pg_changes(self, rxn):
        """Detect protecting group changes on nitrogen atoms in a reaction."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return {}
            
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return {}
            
            # Get nitrogen atoms with protecting groups in reactants and products
            reactant_pg_map = self.get_nitrogen_pg_mapping(reactant_mols)
            product_pg_map = self.get_nitrogen_pg_mapping(product_mols)
            
            # Find changes
            pg_changes = {}
            all_atom_maps = set(reactant_pg_map.keys()) | set(product_pg_map.keys())
            
            for atom_map in all_atom_maps:
                old_pg = reactant_pg_map.get(atom_map, "H")  # Default to H if no PG
                new_pg = product_pg_map.get(atom_map, "H")
                
                if old_pg != new_pg:
                    pg_changes[atom_map] = (old_pg, new_pg)
            
            return pg_changes
            
        except Exception:
            return {}
    
    def get_nitrogen_pg_mapping(self, mols):
        """Get mapping of nitrogen atom map numbers to their protecting groups."""
        pg_mapping = {}
        
        for mol in mols:
            for pg_name, pattern in self.pg_patterns.items():
                patt = Chem.MolFromSmarts(pattern)
                if patt is None:
                    continue
                
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    n_idx = match[0]  # First atom in pattern is nitrogen
                    n_atom = mol.GetAtomWithIdx(n_idx)
                    atom_map = n_atom.GetAtomMapNum()
                    
                    if atom_map > 0:
                        pg_mapping[atom_map] = pg_name
        
        return pg_mapping
    
    def route_scoring(self, x):
        """Score based on presence of multiple protecting group swaps."""
        if x < 0:
            return 0  # Condition not met
        else:
            return 1 - x  # Earlier swaps are slightly better
