"""Generated evaluation code for: Multiple protecting group cycles on single nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleNitrogenProtectingGroupCycles(MultiRxnCondBase):
    """
    Detects multiple protecting group cycles (≥2) on a single nitrogen atom.
    Tracks protection/deprotection sequences like Boc→TFA→Boc→Cbz on the same N.
    """
    
    def __init__(self, config):
        self.min_cycles = config.get("cycle_count", 2)
        if isinstance(self.min_cycles, str) and self.min_cycles.startswith("≥"):
            self.min_cycles = int(self.min_cycles[1:])
        self.atom_type = config.get("atom_type", "nitrogen")
        self.same_atom = config.get("same_atom", True)
        
        # Common protecting group patterns for nitrogen
        self.protecting_groups = {
            "Boc": "[#7]-C(=O)OC(C)(C)C",
            "Cbz": "[#7]-C(=O)OCc1ccccc1", 
            "Fmoc": "[#7]-C(=O)OCc1ccc2c(c1)C3c4ccccc4-c4ccccc4C3C2",
            "Ts": "[#7]-S(=O)(=O)c1ccc(C)cc1",
            "Ac": "[#7]-C(=O)C",
            "Tf": "[#7]-S(=O)(=O)C(F)(F)F"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations on each mapped nitrogen
        nitrogen_operations = {}  # {atom_map_num: [operations]}
        
        for rxn in reactions:
            pg_ops = self.detect_protecting_group_operations(rxn)
            for atom_map, operation in pg_ops:
                if atom_map not in nitrogen_operations:
                    nitrogen_operations[atom_map] = []
                nitrogen_operations[atom_map].append(operation)
        
        # Count complete cycles for each nitrogen
        max_cycles = 0
        for atom_map, operations in nitrogen_operations.items():
            cycles = self.count_protection_cycles(operations)
            max_cycles = max(max_cycles, cycles)
        
        condition = max_cycles >= self.min_cycles
        return condition, len(reactions)
    
    def detect_protecting_group_operations(self, rxn):
        """Detect protection/deprotection operations on nitrogen atoms"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return []
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return []
            
            operations = []
            
            # Get all mapped nitrogen atoms from reactants and products  
            reactant_nitrogens = self.get_mapped_nitrogens(reactant_mols)
            product_nitrogens = self.get_mapped_nitrogens(product_mols)
            
            # Compare protecting group status for each mapped nitrogen
            for atom_map in set(reactant_nitrogens.keys()) | set(product_nitrogens.keys()):
                if atom_map in reactant_nitrogens and atom_map in product_nitrogens:
                    reactant_pg = self.identify_protecting_group(reactant_nitrogens[atom_map])
                    product_pg = self.identify_protecting_group(product_nitrogens[atom_map])
                    
                    if reactant_pg != product_pg:
                        if reactant_pg is None and product_pg is not None:
                            operations.append((atom_map, f"protect_{product_pg}"))
                        elif reactant_pg is not None and product_pg is None:
                            operations.append((atom_map, f"deprotect_{reactant_pg}"))
                        elif reactant_pg is not None and product_pg is not None:
                            operations.append((atom_map, f"exchange_{reactant_pg}_to_{product_pg}"))
            
            return operations
            
        except Exception:
            return []
    
    def get_mapped_nitrogens(self, mols):
        """Get all mapped nitrogen atoms from a list of molecules"""
        mapped_nitrogens = {}
        
        for mol in mols:
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetAtomMapNum() > 0:
                    mapped_nitrogens[atom.GetAtomMapNum()] = (mol, atom)
        
        return mapped_nitrogens
    
    def identify_protecting_group(self, mol_atom_tuple):
        """Identify which protecting group (if any) is attached to the nitrogen"""
        mol, nitrogen_atom = mol_atom_tuple
        
        for pg_name, pattern in self.protecting_groups.items():
            try:
                pg_pattern = Chem.MolFromSmarts(pattern)
                if pg_pattern is None:
                    continue
                    
                matches = mol.GetSubstructMatches(pg_pattern)
                for match in matches:
                    # Check if this nitrogen atom is part of the match
                    if nitrogen_atom.GetIdx() in match:
                        return pg_name
            except Exception:
                continue
        
        return None
    
    def count_protection_cycles(self, operations):
        """Count complete protection/deprotection cycles"""
        if len(operations) < 2:
            return 0
        
        cycles = 0
        i = 0
        
        while i < len(operations) - 1:
            current_op = operations[i]
            
            # Look for protection followed by deprotection
            if current_op.startswith("protect_"):
                pg_name = current_op.split("_", 1)[1]
                
                # Find corresponding deprotection
                for j in range(i + 1, len(operations)):
                    next_op = operations[j]
                    if next_op == f"deprotect_{pg_name}" or next_op.startswith(f"exchange_{pg_name}_"):
                        cycles += 1
                        i = j
                        break
                else:
                    i += 1
            else:
                i += 1
        
        return cycles
