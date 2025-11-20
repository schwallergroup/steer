"""Generated evaluation code for: Multiple protecting group cycling on piperidine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Checks for multiple protecting group cycling on a specific nitrogen atom.
    Looks for a sequence where the same nitrogen undergoes protection/deprotection
    cycles with specified protecting groups (TFA and Cbz).
    """
    
    def __init__(self, config):
        self.atom_type = config.get("atom_type", "nitrogen")
        self.substructure_smarts = config.get("substructure_smarts", "[N;R1]")
        self.required_cycles = config.get("protection_cycles", 2)
        self.protecting_groups = config.get("protecting_groups", ["TFA", "Cbz"])
        
        # SMARTS patterns for common protecting groups
        self.pg_patterns = {
            "TFA": "[N;!H0,!H1][C](=[O])[C]([F])([F])[F]",  # Trifluoroacetamide
            "Cbz": "[N;!H0,!H1][C](=[O])[O][CH2][c]1[cH][cH][cH][cH][cH]1",  # Carboxybenzyl
            "Boc": "[N;!H0,!H1][C](=[O])[O][C]([CH3])([CH3])[CH3]",  # tert-Butoxycarbonyl
            "Fmoc": "[N;!H0,!H1][C](=[O])[O][CH2][CH]1[c]2[cH][cH][cH][cH][c]2[c]2[cH][cH][cH][cH][c]12"  # Fluorenylmethoxycarbonyl
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations on nitrogen atoms by atom map number
        pg_operations = {}  # atom_map_num -> [(operation_type, pg_type, reaction_index)]
        
        for i, rxn in enumerate(reactions):
            operations = self.analyze_protecting_group_changes(rxn)
            for atom_map, operation_type, pg_type in operations:
                if atom_map not in pg_operations:
                    pg_operations[atom_map] = []
                pg_operations[atom_map].append((operation_type, pg_type, i))
        
        # Check if any nitrogen has the required cycling pattern
        condition_met = False
        for atom_map, operations in pg_operations.items():
            if self.has_required_cycling(operations):
                condition_met = True
                break
        
        return condition_met, len(reactions)

    def analyze_protecting_group_changes(self, rxn):
        """
        Analyze a single reaction for protecting group addition/removal.
        Returns list of (atom_map_num, operation_type, pg_type) tuples.
        """
        operations = []
        
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            # Get all molecules with atom maps
            all_reactants = [mol for mol in reactants if mol is not None]
            all_products = [mol for mol in products if mol is not None]
            
            # Find nitrogen atoms that match our substructure in reactants and products
            reactant_nitrogens = self.get_mapped_nitrogens(all_reactants)
            product_nitrogens = self.get_mapped_nitrogens(all_products)
            
            # Compare protecting groups on each nitrogen
            all_atom_maps = set(reactant_nitrogens.keys()) | set(product_nitrogens.keys())
            
            for atom_map in all_atom_maps:
                reactant_pgs = reactant_nitrogens.get(atom_map, set())
                product_pgs = product_nitrogens.get(atom_map, set())
                
                # Check for protection (PG added)
                added_pgs = product_pgs - reactant_pgs
                for pg in added_pgs:
                    operations.append((atom_map, "protection", pg))
                
                # Check for deprotection (PG removed)
                removed_pgs = reactant_pgs - product_pgs
                for pg in removed_pgs:
                    operations.append((atom_map, "deprotection", pg))
                    
        except Exception:
            pass  # Skip problematic reactions
        
        return operations

    def get_mapped_nitrogens(self, molecules):
        """
        Find all mapped nitrogen atoms matching the substructure and their protecting groups.
        Returns dict: atom_map_num -> set of protecting group types
        """
        nitrogen_pgs = {}
        
        for mol in molecules:
            if mol is None:
                continue
                
            # Find nitrogens matching the substructure pattern
            pattern = Chem.MolFromSmarts(self.substructure_smarts)
            if pattern is None:
                continue
                
            matches = mol.GetSubstructMatches(pattern)
            
            for match in matches:
                n_idx = match[0]  # First atom in match should be nitrogen
                n_atom = mol.GetAtomByIdx(n_idx)
                atom_map = n_atom.GetAtomMapNum()
                
                if atom_map > 0:  # Only consider mapped atoms
                    # Check what protecting groups are attached
                    pgs = self.identify_protecting_groups(mol, n_idx)
                    nitrogen_pgs[atom_map] = pgs
                    
        return nitrogen_pgs

    def identify_protecting_groups(self, mol, nitrogen_idx):
        """
        Identify protecting groups attached to a specific nitrogen atom.
        Returns set of protecting group types.
        """
        protecting_groups = set()
        
        for pg_name in self.protecting_groups:
            if pg_name in self.pg_patterns:
                pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        if nitrogen_idx in match:
                            protecting_groups.add(pg_name)
                            break
        
        return protecting_groups

    def has_required_cycling(self, operations):
        """
        Check if the operations show the required protecting group cycling pattern.
        Expected pattern: multiple protection/deprotection cycles with specified groups.
        """
        if len(operations) < self.required_cycles * 2:
            return False
        
        # Sort operations by reaction index
        operations.sort(key=lambda x: x[2])
        
        # Count cycles for each protecting group type
        pg_cycles = {pg: 0 for pg in self.protecting_groups}
        pg_state = {pg: False for pg in self.protecting_groups}  # False = not protected
        
        for operation_type, pg_type, _ in operations:
            if pg_type in self.protecting_groups:
                if operation_type == "protection" and not pg_state[pg_type]:
                    pg_state[pg_type] = True
                elif operation_type == "deprotection" and pg_state[pg_type]:
                    pg_state[pg_type] = False
                    pg_cycles[pg_type] += 1
        
        # Check if we have enough cycles across the specified protecting groups
        total_cycles = sum(pg_cycles.values())
        return total_cycles >= self.required_cycles
