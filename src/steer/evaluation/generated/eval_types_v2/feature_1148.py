"""Generated evaluation code for: Multiple protecting group cycles on same nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleProtectingGroupCycles(MultiRxnCondBase):
    """
    Evaluates if multiple protecting group cycles occur on the same nitrogen atom.
    Tracks protection/deprotection sequences for specified protecting groups.
    """
    
    def __init__(self, config):
        self.atom_type = config.get("atom_type", "N")
        self.required_cycles = config.get("protection_cycles", 2)
        self.protecting_groups = config.get("protecting_groups", ["Boc", "benzyl"])
        self.sequential = config.get("sequential", True)
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[N:1]C(=O)OC(C)(C)C",  # Boc-protected nitrogen
            "benzyl": "[N:1]Cc1ccccc1",     # Benzyl-protected nitrogen
            "Cbz": "[N:1]C(=O)OCc1ccccc1",  # Cbz-protected nitrogen
            "Fmoc": "[N:1]C(=O)OCC1c2ccccc2-c2ccccc21",  # Fmoc-protected nitrogen
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events for each nitrogen
        nitrogen_history = {}
        
        for rxn in reactions:
            self.analyze_protection_events(rxn, nitrogen_history)
        
        # Check if any nitrogen has the required number of cycles
        condition_met = self.check_multiple_cycles(nitrogen_history)
        
        return condition_met, len(reactions)
    
    def analyze_protection_events(self, rxn, nitrogen_history):
        """Analyze a single reaction for protection/deprotection events"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Track protected nitrogens in reactants and products
            reactant_protected = self.find_protected_nitrogens(reactant_mols)
            product_protected = self.find_protected_nitrogens(product_mols)
            
            # Identify protection/deprotection events by atom map numbers
            self.identify_events(reactant_protected, product_protected, nitrogen_history)
            
        except Exception:
            pass
    
    def find_protected_nitrogens(self, molecules):
        """Find all protected nitrogens in a list of molecules"""
        protected = {}
        
        for mol in molecules:
            if mol is None:
                continue
                
            for pg_name in self.protecting_groups:
                if pg_name in self.pg_patterns:
                    pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                    if pattern is not None:
                        matches = mol.GetSubstructMatches(pattern)
                        for match in matches:
                            # Get the nitrogen atom (mapped as :1 in SMARTS)
                            n_atom = mol.GetAtomWithIdx(match[0])
                            map_num = n_atom.GetAtomMapNum()
                            if map_num > 0:
                                if map_num not in protected:
                                    protected[map_num] = set()
                                protected[map_num].add(pg_name)
        
        return protected
    
    def identify_events(self, reactant_protected, product_protected, nitrogen_history):
        """Identify protection/deprotection events and update history"""
        all_nitrogens = set(reactant_protected.keys()) | set(product_protected.keys())
        
        for n_map in all_nitrogens:
            if n_map not in nitrogen_history:
                nitrogen_history[n_map] = {"cycles": [], "current_groups": set()}
            
            reactant_groups = reactant_protected.get(n_map, set())
            product_groups = product_protected.get(n_map, set())
            
            # Deprotection: groups present in reactants but not products
            deprotected = reactant_groups - product_groups
            # Protection: groups present in products but not reactants  
            protected = product_groups - reactant_groups
            
            # Update current state
            nitrogen_history[n_map]["current_groups"] = product_groups
            
            # Record cycles (protection followed by deprotection of same group)
            for group in deprotected:
                # Check if this group was previously protected and now deprotected
                nitrogen_history[n_map]["cycles"].append(("deprotect", group))
            
            for group in protected:
                nitrogen_history[n_map]["cycles"].append(("protect", group))
    
    def check_multiple_cycles(self, nitrogen_history):
        """Check if any nitrogen has multiple complete protecting group cycles"""
        for n_map, history in nitrogen_history.items():
            cycles = history["cycles"]
            
            # Count complete cycles for each protecting group
            group_cycles = {}
            for pg in self.protecting_groups:
                group_cycles[pg] = 0
                protect_count = 0
                deprotect_count = 0
                
                for event, group in cycles:
                    if group == pg:
                        if event == "protect":
                            protect_count += 1
                        elif event == "deprotect":
                            deprotect_count += 1
                            # A complete cycle is protection followed by deprotection
                            if protect_count > 0:
                                group_cycles[pg] += 1
                                protect_count -= 1
            
            # Check if we have the required number of cycles
            total_cycles = sum(group_cycles.values())
            if total_cycles >= self.required_cycles:
                if self.sequential:
                    # Check if cycles involve different protecting groups
                    groups_with_cycles = sum(1 for count in group_cycles.values() if count > 0)
                    return groups_with_cycles >= len(self.protecting_groups)
                else:
                    return True
        
        return False
