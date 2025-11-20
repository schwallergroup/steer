"""Generated evaluation code for: Multiple protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for multiple protecting group cycling strategies.
    
    Checks if the route employs sequential protection/deprotection cycles
    on the same functional groups as specified in the configuration.
    """
    
    def __init__(self, config):
        self.protection_cycles = config.get("protection_cycles", [])
        self.required_cycle_count = config.get("cycle_count", 1)
        
        # Define SMARTS patterns for common protecting groups
        self.protecting_group_patterns = {
            "Boc": "[N:1]C(=O)OC(C)(C)C",
            "acetyl": "[N:1]C(=O)C",
            "Bn": "[N:1]Cc1ccccc1",
            "TFA": "[N:1]C(=O)CF3",
            "deprotected": "[N:1]"  # Free amine/nitrogen
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the required protecting group cycles are present in the route."""
        reactions = self.get_rxns(d)
        
        # Track protection state changes for each nitrogen atom
        nitrogen_states = {}
        cycle_counts = {}
        
        # Process reactions in reverse order (from target to starting materials)
        for rxn in reversed(reactions):
            self.track_protection_changes(rxn, nitrogen_states, cycle_counts)
        
        # Check if any nitrogen underwent the required number of cycles
        max_cycles = max(cycle_counts.values()) if cycle_counts else 0
        condition_met = max_cycles >= self.required_cycle_count
        
        # Also verify that the specific cycle patterns are present
        if condition_met:
            condition_met = self.verify_cycle_patterns(reactions)
        
        return condition_met, len(reactions)
    
    def track_protection_changes(self, rxn, nitrogen_states, cycle_counts):
        """Track protection state changes for nitrogen atoms across reactions."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactant_mol = Chem.MolFromSmiles(reactants_smiles)
            product_mols = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            
            if not reactant_mol or not all(product_mols):
                return
            
            # Find mapped nitrogen atoms and their protection states
            reactant_nitrogens = self.get_nitrogen_protection_states(reactant_mol)
            
            for product_mol in product_mols:
                product_nitrogens = self.get_nitrogen_protection_states(product_mol)
                
                # Match nitrogen atoms by atom mapping
                for map_num in reactant_nitrogens:
                    if map_num in product_nitrogens:
                        reactant_state = reactant_nitrogens[map_num]
                        product_state = product_nitrogens[map_num]
                        
                        # Initialize tracking for this nitrogen
                        if map_num not in nitrogen_states:
                            nitrogen_states[map_num] = []
                            cycle_counts[map_num] = 0
                        
                        # Record state change
                        if reactant_state != product_state:
                            nitrogen_states[map_num].append((reactant_state, product_state))
                            
                            # Check if this completes a cycle
                            if self.is_cycle_completion(nitrogen_states[map_num]):
                                cycle_counts[map_num] += 1
                                
        except Exception:
            pass  # Handle malformed SMILES gracefully
    
    def get_nitrogen_protection_states(self, mol):
        """Determine protection state of each mapped nitrogen atom."""
        nitrogen_states = {}
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetAtomMapNum() > 0:
                map_num = atom.GetAtomMapNum()
                
                # Check which protecting group pattern matches
                for pg_name, pattern in self.protecting_group_patterns.items():
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        matches = mol.GetSubstructMatches(pattern_mol)
                        for match in matches:
                            if atom.GetIdx() in match:
                                nitrogen_states[map_num] = pg_name
                                break
                        if map_num in nitrogen_states:
                            break
                
                # If no pattern matched, consider it deprotected
                if map_num not in nitrogen_states:
                    nitrogen_states[map_num] = "deprotected"
        
        return nitrogen_states
    
    def is_cycle_completion(self, state_history):
        """Check if the state changes represent a complete protection cycle."""
        if len(state_history) < 2:
            return False
        
        # Look for patterns like: deprotected -> protected -> deprotected
        # or specific cycles defined in protection_cycles
        states = [change[1] for change in state_history[-3:]]  # Last 3 states
        
        if len(states) >= 3:
            # Simple cycle detection: return to previous state after intermediate state
            return states[0] == states[2] and states[0] != states[1]
        
        return False
    
    def verify_cycle_patterns(self, reactions):
        """Verify that specific protection cycle patterns from config are present."""
        if not self.protection_cycles:
            return True  # No specific patterns required
        
        # Convert reactions to a simplified representation of protection changes
        protection_changes = []
        
        for rxn in reactions:
            if self.is_protection_reaction(rxn):
                change_type = self.classify_protection_change(rxn)
                if change_type:
                    protection_changes.append(change_type)
        
        # Check if required cycle patterns are present
        cycle_pattern_found = False
        for required_cycle in self.protection_cycles:
            if self.pattern_in_changes(required_cycle, protection_changes):
                cycle_pattern_found = True
                break
        
        return cycle_pattern_found
    
    def is_protection_reaction(self, rxn):
        """Check if reaction involves protecting group chemistry."""
        # Look for common protecting group reagents or products
        protecting_reagents = ["(Boc)2O", "Ac2O", "BnBr", "TFA"]
        
        for reagent in protecting_reagents:
            if reagent in rxn:
                return True
        
        return False
    
    def classify_protection_change(self, rxn):
        """Classify the type of protection/deprotection occurring."""
        if "Boc" in rxn and "Ac" in rxn:
            return "Boc->acetyl"
        elif "Bn" in rxn and "TFA" in rxn:
            return "Bn->TFA->deprotected"
        
        return None
    
    def pattern_in_changes(self, pattern, changes):
        """Check if a specific pattern sequence is present in the protection changes."""
        pattern_steps = pattern.split("->")
        if len(pattern_steps) <= len(changes):
            # Simple substring matching for protection change patterns
            changes_str = "->".join(changes)
            return pattern in changes_str
        
        return False
