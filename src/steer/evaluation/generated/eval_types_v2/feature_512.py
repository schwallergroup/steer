"""Generated evaluation code for: Multiple protecting group swap cycles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleProtectingGroupSwapCycles(MultiRxnCondBase):
    """
    Evaluates routes based on the presence of multiple protecting group swap cycles.
    Detects sequences where protecting groups are swapped back and forth (e.g., Boc→Cbz→Boc).
    """
    
    def __init__(self, config):
        self.target_swap_cycles = config.get("swap_cycles", 2)
        self.groups_involved = config.get("groups_involved", ["Boc", "Cbz"])
        self.consecutive_swaps = config.get("consecutive_swaps", True)
        
        # Define SMARTS patterns for common protecting groups
        self.pg_patterns = {
            "Boc": "[NX3][CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",  # tert-butoxycarbonyl
            "Cbz": "[NX3][CX3](=[OX1])[OX2][CH2]c1ccccc1",  # benzyloxycarbonyl
            "Fmoc": "[NX3][CX3](=[OX1])[OX2][CH2]C1c2ccccc2-c2ccccc21",  # fluorenylmethoxycarbonyl
            "Ts": "[NX3][SX4](=[OX1])(=[OX1])c1ccc([CH3])cc1",  # tosyl
            "Ac": "[NX3][CX3](=[OX1])[CH3]"  # acetyl
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations throughout the route
        pg_operations = []
        
        for rxn in reactions:
            operation = self.analyze_pg_operation(rxn)
            if operation:
                pg_operations.append(operation)
        
        # Count swap cycles
        swap_cycles = self.count_swap_cycles(pg_operations)
        
        if self.consecutive_swaps:
            # Check if swaps occur in consecutive reactions
            consecutive_cycles = self.count_consecutive_swap_cycles(pg_operations)
            condition = consecutive_cycles >= self.target_swap_cycles
        else:
            condition = swap_cycles >= self.target_swap_cycles
            
        return condition, len(reactions)

    def analyze_pg_operation(self, rxn):
        """Analyze a reaction to determine if it's a protecting group operation."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".") if smi]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".") if smi]
            
            if not all(reactants) or not all(products):
                return None
            
            # Check for protecting group addition/removal/swap
            reactant_pgs = self.detect_protecting_groups(reactants)
            product_pgs = self.detect_protecting_groups(products)
            
            # Determine operation type
            if len(product_pgs) > len(reactant_pgs):
                # Protection reaction
                added_pg = list(set(product_pgs) - set(reactant_pgs))
                if added_pg:
                    return {"type": "protection", "group": added_pg[0]}
            elif len(product_pgs) < len(reactant_pgs):
                # Deprotection reaction
                removed_pg = list(set(reactant_pgs) - set(product_pgs))
                if removed_pg:
                    return {"type": "deprotection", "group": removed_pg[0]}
            elif reactant_pgs != product_pgs and len(reactant_pgs) == len(product_pgs):
                # Swap reaction
                removed_pg = list(set(reactant_pgs) - set(product_pgs))
                added_pg = list(set(product_pgs) - set(reactant_pgs))
                if removed_pg and added_pg:
                    return {"type": "swap", "from": removed_pg[0], "to": added_pg[0]}
                    
        except Exception:
            pass
            
        return None

    def detect_protecting_groups(self, mols):
        """Detect protecting groups in a list of molecules."""
        detected_pgs = []
        
        for mol in mols:
            if mol is None:
                continue
                
            for pg_name, pattern in self.pg_patterns.items():
                if pg_name in self.groups_involved:
                    try:
                        pg_mol = Chem.MolFromSmarts(pattern)
                        if pg_mol and mol.HasSubstructMatch(pg_mol):
                            detected_pgs.append(pg_name)
                    except Exception:
                        continue
                        
        return detected_pgs

    def count_swap_cycles(self, pg_operations):
        """Count the number of protecting group swap cycles."""
        cycles = 0
        
        # Look for patterns like: protection A -> deprotection A -> protection B -> deprotection B
        # Or direct swaps: swap A->B -> swap B->A
        
        for i in range(len(pg_operations) - 1):
            current_op = pg_operations[i]
            
            if current_op["type"] == "swap":
                # Look for reverse swap later
                from_group = current_op["from"]
                to_group = current_op["to"]
                
                for j in range(i + 1, len(pg_operations)):
                    future_op = pg_operations[j]
                    if (future_op["type"] == "swap" and 
                        future_op["from"] == to_group and 
                        future_op["to"] == from_group):
                        cycles += 1
                        break
        
        return cycles

    def count_consecutive_swap_cycles(self, pg_operations):
        """Count consecutive protecting group swap cycles."""
        cycles = 0
        i = 0
        
        while i < len(pg_operations) - 1:
            current_op = pg_operations[i]
            
            if current_op["type"] == "swap":
                from_group = current_op["from"]
                to_group = current_op["to"]
                
                # Check if the next operation is the reverse swap
                if (i + 1 < len(pg_operations) and
                    pg_operations[i + 1]["type"] == "swap" and
                    pg_operations[i + 1]["from"] == to_group and
                    pg_operations[i + 1]["to"] == from_group):
                    cycles += 1
                    i += 2  # Skip the next operation as it's part of this cycle
                else:
                    i += 1
            else:
                i += 1
                
        return cycles
