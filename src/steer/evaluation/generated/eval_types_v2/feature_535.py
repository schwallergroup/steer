"""Generated evaluation code for: Multiple silyl protecting group swaps"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleSilylProtectingGroupSwaps(MultiRxnCondBase):
    """
    Evaluates synthesis routes for multiple silyl protecting group swaps.
    Checks if the route contains the specified number of silyl group exchanges
    (deprotection followed by protection with a different silyl group).
    """
    
    def __init__(self, config):
        self.protecting_group_type = config.get("protecting_group_type", "silyl")
        self.strategy_type = config.get("strategy_type", "multiple_swaps")
        self.target_swap_count = config.get("swap_count", 2)
        
        # Define silyl protecting group SMARTS patterns
        self.silyl_patterns = [
            "[Si]([CH3])([CH3])[CH3]",  # TMS (trimethylsilyl)
            "[Si]([CH2][CH3])([CH2][CH3])[CH2][CH3]",  # TES (triethylsilyl)
            "[Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]",  # TBS/TBDMS
            "[Si]([CH3])([CH3])[c1]ccccc1",  # TBDPS (tert-butyldiphenylsilyl)
            "[Si]([CH2][CH3])([CH2][CH3])([CH2][CH3])",  # TES alternative
            "[Si]([CH3])([CH3])[CH2][CH2][CH3]"  # TIPS (triisopropylsilyl)
        ]
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        swap_count = self.count_silyl_swaps(reactions)
        
        condition = swap_count >= self.target_swap_count
        return condition, len(reactions)
    
    def count_silyl_swaps(self, reactions) -> int:
        """Count the number of silyl protecting group swaps in the reaction sequence."""
        silyl_changes = []
        
        for rxn in reactions:
            change_type = self.analyze_silyl_change(rxn)
            if change_type:
                silyl_changes.append(change_type)
        
        # Count swaps: deprotection followed by protection with different group
        swap_count = 0
        i = 0
        while i < len(silyl_changes) - 1:
            if silyl_changes[i] == "deprotection":
                # Look for subsequent protection
                for j in range(i + 1, min(i + 3, len(silyl_changes))):  # Within next 2 steps
                    if silyl_changes[j] == "protection":
                        swap_count += 1
                        i = j
                        break
                else:
                    i += 1
            else:
                i += 1
        
        return swap_count
    
    def analyze_silyl_change(self, rxn):
        """Analyze if a reaction involves silyl protection or deprotection."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return None
            
            # Count silyl groups in reactants and products
            reactant_silyl_count = sum(self.count_silyl_groups(mol) for mol in reactants)
            product_silyl_count = sum(self.count_silyl_groups(mol) for mol in products)
            
            if product_silyl_count > reactant_silyl_count:
                return "protection"
            elif product_silyl_count < reactant_silyl_count:
                return "deprotection"
            else:
                # Same count - could be a swap in single step
                if self.is_silyl_substitution(reactants, products):
                    return "substitution"
            
            return None
            
        except Exception:
            return None
    
    def count_silyl_groups(self, mol):
        """Count the number of silyl protecting groups in a molecule."""
        if mol is None:
            return 0
        
        count = 0
        for pattern in self.silyl_patterns:
            try:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                    matches = mol.GetSubstructMatches(pattern_mol)
                    count += len(matches)
            except Exception:
                continue
        
        return count
    
    def is_silyl_substitution(self, reactants, products):
        """Check if different types of silyl groups are present in reactants vs products."""
        reactant_types = set()
        product_types = set()
        
        for mol in reactants:
            reactant_types.update(self.get_silyl_types(mol))
        
        for mol in products:
            product_types.update(self.get_silyl_types(mol))
        
        # Different silyl types suggest substitution
        return len(reactant_types) > 0 and len(product_types) > 0 and reactant_types != product_types
    
    def get_silyl_types(self, mol):
        """Get the types of silyl groups present in a molecule."""
        if mol is None:
            return set()
        
        types = set()
        for i, pattern in enumerate(self.silyl_patterns):
            try:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                    types.add(i)  # Use index as type identifier
            except Exception:
                continue
        
        return types
