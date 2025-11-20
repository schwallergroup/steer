"""Generated evaluation code for: Complex protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for complex protecting group cycling strategies.
    Detects when ester protecting groups (tert-butyl, benzyl, ethyl) are installed,
    removed, and reinstalled in cyclic patterns throughout the synthesis.
    """
    
    def __init__(self, config):
        self.min_cycles = int(config["protection_deprotection_cycles"].replace(">", ""))
        self.group_types = config["group_types"]
        
        # Define SMARTS patterns for protecting group detection
        self.pg_patterns = {
            "tert-butyl_ester": "[C:1](=O)OC(C)(C)C",
            "benzyl_ester": "[C:1](=O)OCc1ccccc1", 
            "ethyl_ester": "[C:1](=O)OCC"
        }
        
        # Compile patterns for enabled group types
        self.compiled_patterns = {}
        for group_type in self.group_types:
            if group_type in self.pg_patterns:
                pattern = Chem.MolFromSmarts(self.pg_patterns[group_type])
                self.compiled_patterns[group_type] = pattern

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group states across reactions
        pg_history = self._analyze_pg_cycling(reactions)
        
        # Count complete protection-deprotection cycles
        cycle_count = self._count_cycles(pg_history)
        
        condition = cycle_count > self.min_cycles
        return condition, len(reactions)

    def _analyze_pg_cycling(self, reactions) -> Dict[str, List[str]]:
        """Analyze protecting group installation/removal across reactions."""
        pg_history = {group: [] for group in self.group_types}
        
        for rxn in reactions:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            try:
                # Parse reactants and products
                reactant_mols = []
                for smi in reactants_smiles.split("."):
                    mol = Chem.MolFromSmiles(smi.strip())
                    if mol:
                        reactant_mols.append(mol)
                        
                product_mols = []
                for smi in products_smiles.split("."):
                    mol = Chem.MolFromSmiles(smi.strip())
                    if mol:
                        product_mols.append(mol)
                
                # Check each protecting group type
                for group_type, pattern in self.compiled_patterns.items():
                    reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) 
                                         for mol in reactant_mols)
                    product_matches = sum(len(mol.GetSubstructMatches(pattern)) 
                                        for mol in product_mols)
                    
                    if product_matches > reactant_matches:
                        pg_history[group_type].append("protection")
                    elif product_matches < reactant_matches:
                        pg_history[group_type].append("deprotection")
                    else:
                        pg_history[group_type].append("none")
                        
            except Exception:
                # Skip problematic reactions
                for group_type in self.group_types:
                    pg_history[group_type].append("none")
                    
        return pg_history

    def _count_cycles(self, pg_history: Dict[str, List[str]]) -> int:
        """Count complete protection-deprotection cycles across all group types."""
        total_cycles = 0
        
        for group_type, history in pg_history.items():
            cycles = 0
            state = "unprotected"  # Start assuming unprotected
            
            for event in history:
                if event == "protection" and state == "unprotected":
                    state = "protected"
                elif event == "deprotection" and state == "protected":
                    state = "unprotected"
                    cycles += 1
                elif event == "protection" and state == "protected":
                    # Re-protection after deprotection counts as cycling
                    cycles += 1
                    
            total_cycles += cycles
            
        return total_cycles

    def route_scoring(self, x) -> float:
        """Convert cycle count to score. More cycles = better score."""
        if x <= self.min_cycles:
            return 0
        # Scale score based on excess cycles beyond minimum
        return min(10, (x - self.min_cycles) * 2)
