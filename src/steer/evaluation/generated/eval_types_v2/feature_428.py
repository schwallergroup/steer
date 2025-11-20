"""Generated evaluation code for: Sequential functional group modifications on piperidine scaffold"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialPiperidineModification(MultiRxnCondBase):
    """
    Evaluates routes that perform sequential functional group modifications 
    on a piperidine scaffold with bifunctional capability.
    """
    
    def __init__(self, config):
        self.scaffold_smarts = config.get("scaffold_smarts", "[C,N]1[C,N][C,N][C,N][C,N][C,N]1")
        self.sequential_modifications = config.get("sequential_modifications", True)
        self.bifunctional = config.get("bifunctional", True)
        self.scaffold_pattern = Chem.MolFromSmarts(self.scaffold_smarts)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find reactions that modify the piperidine scaffold
        scaffold_modifications = []
        for i, rxn in enumerate(reactions):
            if self.modifies_piperidine_scaffold(rxn):
                scaffold_modifications.append(i)
        
        # Check if we have at least 2 modifications (bifunctional requirement)
        if len(scaffold_modifications) < 2:
            return False, len(reactions)
        
        # Check if modifications are sequential (consecutive in reaction order)
        if self.sequential_modifications:
            sequential = self.are_modifications_sequential(scaffold_modifications)
            if not sequential:
                return False, len(reactions)
        
        # Check if the scaffold has bifunctional capability
        if self.bifunctional:
            bifunctional_present = self.has_bifunctional_scaffold(reactions)
            if not bifunctional_present:
                return False, len(reactions)
        
        return True, len(reactions)
    
    def modifies_piperidine_scaffold(self, rxn):
        """Check if a reaction modifies a piperidine scaffold"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1].split(".")
        
        # Check if reactants contain piperidine scaffold
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        
        has_scaffold_reactant = any(
            mol.HasSubstructMatch(self.scaffold_pattern) for mol in reactant_mols
        )
        
        if not has_scaffold_reactant:
            return False
        
        # Check if products also contain modified piperidine scaffold
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products]
        product_mols = [mol for mol in product_mols if mol is not None]
        
        has_scaffold_product = any(
            mol.HasSubstructMatch(self.scaffold_pattern) for mol in product_mols
        )
        
        return has_scaffold_product
    
    def are_modifications_sequential(self, modification_indices):
        """Check if modifications occur in sequential order"""
        if len(modification_indices) < 2:
            return False
        
        # Sort indices and check if they are consecutive
        sorted_indices = sorted(modification_indices)
        for i in range(len(sorted_indices) - 1):
            if sorted_indices[i+1] - sorted_indices[i] > 2:  # Allow for one intervening reaction
                return False
        
        return True
    
    def has_bifunctional_scaffold(self, reactions):
        """Check if any molecule in the route has a bifunctional piperidine scaffold"""
        # Look for piperidine with at least 2 functional groups
        bifunctional_patterns = [
            "[C,N]1[C,N][C,N]([*:1])[C,N][C,N]([*:2])[C,N]1",  # 1,4-disubstituted
            "[C,N]1[C,N]([*:1])[C,N][C,N]([*:2])[C,N][C,N]1",   # 1,3-disubstituted  
            "[C,N]1[C,N]([*:1])[C,N]([*:2])[C,N][C,N][C,N]1",   # 1,2-disubstituted
        ]
        
        for rxn in reactions:
            rxn_parts = rxn.split(">>")
            all_smiles = rxn_parts[0] + "." + rxn_parts[1]
            
            for smi in all_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None:
                    for pattern_smarts in bifunctional_patterns:
                        pattern = Chem.MolFromSmarts(pattern_smarts)
                        if pattern is not None and mol.HasSubstructMatch(pattern):
                            return True
        
        return False
