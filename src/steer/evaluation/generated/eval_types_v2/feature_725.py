"""Generated evaluation code for: Multiple functional group interconversion cycles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiFunctionalGroupCycles(MultiRxnCondBase):
    """
    Detects multiple functional group interconversion cycles in synthesis routes.
    Checks if the route contains at least min_cycles of transformations between
    specified functional groups (nitrile, ester, carboxylic_acid, amide).
    """
    
    def __init__(self, config):
        self.min_cycles = config.get("min_cycles", 2)
        self.functional_groups = config.get("functional_groups", [])
        
        # Define SMARTS patterns for functional groups
        self.fg_patterns = {
            "nitrile": "[C]#[N]",
            "ester": "[C](=[O])[O][C]",
            "carboxylic_acid": "[C](=[O])[O][H]",
            "amide": "[C](=[O])[N]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track functional group transformations
        fg_transformations = []
        
        for rxn in reactions:
            transformation = self.detect_fg_transformation(rxn)
            if transformation:
                fg_transformations.append(transformation)
        
        # Count cycles in the transformation sequence
        cycles = self.count_fg_cycles(fg_transformations)
        
        condition = cycles >= self.min_cycles
        return condition, len(reactions)
    
    def detect_fg_transformation(self, rxn):
        """Detect functional group transformation in a reaction."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Find functional groups in reactants and products
            reactant_fgs = set()
            product_fgs = set()
            
            for mol in reactant_mols:
                if mol:
                    reactant_fgs.update(self.get_functional_groups(mol))
            
            for mol in product_mols:
                if mol:
                    product_fgs.update(self.get_functional_groups(mol))
            
            # Look for transformations between our target functional groups
            for fg_from in self.functional_groups:
                for fg_to in self.functional_groups:
                    if fg_from != fg_to and fg_from in reactant_fgs and fg_to in product_fgs:
                        # Check if this is actually a transformation (fg_from disappears or fg_to appears)
                        if fg_from not in product_fgs or fg_to not in reactant_fgs:
                            return (fg_from, fg_to)
            
            return None
            
        except Exception:
            return None
    
    def get_functional_groups(self, mol):
        """Identify functional groups present in a molecule."""
        fgs = []
        for fg_name, pattern in self.fg_patterns.items():
            if fg_name in self.functional_groups:
                try:
                    query = Chem.MolFromSmarts(pattern)
                    if query and mol.HasSubstructMatch(query):
                        fgs.append(fg_name)
                except Exception:
                    continue
        return fgs
    
    def count_fg_cycles(self, transformations):
        """Count cycles in functional group transformations."""
        if len(transformations) < 2:
            return 0
        
        cycles = 0
        i = 0
        
        while i < len(transformations) - 1:
            current_transform = transformations[i]
            
            # Look for cycles starting from this transformation
            cycle_found = False
            for j in range(i + 1, len(transformations)):
                next_transform = transformations[j]
                
                # Check for direct cycle (A->B, B->A)
                if (current_transform[0] == next_transform[1] and 
                    current_transform[1] == next_transform[0]):
                    cycles += 1
                    cycle_found = True
                    break
                
                # Check for indirect cycle (A->B, then later B->C->A or similar)
                if current_transform[1] == next_transform[0]:
                    # Found continuation, look for return to original
                    for k in range(j + 1, len(transformations)):
                        third_transform = transformations[k]
                        if third_transform[1] == current_transform[0]:
                            cycles += 1
                            cycle_found = True
                            break
                    if cycle_found:
                        break
            
            i += 1
        
        return cycles
