"""Generated evaluation code for: Multi-component heterocycle assembly final step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiComponentHeterocycleAssembly(BaseScoring):
    """
    Evaluates whether a multi-component heterocycle assembly reaction occurs as the final step.
    Checks for reactions with 3+ reactants that form heterocyclic products.
    """
    
    def __init__(self, config: Dict):
        self.components_count = config.get("components_count", 3)
        self.step_position = config.get("step_position", "final")
        
        # Common heterocycle patterns
        self.heterocycle_patterns = [
            "[nH]1cccc1",  # pyrrole
            "n1cccc1",     # pyrrole (aromatic N)
            "c1cc[nH]n1",  # pyrazole
            "c1ccnc1",     # pyridine
            "c1ccoc1",     # furan
            "c1ccsc1",     # thiophene
            "c1cncc1",     # pyrazine
            "c1cncn1",     # pyrimidine
            "c1ccncc1",    # pyridine
            "c1nc[nH]n1",  # triazole
            "c1nnc[nH]1",  # triazole
        ]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        else:
            if self.step_position == "final":
                # For final step, closer to 1.0 (end of synthesis) is better
                return 1 - abs(1.0 - x) if x > 0.8 else 0
            else:
                # For other positions, return based on target depth
                return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a multi-component heterocycle assembly"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            products = products_smiles.split(".")
            
            # Check if we have the required number of components
            if len(reactants) < self.components_count:
                return False
            
            # Parse molecules
            reactant_mols = []
            for smi in reactants:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    product_mols.append(mol)
            
            if len(reactant_mols) < self.components_count or not product_mols:
                return False
            
            # Check if any product contains a heterocycle
            has_heterocycle_product = False
            for product_mol in product_mols:
                for pattern in self.heterocycle_patterns:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and product_mol.HasSubstructMatch(pattern_mol):
                        has_heterocycle_product = True
                        break
                if has_heterocycle_product:
                    break
            
            if not has_heterocycle_product:
                return False
            
            # Check if heterocycle is newly formed (not present in all reactants)
            heterocycle_in_all_reactants = True
            for pattern in self.heterocycle_patterns:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol:
                    # Check if this heterocycle pattern is in any product
                    in_product = any(mol.HasSubstructMatch(pattern_mol) for mol in product_mols)
                    if in_product:
                        # Check if it's in all reactants (if so, not newly formed)
                        in_all_reactants = all(mol.HasSubstructMatch(pattern_mol) for mol in reactant_mols)
                        if not in_all_reactants:
                            heterocycle_in_all_reactants = False
                            break
            
            # Return True if we have multi-component reaction forming new heterocycle
            return has_heterocycle_product and not heterocycle_in_all_reactants
            
        except Exception:
            return False
