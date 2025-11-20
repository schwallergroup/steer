"""Generated evaluation code for: Alcohol protection via acetate ester strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholAcetateProtection(BaseScoring):
    """
    Evaluates synthesis routes for alcohol protection via acetate ester strategy.
    Checks if a primary alcohol is protected as an acetate ester and carried through
    multiple synthetic steps before deprotection.
    """
    
    def __init__(self, config: Dict):
        self.steps_protected = config["parameters"].get("steps_protected", 3)
        self.acetate_pattern = Chem.MolFromSmarts("[CH2]-O-C(=O)-C")  # Primary alcohol acetate
        self.alcohol_pattern = Chem.MolFromSmarts("[CH2]-O")  # Primary alcohol
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Better score if protection occurs early and lasts the required steps
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves acetate protection of a primary alcohol
        and the protection persists for the required number of steps.
        """
        rxn_smiles = d["metadata"].get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Check if this is an acetate protection reaction
            if self._is_acetate_protection(reactants, products):
                # Check if the acetate persists for required steps
                return self._check_protection_persistence(d)
                
            return False
            
        except:
            return False
    
    def _is_acetate_protection(self, reactants, products) -> bool:
        """Check if reaction converts primary alcohol to acetate ester."""
        # Look for primary alcohol in reactants
        has_alcohol = any(mol.HasSubstructMatch(self.alcohol_pattern) for mol in reactants if mol)
        
        # Look for acetate ester in products  
        has_acetate = any(mol.HasSubstructMatch(self.acetate_pattern) for mol in products if mol)
        
        # Check for typical acetylation reagents (acetic anhydride, acetyl chloride)
        acetylation_reagents = [
            Chem.MolFromSmarts("CC(=O)OC(=O)C"),  # Acetic anhydride
            Chem.MolFromSmarts("CC(=O)Cl")        # Acetyl chloride
        ]
        
        has_reagent = any(
            any(mol.HasSubstructMatch(reagent) for mol in reactants if mol)
            for reagent in acetylation_reagents
        )
        
        return has_alcohol and has_acetate and has_reagent
    
    def _check_protection_persistence(self, d) -> bool:
        """
        Check if the acetate protection persists through the required number of steps
        by traversing forward in the synthesis tree.
        """
        current_node = d
        steps_count = 0
        
        # Traverse forward through synthesis steps
        while current_node and steps_count < self.steps_protected:
            # Check if current molecule contains the acetate protection
            rxn_smiles = current_node["metadata"].get("mapped_reaction_smiles", "")
            if rxn_smiles:
                try:
                    products = rxn_smiles.split(">>")[1]
                    product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
                    
                    # Check if acetate is still present
                    has_acetate = any(mol.HasSubstructMatch(self.acetate_pattern) for mol in product_mols if mol)
                    
                    if not has_acetate:
                        # Check if this is deprotection (acetate removed, alcohol restored)
                        has_alcohol = any(mol.HasSubstructMatch(self.alcohol_pattern) for mol in product_mols if mol)
                        if has_alcohol and steps_count >= self.steps_protected:
                            return True
                        else:
                            return False  # Protection lost too early
                            
                except:
                    return False
            
            steps_count += 1
            # Move to next step (this would need to be implemented based on tree structure)
            current_node = self._get_next_step(current_node)
        
        return steps_count >= self.steps_protected
    
    def _get_next_step(self, node):
        """Helper to get next step in synthesis - implementation depends on tree structure."""
        # This would need to be implemented based on the actual tree structure
        # For now, return None to end traversal
        return None
