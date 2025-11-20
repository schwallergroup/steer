"""Generated evaluation code for: N-benzyl protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NBenzylProtectingGroupStrategy(BaseScoring):
    """
    Evaluates the use of N-benzyl protecting group strategy in synthesis routes.
    Checks for the presence of N-benzyl protection followed by hydrogenolysis removal.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        else:
            # Earlier use of protecting group strategy is better
            if self.condition_type == "bool":
                return 10
            else:
                return max(0, 10 * (1 - abs(x - self.target_depth)))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves N-benzyl protecting group strategy
        (either protection or deprotection step)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for N-benzyl protection (amine + benzyl halide -> N-benzyl amine)
            if self._is_benzyl_protection(reactants, products):
                return True
                
            # Check for N-benzyl deprotection (hydrogenolysis)
            if self._is_benzyl_deprotection(reactants, products):
                return True
                
        except Exception:
            return False
            
        return False
    
    def _is_benzyl_protection(self, reactants: str, products: str) -> bool:
        """Check if reaction is N-benzyl protection"""
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Look for primary or secondary amine in reactants
            amine_pattern = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
            has_amine_reactant = any(mol.HasSubstructMatch(amine_pattern) for mol in reactant_mols)
            
            # Look for benzyl halide or benzyl alcohol in reactants
            benzyl_electrophile = Chem.MolFromSmarts("c1ccccc1C[Cl,Br,I,OH]")
            has_benzyl_electrophile = any(mol.HasSubstructMatch(benzyl_electrophile) for mol in reactant_mols)
            
            # Look for N-benzyl amine in products
            n_benzyl_pattern = Chem.MolFromSmarts("c1ccccc1CN")
            has_n_benzyl_product = any(mol.HasSubstructMatch(n_benzyl_pattern) for mol in product_mols)
            
            return has_amine_reactant and has_benzyl_electrophile and has_n_benzyl_product
            
        except Exception:
            return False
    
    def _is_benzyl_deprotection(self, reactants: str, products: str) -> bool:
        """Check if reaction is N-benzyl deprotection (hydrogenolysis)"""
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Look for N-benzyl amine in reactants
            n_benzyl_pattern = Chem.MolFromSmarts("c1ccccc1CN")
            has_n_benzyl_reactant = any(mol.HasSubstructMatch(n_benzyl_pattern) for mol in reactant_mols)
            
            # Look for H2 in reactants (hydrogenolysis conditions)
            has_hydrogen = any(Chem.MolToSmiles(mol) == "[H][H]" for mol in reactant_mols)
            
            # Look for free amine in products
            free_amine_pattern = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$(Nc1ccccc1C)]")
            has_free_amine_product = any(mol.HasSubstructMatch(free_amine_pattern) for mol in product_mols)
            
            # Look for toluene in products
            toluene_pattern = Chem.MolFromSmarts("c1ccccc1C")
            has_toluene_product = any(mol.HasSubstructMatch(toluene_pattern) for mol in product_mols)
            
            return has_n_benzyl_reactant and (has_hydrogen or has_free_amine_product) and has_toluene_product
            
        except Exception:
            return False
