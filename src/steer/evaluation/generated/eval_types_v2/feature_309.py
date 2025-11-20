"""Generated evaluation code for: Strategic amidine protection during palladium catalysis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AmidineProtectionStrategy(BaseScoring):
    """
    Evaluates strategic amidine protection during palladium catalysis.
    Checks if Boc protection is applied to amidines before palladium carbonylation
    to prevent catalyst poisoning.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is better (lower depth)
            if self.condition_type == "bool":
                return 1 if x >= 0 else 0
            else:
                # Reward early protection, penalize late protection
                return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of amidines before Pd catalysis"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants, products = mapped_rxn.split(">>")
        
        # Check if this is a Boc protection reaction
        if not self._is_boc_protection(reactants, products):
            return False
            
        # Check if amidine is present in reactants
        if not self._has_amidine_group(reactants):
            return False
            
        # Check if there's a subsequent palladium carbonylation in the route
        return self._has_subsequent_pd_carbonylation(d)
    
    def _is_boc_protection(self, reactants: str, products: str) -> bool:
        """Check if reaction introduces Boc protecting group"""
        try:
            # Boc anhydride or Boc chloride patterns
            boc_reagents = [
                "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc2O
                "CC(C)(C)OC(=O)Cl"  # BocCl
            ]
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check for Boc reagent in reactants
            has_boc_reagent = any(
                any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                    for pattern in boc_reagents if mol)
                for mol in reactant_mols
            )
            
            # Check for Boc-protected product
            boc_protected_pattern = "CC(C)(C)OC(=O)N"  # Boc-N pattern
            has_boc_product = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(boc_protected_pattern))
                for mol in product_mols if mol
            )
            
            return has_boc_reagent and has_boc_product
            
        except:
            return False
    
    def _has_amidine_group(self, reactants: str) -> bool:
        """Check if amidine group is present in reactants"""
        try:
            amidine_pattern = "[#6]=[#7]-[#6](-[#7])"  # C=N-C(-N) amidine pattern
            
            for reactant in reactants.split("."):
                mol = Chem.MolFromSmiles(reactant.strip())
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(amidine_pattern)):
                    return True
            return False
        except:
            return False
    
    def _has_subsequent_pd_carbonylation(self, d) -> bool:
        """Check if there's a palladium carbonylation reaction later in the route"""
        try:
            # Look for palladium catalyst indicators and CO insertion
            pd_patterns = [
                "[Pd]",  # Palladium
                "CO",    # Carbon monoxide
            ]
            
            carbonyl_products = [
                "C(=O)",  # Carbonyl formation
            ]
            
            # This is a simplified check - in practice, you'd traverse the full route tree
            # to look for subsequent reactions
            route_metadata = d.get("route_metadata", {})
            subsequent_reactions = route_metadata.get("subsequent_reactions", [])
            
            for rxn in subsequent_reactions:
                rxn_smiles = rxn.get("reaction_smiles", "")
                if ">>" in rxn_smiles:
                    reactants, products = rxn_smiles.split(">>")
                    
                    # Check for Pd catalyst and CO
                    has_pd = any(pattern in reactants for pattern in pd_patterns[:1])  # Pd
                    has_co = "CO" in reactants or "C=O" in reactants
                    has_carbonyl_product = any(pattern in products for pattern in carbonyl_products)
                    
                    if has_pd and (has_co or has_carbonyl_product):
                        return True
            
            return False
        except:
            return False
