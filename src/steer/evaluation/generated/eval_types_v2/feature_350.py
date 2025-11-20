"""Generated evaluation code for: Alcohol protection strategy during nitro reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholProtectionStrategy(BaseScoring):
    """
    Evaluates alcohol protection strategy during nitro reduction.
    Checks if alcohol is protected with acetate before nitro reduction occurs.
    """
    
    def __init__(self, config: Dict):
        self.functional_group = config["parameters"]["functional_group"]
        self.protecting_group = config["parameters"]["protecting_group"]
        self.protection_step = config["parameters"]["protection_step"]
        self.deprotection_step = config["parameters"]["deprotection_step"]
        
        # SMARTS patterns for detection
        self.alcohol_pattern = Chem.MolFromSmarts("[OH1][CH2,CH1,CH0]")
        self.acetate_pattern = Chem.MolFromSmarts("[CH3]C(=O)O[CH2,CH1,CH0]")
        self.nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
        self.amine_pattern = Chem.MolFromSmarts("[NH2,NH1,NH0]")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is better, score 0-10
            return max(0, 10 - (x * 10))

    def hit_condition(self, d):
        """
        Check if this reaction represents alcohol protection with acetate
        before nitro reduction occurs in the route.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if this is an acetate protection reaction
            has_alcohol_reactant = any(mol.HasSubstructMatch(self.alcohol_pattern) for mol in reactants)
            has_acetate_product = any(mol.HasSubstructMatch(self.acetate_pattern) for mol in products)
            
            if has_alcohol_reactant and has_acetate_product:
                # Verify that nitro group is present (will be reduced later)
                has_nitro = any(mol.HasSubstructMatch(self.nitro_pattern) for mol in reactants + products)
                return has_nitro
                
            return False
            
        except Exception:
            return False

    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Override to check the complete protection strategy across the route.
        Ensures protection occurs before nitro reduction.
        """
        reactions = []
        self._collect_reactions(d, reactions, 0)
        
        protection_depth = -1
        nitro_reduction_depth = -1
        
        for depth, rxn_data in reactions:
            rxn_smiles = rxn_data.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                continue
                
            try:
                rxn_parts = rxn_smiles.split(">>")
                if len(rxn_parts) != 2:
                    continue
                    
                reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
                products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
                
                if not all(reactants) or not all(products):
                    continue
                
                # Check for acetate protection
                if self.hit_condition(rxn_data):
                    protection_depth = depth
                
                # Check for nitro reduction
                has_nitro_reactant = any(mol.HasSubstructMatch(self.nitro_pattern) for mol in reactants)
                has_amine_product = any(mol.HasSubstructMatch(self.amine_pattern) for mol in products)
                
                if has_nitro_reactant and has_amine_product and nitro_reduction_depth == -1:
                    nitro_reduction_depth = depth
                    
            except Exception:
                continue
        
        # Strategy is successful if protection occurs before nitro reduction
        if protection_depth >= 0 and nitro_reduction_depth >= 0 and protection_depth < nitro_reduction_depth:
            total_depth = len(reactions)
            return True, protection_depth / total_depth if total_depth > 0 else 0
        
        return False, -1

    def _collect_reactions(self, node, reactions, depth):
        """Helper method to collect all reactions in the route tree."""
        if "metadata" in node:
            reactions.append((depth, node))
        
        for child in node.get("children", []):
            self._collect_reactions(child, reactions, depth + 1)
