"""Generated evaluation code for: Multi-step synthesis of commercial reagent tributyltin hydride"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CommercialReagentSynthesis(MultiRxnCondBase):
    """
    Evaluates routes that unnecessarily synthesize commercially available reagents.
    Penalizes routes where a commercial reagent is synthesized through multiple steps
    instead of being used directly from commercial sources.
    """
    
    def __init__(self, config):
        self.reagent_smiles = config["reagent_smiles"]
        self.min_synthesis_steps = config.get("synthesis_steps", 3)
        self.reagent_mol = Chem.MolFromSmiles(self.reagent_smiles)
        if self.reagent_mol is None:
            raise ValueError(f"Invalid reagent SMILES: {self.reagent_smiles}")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if the commercial reagent is synthesized through multiple steps.
        Returns (condition_met, synthesis_depth) where condition_met indicates
        unnecessary synthesis of the commercial reagent.
        """
        synthesis_depth = self._find_reagent_synthesis_depth(d)
        
        # Condition is met if reagent is synthesized over multiple steps
        condition_met = synthesis_depth >= self.min_synthesis_steps
        
        return condition_met, synthesis_depth
    
    def _find_reagent_synthesis_depth(self, node, current_depth=0):
        """
        Recursively search the synthesis tree to find if and at what depth
        the target reagent is synthesized.
        """
        # Check if current node produces the target reagent
        if self._node_produces_reagent(node):
            return current_depth
        
        # If this is a leaf node (no children), reagent not found in this branch
        if "children" not in node or not node["children"]:
            return 0
        
        # Recursively check children
        max_depth = 0
        for child in node["children"]:
            child_depth = self._find_reagent_synthesis_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _node_produces_reagent(self, node):
        """
        Check if a reaction node produces the target commercial reagent.
        """
        if "metadata" not in node or "mapped_reaction_smiles" not in node["metadata"]:
            return False
        
        reaction_smiles = node["metadata"]["mapped_reaction_smiles"]
        
        # Split reaction into reactants and products
        if ">>" not in reaction_smiles:
            return False
        
        reactants_smiles, products_smiles = reaction_smiles.split(">>")
        
        # Check if target reagent is produced
        product_mols = []
        for prod_smiles in products_smiles.split("."):
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            if prod_mol is not None:
                product_mols.append(prod_mol)
        
        # Check if any product matches the target reagent
        for prod_mol in product_mols:
            if self._molecules_match(prod_mol, self.reagent_mol):
                return True
        
        return False
    
    def _molecules_match(self, mol1, mol2):
        """
        Check if two molecules are equivalent using InChI comparison.
        """
        try:
            inchi1 = Chem.MolToInchi(mol1)
            inchi2 = Chem.MolToInchi(mol2)
            return inchi1 == inchi2
        except:
            # Fallback to SMILES comparison if InChI fails
            try:
                smiles1 = Chem.MolToSmiles(mol1, canonical=True)
                smiles2 = Chem.MolToSmiles(mol2, canonical=True)
                return smiles1 == smiles2
            except:
                return False
    
    def route_scoring(self, synthesis_depth):
        """
        Convert synthesis depth to penalty score.
        Higher synthesis depth for commercial reagents gets higher penalty.
        """
        if synthesis_depth == 0:
            return 0  # No unnecessary synthesis detected
        elif synthesis_depth < self.min_synthesis_steps:
            return 2  # Minor penalty for short synthesis
        else:
            # Scale penalty with synthesis depth, max at 10
            return min(10, 2 + (synthesis_depth - self.min_synthesis_steps) * 1.5)
