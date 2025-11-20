"""Generated evaluation code for: Convergent synthesis via two fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are coupled
    via Fischer indole synthesis at a specified timing in the route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.coupling_reaction = config["parameters"]["coupling_reaction"]
        self.coupling_timing = config["parameters"]["coupling_timing"]
        
        # Fischer indole synthesis patterns
        # Phenylhydrazine + carbonyl -> indole
        self.phenylhydrazine_pattern = "c1ccc(NN)cc1"
        self.carbonyl_pattern = "[CX3]=[OX1]"  # Ketone or aldehyde
        self.indole_product_pattern = "c1ccc2[nH]ccc2c1"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        
        # Score based on timing preference
        if self.coupling_timing == "late":
            return (1 - x) * 10  # Later is better (closer to target)
        elif self.coupling_timing == "early":
            return x * 10  # Earlier is better (closer to starting materials)
        else:  # "middle"
            # Peak score around 0.5 (middle of route)
            return (1 - abs(x - 0.5) * 2) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a Fischer indole synthesis
        with the required fragment count.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Check if we have the expected number of fragments
            if len(reactants) != self.fragment_count:
                return False
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check for Fischer indole synthesis pattern
            return self._is_fischer_indole_synthesis(reactant_mols, product_mol)
            
        except Exception:
            return False
    
    def _is_fischer_indole_synthesis(self, reactants, product):
        """
        Check if the reaction matches Fischer indole synthesis pattern:
        phenylhydrazine + carbonyl compound -> indole
        """
        # Product should contain indole substructure
        indole_pattern = Chem.MolFromSmarts(self.indole_product_pattern)
        if not product.HasSubstructMatch(indole_pattern):
            return False
        
        # Check reactants for required components
        has_phenylhydrazine = False
        has_carbonyl = False
        
        phenylhydrazine_mol = Chem.MolFromSmarts(self.phenylhydrazine_pattern)
        carbonyl_mol = Chem.MolFromSmarts(self.carbonyl_pattern)
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(phenylhydrazine_mol):
                has_phenylhydrazine = True
            if reactant.HasSubstructMatch(carbonyl_mol):
                has_carbonyl = True
        
        return has_phenylhydrazine and has_carbonyl
