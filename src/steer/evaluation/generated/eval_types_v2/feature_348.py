"""Generated evaluation code for: Intramolecular cyclization for core assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IntramolecularCyclization(BaseScoring):
    """
    Evaluates synthesis routes for intramolecular cyclization reactions that form
    pyrazolo-fused cores from phenylhydrazine derivatives and 1,3-dicarbonyl substrates.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # SMARTS patterns for key structural features
        self.phenylhydrazine_pattern = "[NH2][NH1]c1ccccc1"  # Phenylhydrazine core
        self.dicarbonyl_pattern = "[CX3](=O)[CH2,CH1][CX3](=O)"  # 1,3-dicarbonyl
        self.pyrazolo_core_pattern = "n1nc2ccccc2c1"  # Pyrazolo fused ring system
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Cyclization doesn't happen
            # Earlier cyclization (lower depth) is generally better for core assembly
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves intramolecular cyclization forming pyrazolo core"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if we're forming a pyrazolo core in products
            pyrazolo_pattern = Chem.MolFromSmarts(self.pyrazolo_core_pattern)
            has_pyrazolo_product = any(mol.HasSubstructMatch(pyrazolo_pattern) for mol in products)
            
            if not has_pyrazolo_product:
                return False
            
            # Check if reactants contain the expected substrates
            phenylhydrazine_pattern = Chem.MolFromSmarts(self.phenylhydrazine_pattern)
            dicarbonyl_pattern = Chem.MolFromSmarts(self.dicarbonyl_pattern)
            
            has_phenylhydrazine = False
            has_dicarbonyl = False
            
            for mol in reactants:
                if mol.HasSubstructMatch(phenylhydrazine_pattern):
                    has_phenylhydrazine = True
                if mol.HasSubstructMatch(dicarbonyl_pattern):
                    has_dicarbonyl = True
            
            # Check for intramolecular case (single reactant with both patterns)
            intramolecular = any(
                mol.HasSubstructMatch(phenylhydrazine_pattern) and 
                mol.HasSubstructMatch(dicarbonyl_pattern)
                for mol in reactants
            )
            
            # Return True if we have pyrazolo formation from appropriate substrates
            # Either intramolecular or intermolecular cyclization
            return has_pyrazolo_product and (intramolecular or (has_phenylhydrazine and has_dicarbonyl))
            
        except Exception as e:
            return False
