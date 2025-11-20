"""Generated evaluation code for: Furan to pyridine ring transformation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FuranToPyridineTransformation(BaseScoring):
    """
    Evaluates synthesis routes for furan to pyridine ring transformation.
    Specifically looks for conversion of 2-acylfuran to 6-hydroxy-2-pyridyl moiety,
    typically achieved through reaction with ammonia.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("condition_type", "bool")
        self.target_depth = config.get("target_depth", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Transformation doesn't occur
        else:
            # Earlier transformation is generally better for strategic bond formations
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction performs furan to pyridine transformation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Look for furan in reactants and pyridine in products
            has_furan_reactant = False
            has_pyridine_product = False
            
            # Furan patterns - look for 2-acylfuran specifically
            furan_patterns = [
                "[#6]1[#6][#6][#8][#6]1",  # Basic furan ring
                "[#6](=[#8])[#6]1[#6][#6][#6][#8]1",  # 2-acylfuran
                "[#6]1[#8][#6][#6][#6]1[#6]=[#8]"  # Alternative 2-acylfuran
            ]
            
            # Pyridine patterns - look for 6-hydroxy-2-pyridyl
            pyridine_patterns = [
                "[#6]1[#6][#6][#6][#7][#6]1",  # Basic pyridine ring
                "[#8][#6]1[#6][#6][#6][#6][#7]1",  # 6-hydroxypyridine
                "[#6]1[#7][#6]([#8])[#6][#6][#6]1"  # Alternative 6-hydroxypyridine
            ]
            
            # Check reactants for furan
            for reactant_smiles in reactants:
                try:
                    mol = Chem.MolFromSmiles(reactant_smiles)
                    if mol is not None:
                        for pattern in furan_patterns:
                            pattern_mol = Chem.MolFromSmarts(pattern)
                            if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                                has_furan_reactant = True
                                break
                        if has_furan_reactant:
                            break
                except:
                    continue
            
            # Check products for pyridine
            for product_smiles in products:
                try:
                    mol = Chem.MolFromSmiles(product_smiles)
                    if mol is not None:
                        for pattern in pyridine_patterns:
                            pattern_mol = Chem.MolFromSmarts(pattern)
                            if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                                has_pyridine_product = True
                                break
                        if has_pyridine_product:
                            break
                except:
                    continue
            
            # Also check for ammonia/nitrogen source in reactants
            has_nitrogen_source = False
            nitrogen_sources = ["N", "[NH3]", "[NH4+]", "[NH2-]"]
            
            for reactant_smiles in reactants:
                if any(n_source in reactant_smiles for n_source in nitrogen_sources):
                    has_nitrogen_source = True
                    break
            
            return has_furan_reactant and has_pyridine_product and has_nitrogen_source
            
        except Exception:
            return False
