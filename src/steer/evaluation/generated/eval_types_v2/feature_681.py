"""Generated evaluation code for: Early chiral auxiliary cleavage strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyChiralAuxiliaryCleavage(BaseScoring):
    """
    Evaluates whether Evans auxiliary cleavage occurs early in the synthesis route.
    The Evans auxiliary should be removed at or before the target depth to establish
    stereochemistry early in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config.get("target_depth", 9)
        self.condition_type = config.get("condition_type", "depth")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Evans auxiliary cleavage doesn't occur
        
        # Convert depth to fraction and score early cleavage favorably
        depth_fraction = x
        if depth_fraction <= self.target_depth / 20.0:  # Normalize to 0-1 scale
            return 10  # Perfect score for early cleavage
        else:
            # Penalize late cleavage
            return max(0, 10 - (depth_fraction - self.target_depth / 20.0) * 20)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Evans auxiliary cleavage by checking for the removal of 
        oxazolidinone-based chiral auxiliaries.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".") if smi]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".") if smi]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Evans auxiliary patterns (oxazolidinones)
            evans_patterns = [
                "O=C1OCC(*)N1",  # Basic oxazolidinone core
                "O=C1OC[C@H](*)N1",  # S-stereoisomer
                "O=C1OC[C@@H](*)N1",  # R-stereoisomer
                "O=C1OC[C@H](c2ccccc2)N1",  # Phenyl auxiliary
                "O=C1OC[C@@H](c2ccccc2)N1",  # Phenyl auxiliary opposite stereo
                "O=C1OC[C@H](C(C)C)N1",  # Isopropyl auxiliary
                "O=C1OC[C@@H](C(C)C)N1"   # Isopropyl auxiliary opposite stereo
            ]
            
            # Check if Evans auxiliary is present in reactants but not in products
            evans_in_reactants = False
            evans_in_products = False
            
            for pattern in evans_patterns:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol is None:
                    continue
                    
                # Check reactants for Evans auxiliary
                for mol in reactant_mols:
                    if mol and mol.HasSubstructMatch(pattern_mol):
                        evans_in_reactants = True
                        break
                
                # Check products for Evans auxiliary
                for mol in product_mols:
                    if mol and mol.HasSubstructMatch(pattern_mol):
                        evans_in_products = True
                        break
                
                if evans_in_reactants:
                    break
            
            # Evans auxiliary cleavage occurs if present in reactants but not products
            return evans_in_reactants and not evans_in_products
            
        except Exception:
            return False
