"""Generated evaluation code for: Convergent synthesis via amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCoupling(BaseScoring):
    """
    Evaluates convergent synthesis strategy via amide coupling.
    Checks if amide bond formation occurs at late stage (low depth) between
    two complex fragments, indicating efficient convergent approach.
    """
    
    def __init__(self, config: Dict):
        self.target_stage = config["parameters"].get("stage", "late")
        self.min_fragment_complexity = config["parameters"].get("fragment_complexity", "high")
        self.complexity_threshold = 15 if self.min_fragment_complexity == "high" else 10
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't occur
        
        if self.target_stage == "late":
            # Reward early depth (closer to final product)
            return max(0, 1 - x) * 10
        else:
            # For other stages, penalize deviation from middle depths
            target_depth = 0.5
            return max(0, 1 - abs(x - target_depth)) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if reaction forms amide bond between complex fragments"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
            
            # Check if amide bond is formed
            if not self._is_amide_formation(reactants, product):
                return False
                
            # Check fragment complexity
            if not self._are_fragments_complex(reactants):
                return False
                
            return True
            
        except Exception:
            return False
    
    def _is_amide_formation(self, reactants, product) -> bool:
        """Detect amide bond formation by checking for new C(=O)N pattern"""
        # Amide pattern
        amide_pattern = Chem.MolFromSmarts("[C](=O)[N]")
        if not amide_pattern:
            return False
            
        # Count amides in product vs reactants
        product_amides = len(product.GetSubstructMatches(amide_pattern))
        reactant_amides = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
        
        # New amide bond formed
        if product_amides > reactant_amides:
            return True
            
        # Alternative check: look for carboxylic acid + amine reactants
        carboxylic_acid = Chem.MolFromSmarts("[C](=O)[OH]")
        amine = Chem.MolFromSmarts("[N;!$(N=*);!$(N#*)]")
        
        has_acid = any(r.HasSubstructMatch(carboxylic_acid) for r in reactants)
        has_amine = any(r.HasSubstructMatch(amine) for r in reactants)
        
        return has_acid and has_amine
    
    def _are_fragments_complex(self, reactants) -> bool:
        """Check if reactants meet complexity threshold"""
        if len(reactants) < 2:
            return False
            
        complexities = [self._calculate_complexity(r) for r in reactants]
        
        # At least two fragments should be above threshold
        complex_fragments = sum(1 for c in complexities if c >= self.complexity_threshold)
        return complex_fragments >= 2
    
    def _calculate_complexity(self, mol) -> int:
        """Simple complexity metric based on heavy atoms, rings, and heteroatoms"""
        if not mol:
            return 0
            
        heavy_atoms = mol.GetNumHeavyAtoms()
        rings = mol.GetRingInfo().NumRings()
        heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        
        # Weighted complexity score
        return heavy_atoms + (rings * 3) + (heteroatoms * 2)
