"""Generated evaluation code for: Late pyrimidine ring formation via cyclocondensation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrimidineFormation(BaseScoring):
    """
    Evaluates if pyrimidine ring formation occurs late in the synthesis via cyclocondensation.
    Checks for formation of pyrimidine rings (c1ncncn1) through cyclocondensation reactions
    in the later stages of the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ncncn1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.method = config["parameters"]["method"]  # "cyclocondensation"
        self.pyrimidine_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # Patterns for detecting cyclocondensation precursors
        self.guanidine_pattern = Chem.MolFromSmarts("[N;H2,H1,H0]C([N;H2,H1,H0])=[N;H1,H0]")
        self.beta_keto_ester_pattern = Chem.MolFromSmarts("C(=O)CC(=O)[O,N]")
        self.carbonyl_nitrile_pattern = Chem.MolFromSmarts("C(=O)C#N")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrimidine ring formation doesn't happen
        else:
            # Late-stage formation is better (higher depth fraction gets higher score)
            if self.timing == "late":
                return x * 10  # Scale to 0-10, favor later stages
            else:
                return (1 - x) * 10  # Favor earlier stages
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a pyrimidine ring via cyclocondensation
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".") if r.strip()]
            
            if not product or not all(reactants):
                return False
            
            # Check if pyrimidine ring is formed (present in product but not in reactants)
            product_has_pyrimidine = product.HasSubstructMatch(self.pyrimidine_pattern)
            reactants_have_pyrimidine = any(r.HasSubstructMatch(self.pyrimidine_pattern) for r in reactants)
            
            if not product_has_pyrimidine or reactants_have_pyrimidine:
                return False
            
            # Check if this is a cyclocondensation by looking for characteristic patterns
            return self._is_cyclocondensation(reactants)
            
        except Exception:
            return False
    
    def _is_cyclocondensation(self, reactants) -> bool:
        """
        Check if reactants contain patterns typical of cyclocondensation for pyrimidine formation
        """
        has_guanidine = any(r.HasSubstructMatch(self.guanidine_pattern) for r in reactants)
        has_beta_keto_ester = any(r.HasSubstructMatch(self.beta_keto_ester_pattern) for r in reactants)
        has_carbonyl_nitrile = any(r.HasSubstructMatch(self.carbonyl_nitrile_pattern) for r in reactants)
        
        # Multiple nitrogen sources and carbonyl-containing compounds suggest cyclocondensation
        nitrogen_sources = sum(1 for r in reactants if any(atom.GetAtomicNum() == 7 for atom in r.GetAtoms()))
        carbonyl_sources = sum(1 for r in reactants if r.HasSubstructMatch(Chem.MolFromSmarts("[C,c]=[O,o]")))
        
        # Cyclocondensation typically involves:
        # 1. Guanidine + beta-keto ester, or
        # 2. Multiple nitrogen and carbonyl sources coming together
        return (has_guanidine and (has_beta_keto_ester or has_carbonyl_nitrile)) or \
               (nitrogen_sources >= 1 and carbonyl_sources >= 1 and len(reactants) >= 2)
