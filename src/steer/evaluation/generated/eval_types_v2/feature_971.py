"""Generated evaluation code for: Chiral auxiliary protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChiralAuxiliaryStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of chiral auxiliary protecting group strategy.
    Specifically detects the use of tert-butanesulfinyl auxiliary for imine protection
    by looking for sulfinylimine formation and subsequent cleavage reactions.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group_type = config["parameters"]["protecting_group_type"]
        self.functional_group = config["parameters"]["functional_group"]
        self.auxiliary_name = config["parameters"]["auxiliary_name"]
        
        # SMARTS patterns for tert-butanesulfinyl auxiliary detection
        self.sulfinamide_pattern = "CC(C)(C)S(=O)N"  # tert-butanesulfinamide
        self.sulfinylimine_pattern = "CC(C)(C)S(=O)N=C"  # sulfinylimine product
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not used
        else:
            # Earlier use of chiral auxiliary is generally better for stereocontrol
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves chiral auxiliary strategy.
        Look for either formation of sulfinylimine or its cleavage.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check for sulfinylimine formation (protection step)
            formation_detected = self._detect_auxiliary_formation(reactants, products)
            
            # Check for auxiliary cleavage (deprotection step)
            cleavage_detected = self._detect_auxiliary_cleavage(reactants, products)
            
            return formation_detected or cleavage_detected
            
        except Exception:
            return False
    
    def _detect_auxiliary_formation(self, reactants, products):
        """Detect formation of sulfinylimine from sulfinamide + carbonyl compound"""
        sulfinamide_pattern = Chem.MolFromSmarts(self.sulfinamide_pattern)
        sulfinylimine_pattern = Chem.MolFromSmarts(self.sulfinylimine_pattern)
        
        if not sulfinamide_pattern or not sulfinylimine_pattern:
            return False
        
        # Check if we have sulfinamide in reactants
        has_sulfinamide_reactant = any(
            mol.HasSubstructMatch(sulfinamide_pattern) for mol in reactants
        )
        
        # Check if we form sulfinylimine in products
        has_sulfinylimine_product = any(
            mol.HasSubstructMatch(sulfinylimine_pattern) for mol in products
        )
        
        return has_sulfinamide_reactant and has_sulfinylimine_product
    
    def _detect_auxiliary_cleavage(self, reactants, products):
        """Detect cleavage of sulfinylimine to regenerate free amine/imine"""
        sulfinylimine_pattern = Chem.MolFromSmarts(self.sulfinylimine_pattern)
        sulfinamide_pattern = Chem.MolFromSmarts(self.sulfinamide_pattern)
        
        if not sulfinamide_pattern or not sulfinylimine_pattern:
            return False
        
        # Check if we have sulfinylimine in reactants
        has_sulfinylimine_reactant = any(
            mol.HasSubstructMatch(sulfinylimine_pattern) for mol in reactants
        )
        
        # Check if we regenerate sulfinamide or its derivatives in products
        # (indicating cleavage of the auxiliary)
        has_auxiliary_byproduct = any(
            mol.HasSubstructMatch(sulfinamide_pattern) for mol in products
        )
        
        return has_sulfinylimine_reactant and has_auxiliary_byproduct
