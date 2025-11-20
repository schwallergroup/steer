"""Generated evaluation code for: Convergent synthesis via Suzuki coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiCoupling(BaseScoring):
    """
    Evaluates synthesis routes for convergent assembly via Suzuki-Miyaura coupling.
    Detects formation of C-C bonds between organoborane and organohalide fragments
    and rewards mid-stage timing for convergent synthesis strategy.
    """
    
    def __init__(self, config: Dict):
        self.target_timing = config.get("timing", "mid_stage")
        self.min_fragments = config.get("fragments", 2)
        self.bond_type = config.get("bond_type", "C-C")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        
        if self.target_timing == "mid_stage":
            # Reward mid-stage coupling (depth fraction around 0.3-0.7)
            if 0.3 <= x <= 0.7:
                return 10
            elif x < 0.3:
                return 7 * (x / 0.3)  # Early stage penalty
            else:
                return 7 * ((1.0 - x) / 0.3)  # Late stage penalty
        elif self.target_timing == "early":
            return 10 * (1 - x)  # Earlier is better
        else:  # late_stage
            return 10 * x  # Later is better
    
    def hit_condition(self, d):
        """Check if reaction is a Suzuki-Miyaura coupling"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            products_smiles, reactants_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(products_smiles.strip())]
            
            # Filter out None molecules
            reactants = [r for r in reactants if r is not None]
            products = [p for p in products if p is not None]
            
            if len(reactants) < self.min_fragments or len(products) == 0:
                return False
                
            return self._is_suzuki_coupling(reactants, products[0])
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, reactants, product):
        """Detect Suzuki coupling by checking for organoborane and organohalide reactants"""
        
        # SMARTS patterns for Suzuki coupling partners
        organoborane_pattern = Chem.MolFromSmarts("[C,c]-B(-O)(-O)")  # Boronic acid/ester
        organoborane_alt = Chem.MolFromSmarts("[C,c]-B")  # Simple organoborane
        organohalide_pattern = Chem.MolFromSmarts("[C,c]-[Cl,Br,I]")  # Aryl/alkyl halide
        triflate_pattern = Chem.MolFromSmarts("[C,c]-OS(=O)(=O)C(F)(F)F")  # Triflate
        
        has_borane = False
        has_halide = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(organoborane_pattern) or \
               reactant.HasSubstructMatch(organoborane_alt):
                has_borane = True
            elif reactant.HasSubstructMatch(organohalide_pattern) or \
                 reactant.HasSubstructMatch(triflate_pattern):
                has_halide = True
                
        # Must have both coupling partners
        if not (has_borane and has_halide):
            return False
            
        # Check for C-C bond formation by comparing reactant and product structures
        return self._check_cc_bond_formation(reactants, product)
    
    def _check_cc_bond_formation(self, reactants, product):
        """Verify C-C bond formation between coupling partners"""
        try:
            # Count carbon atoms to ensure we're forming new C-C bonds
            reactant_carbons = sum(len([atom for atom in mol.GetAtoms() 
                                      if atom.GetSymbol() == 'C']) for mol in reactants)
            product_carbons = len([atom for atom in product.GetAtoms() 
                                 if atom.GetSymbol() == 'C'])
            
            # Should have similar carbon count (accounting for potential catalyst ligands)
            if abs(reactant_carbons - product_carbons) > 2:
                return False
                
            # Check that we don't have boron or halogen in major product
            # (they should be consumed in coupling)
            boron_atoms = len([atom for atom in product.GetAtoms() if atom.GetSymbol() == 'B'])
            halogen_atoms = len([atom for atom in product.GetAtoms() 
                               if atom.GetSymbol() in ['Cl', 'Br', 'I']])
            
            # Coupling should consume these groups (allow small amounts for workup byproducts)
            return boron_atoms <= 1 and halogen_atoms <= 1
            
        except Exception:
            return False
