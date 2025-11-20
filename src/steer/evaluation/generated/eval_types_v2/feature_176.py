"""Generated evaluation code for: Anhydro bridge rearrangement to glycoside"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AnhydroBridgeRearrangement(BaseScoring):
    """
    Evaluates synthesis routes for anhydro bridge rearrangement to glycoside formation.
    Detects reactions that break anhydro linkages and form N-glycosidic bonds.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
        
        # SMARTS patterns for anhydro bridges and N-glycosides
        self.anhydro_pattern = "[CH1,CH2]-O-[CH1,CH2]"  # Simple anhydro bridge
        self.n_glycoside_pattern = "[CH1]-N-[*]"  # N-glycosidic bond
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Rearrangement doesn't happen
        else:
            if self.condition_type == "bool":
                return 1  # Found the rearrangement
            else:
                # Earlier rearrangement is typically better for synthetic strategy
                return 1 - x
    
    def hit_condition(self, d):
        """Check if this reaction involves anhydro bridge rearrangement to N-glycoside"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for anhydro bridge in reactants
            anhydro_pattern_mol = Chem.MolFromSmarts(self.anhydro_pattern)
            has_anhydro_reactant = any(mol.HasSubstructMatch(anhydro_pattern_mol) for mol in reactants)
            
            # Check for N-glycoside in products
            n_glycoside_pattern_mol = Chem.MolFromSmarts(self.n_glycoside_pattern)
            has_n_glycoside_product = any(mol.HasSubstructMatch(n_glycoside_pattern_mol) for mol in products)
            
            # Check that anhydro bridge is broken (less anhydro bridges in products than reactants)
            anhydro_count_reactants = sum(len(mol.GetSubstructMatches(anhydro_pattern_mol)) for mol in reactants)
            anhydro_count_products = sum(len(mol.GetSubstructMatches(anhydro_pattern_mol)) for mol in products)
            
            anhydro_broken = anhydro_count_products < anhydro_count_reactants
            
            # Additional check for rearrangement: look for ring opening/closing patterns
            ring_change = self._detect_ring_change(reactants, products)
            
            return has_anhydro_reactant and has_n_glycoside_product and anhydro_broken and ring_change
            
        except Exception:
            return False
    
    def _detect_ring_change(self, reactants, products):
        """Detect if there's a significant change in ring systems (typical for rearrangements)"""
        try:
            # Count ring atoms in reactants vs products
            reactant_ring_atoms = sum(sum(1 for atom in mol.GetAtoms() if atom.IsInRing()) for mol in reactants)
            product_ring_atoms = sum(sum(1 for atom in mol.GetAtoms() if atom.IsInRing()) for mol in products)
            
            # Look for ring opening/closing or significant ring structure change
            ring_atom_change = abs(reactant_ring_atoms - product_ring_atoms)
            
            # Also check for changes in ring count
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
            product_rings = sum(mol.GetRingInfo().NumRings() for mol in products)
            ring_count_change = abs(reactant_rings - product_rings)
            
            return ring_atom_change > 0 or ring_count_change > 0
            
        except Exception:
            return False
