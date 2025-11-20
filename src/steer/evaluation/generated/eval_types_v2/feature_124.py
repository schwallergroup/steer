"""Generated evaluation code for: Early pyrazolo-pyridine core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazoloPyridineFormation(BaseScoring):
    """
    Evaluates early pyrazolo-pyridine core formation in synthesis routes.
    Detects cyclocondensation reactions that form fused pyrazolo[1,5-a]pyridine systems.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")  # early, middle, late
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Core formation doesn't happen
        else:
            # Early formation is preferred (lower depth fraction = higher score)
            if self.timing_preference == "early":
                return 1 - x  # Score decreases with depth
            elif self.timing_preference == "late":
                return x  # Score increases with depth
            else:  # middle
                return 1 - abs(x - 0.5) * 2  # Peak score at middle depth
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction forms a pyrazolo[1,5-a]pyridine core via cyclocondensation.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn = metadata["mapped_reaction_smiles"].split(">>")
        if len(rxn) != 2:
            return False
            
        products = rxn[0]
        reactants = rxn[1].split(".")
        
        try:
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if product contains pyrazolo[1,5-a]pyridine core
            if not self._has_pyrazolopyridine_core(product_mol):
                return False
                
            # Check if this is a cyclocondensation (ring-forming reaction)
            # Product should have more rings than any individual reactant
            product_ring_count = Chem.rdMolDescriptors.CalcNumRings(product_mol)
            max_reactant_rings = max(Chem.rdMolDescriptors.CalcNumRings(mol) for mol in reactant_mols)
            
            # Must form at least one new ring
            if product_ring_count <= max_reactant_rings:
                return False
                
            # Check for typical cyclocondensation pattern:
            # Should involve formation of N-N and C-N bonds in fused system
            return self._is_cyclocondensation_pattern(product_mol, reactant_mols)
            
        except Exception:
            return False
    
    def _has_pyrazolopyridine_core(self, mol) -> bool:
        """Check if molecule contains pyrazolo[1,5-a]pyridine core structure."""
        # SMARTS pattern for pyrazolo[1,5-a]pyridine core
        # Matches the fused pyrazole-pyridine system
        pyrazolopyridine_pattern = "[#7]1[#7][#6][#6]2[#6][#6][#6][#7][#6]12"
        
        pattern_mol = Chem.MolFromSmarts(pyrazolopyridine_pattern)
        if pattern_mol is None:
            return False
            
        return mol.HasSubstructMatch(pattern_mol)
    
    def _is_cyclocondensation_pattern(self, product_mol, reactant_mols) -> bool:
        """
        Check if the reaction pattern matches cyclocondensation characteristics.
        """
        # Check for presence of nitrogen-containing reactants
        has_n_reactant = any(
            any(atom.GetSymbol() == 'N' for atom in mol.GetAtoms()) 
            for mol in reactant_mols
        )
        
        if not has_n_reactant:
            return False
            
        # Check for formation of heterocyclic system
        # Product should have fused rings with nitrogen atoms
        ring_info = product_mol.GetRingInfo()
        ring_atoms = ring_info.AtomRings()
        
        if len(ring_atoms) < 2:  # Must have at least 2 rings for fused system
            return False
            
        # Check for shared atoms between rings (fused system)
        has_fused_rings = False
        for i, ring1 in enumerate(ring_atoms):
            for ring2 in ring_atoms[i+1:]:
                if len(set(ring1) & set(ring2)) >= 2:  # Share at least 2 atoms
                    has_fused_rings = True
                    break
            if has_fused_rings:
                break
                
        return has_fused_rings
