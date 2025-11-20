"""Generated evaluation code for: Late intramolecular cyclization for tetracyclic core"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIntramolecularCyclization(BaseScoring):
    """
    Evaluates routes for late-stage intramolecular cyclization forming tetracyclic cores,
    specifically looking for Buchwald-Hartwig C-N coupling reactions that create
    complex polycyclic scaffolds in the final steps.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclization doesn't occur
        else:
            # Reward late-stage cyclization (higher depth fraction = later in route)
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Exponential reward for later cyclization
                return min(10, x * 10)
    
    def hit_condition(self, d) -> bool:
        """Check if reaction is an intramolecular cyclization forming tetracyclic core"""
        metadata = d.get("metadata", {})
        
        # Check if it's a Buchwald-Hartwig reaction
        policy_name = metadata.get("policy_name", "")
        if "buchwald" not in policy_name.lower() and "hartwig" not in policy_name.lower():
            # Also check reaction SMARTS or other identifiers for C-N coupling
            rxn_smiles = metadata.get("mapped_reaction_smiles", "")
            if not self._is_cn_coupling_reaction(rxn_smiles):
                return False
        
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactants):
                return False
                
            # Check if reaction is intramolecular (single reactant forms cyclic product)
            if len(reactants) != 1:
                return False
                
            reactant_mol = reactants[0]
            
            # Count rings in reactant vs product
            reactant_rings = self._count_ring_systems(reactant_mol)
            product_rings = self._count_ring_systems(product_mol)
            
            # Check if we formed exactly one new ring system
            if product_rings - reactant_rings != 1:
                return False
                
            # Check if product has tetracyclic core (4+ fused rings)
            if not self._has_tetracyclic_core(product_mol):
                return False
                
            # Verify intramolecular C-N bond formation occurred
            return self._verify_cn_cyclization(reactant_mol, product_mol)
            
        except Exception:
            return False
    
    def _is_cn_coupling_reaction(self, rxn_smiles: str) -> bool:
        """Check if reaction involves C-N coupling based on SMILES pattern"""
        if not rxn_smiles:
            return False
        # Look for patterns indicating C-N bond formation
        return any(indicator in rxn_smiles.lower() for indicator in 
                  ['[n]', '[nh]', 'n1', 'nc', 'cn'])
    
    def _count_ring_systems(self, mol) -> int:
        """Count number of separate ring systems in molecule"""
        if not mol:
            return 0
        ring_info = mol.GetRingInfo()
        return ring_info.NumRings()
    
    def _has_tetracyclic_core(self, mol) -> bool:
        """Check if molecule contains a tetracyclic (4+ ring) fused system"""
        if not mol:
            return False
            
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() < 4:
            return False
            
        # Check for fused ring systems by examining shared atoms
        rings = ring_info.AtomRings()
        if len(rings) < 4:
            return False
            
        # Find the largest fused ring system
        fused_system = self._find_largest_fused_system(rings)
        return len(fused_system) >= 4
    
    def _find_largest_fused_system(self, rings) -> list:
        """Find the largest set of fused rings"""
        if not rings:
            return []
            
        fused_systems = []
        for i, ring1 in enumerate(rings):
            current_system = [i]
            ring1_atoms = set(ring1)
            
            for j, ring2 in enumerate(rings):
                if i != j:
                    ring2_atoms = set(ring2)
                    # Rings are fused if they share atoms
                    if ring1_atoms & ring2_atoms:
                        if j not in current_system:
                            current_system.append(j)
                            
            fused_systems.append(current_system)
            
        # Return the largest fused system
        return max(fused_systems, key=len) if fused_systems else []
    
    def _verify_cn_cyclization(self, reactant_mol, product_mol) -> bool:
        """Verify that a C-N bond was formed to create the cycle"""
        if not reactant_mol or not product_mol:
            return False
            
        # Count C-N bonds in reactant vs product
        reactant_cn_bonds = self._count_cn_bonds(reactant_mol)
        product_cn_bonds = self._count_cn_bonds(product_mol)
        
        # Should have formed at least one new C-N bond
        return product_cn_bonds > reactant_cn_bonds
    
    def _count_cn_bonds(self, mol) -> int:
        """Count C-N bonds in molecule"""
        if not mol:
            return 0
            
        cn_count = 0
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or
                (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C')):
                cn_count += 1
                
        return cn_count
