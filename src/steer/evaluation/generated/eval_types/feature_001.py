"""Generated evaluation code for: Ring-closing metathesis for bicyclic core formation"""

from typing import Dict, Tuple
from rdkit import Chem
# Import standalone base classes
import sys, os
_base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base_dir)
from standalone_base import BaseScoring, MultiRxnCondBase

class RingClosingMetathesis(BaseScoring):
    """
    Evaluates synthesis routes for the presence of ring-closing metathesis (RCM) reactions
    for bicyclic core formation. RCM is typically identified by the formation of C=C bonds
    within cyclic structures from diene precursors.
    """
    
    def __init__(self, config: Dict):
        self.required = config["parameters"].get("required", True)
        # Common RCM catalyst fragments that might appear in mapped reactions
        self.rcm_catalysts = [
            "[Ru]",  # Ruthenium-based catalysts (Grubbs, Hoveyda-Grubbs)
            "C=C[Ru]",  # Ruthenium carbene
            "[Mo]",  # Molybdenum-based catalysts
        ]
    
    def route_scoring(self, x) -> float:
        if self.required:
            if x < 0:  # RCM not found
                return 0
            else:
                return 10 - (x * 2)  # Earlier RCM is better (higher score)
        else:
            if x < 0:  # RCM not found - this is acceptable
                return 10
            else:
                return 5  # Neutral score if RCM found but not required
    
    def hit_condition(self, d) -> bool:
        """
        Detects RCM by looking for:
        1. Formation of C=C bond in a ring
        2. Presence of RCM catalyst indicators
        3. Characteristic diene -> cyclic alkene transformation
        """
        metadata = d.get("metadata", {})
        
        # Check for RCM catalyst mentions in reaction metadata
        if "policy_name" in metadata:
            policy = metadata["policy_name"].lower()
            if "metathesis" in policy or "rcm" in policy:
                return True
        
        # Analyze the mapped reaction SMILES
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            products_smiles, reactants_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules and small molecules (likely catalysts/byproducts)
            reactants = [mol for mol in reactants if mol and mol.GetNumAtoms() > 5]
            products = [mol for mol in products if mol and mol.GetNumAtoms() > 5]
            
            if not reactants or not products:
                return False
            
            # Check for RCM pattern: diene reactant -> cyclic alkene product
            return self._detect_rcm_transformation(reactants, products)
            
        except:
            return False
    
    def _detect_rcm_transformation(self, reactants, products) -> bool:
        """
        Detects RCM by analyzing the transformation pattern:
        - Reactant should have two C=C bonds (diene)
        - Product should have one C=C bond in a ring (cyclic alkene)
        - Overall atom count should decrease (ethylene loss)
        """
        for reactant in reactants:
            for product in products:
                if self._is_rcm_pair(reactant, product):
                    return True
        return False
    
    def _is_rcm_pair(self, reactant, product) -> bool:
        """
        Checks if a reactant-product pair represents an RCM transformation
        """
        # Check atom count decrease (typical ethylene loss in RCM)
        if reactant.GetNumAtoms() <= product.GetNumAtoms():
            return False
        
        # Count terminal alkenes in reactant (common RCM substrates)
        terminal_alkene_pattern = Chem.MolFromSmarts("C=C")
        reactant_alkenes = len(reactant.GetSubstructMatches(terminal_alkene_pattern))
        
        if reactant_alkenes < 2:  # Need at least 2 alkenes for RCM
            return False
        
        # Check for ring formation in product
        product_rings = product.GetRingInfo().NumRings()
        reactant_rings = reactant.GetRingInfo().NumRings()
        
        # RCM should form at least one new ring
        if product_rings <= reactant_rings:
            return False
        
        # Check for preserved alkene in product (within a ring)
        # Extract all ring atom indices in product
        ring_info = product.GetRingInfo()
        ring_atom_idxs = ring_info.AtomRings()
        cyclic_alkene_count = 0
        for ring in ring_atom_idxs:
            # For each pair of adjacent atoms in the ring, check if it's a double bond
            ring_bonds = []
            num_ring_atoms = len(ring)
            for i in range(num_ring_atoms):
                idx1 = ring[i]
                idx2 = ring[(i+1) % num_ring_atoms]
                bond = product.GetBondBetweenAtoms(idx1, idx2)
                if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    cyclic_alkene_count += 1
        product_cyclic_alkenes = cyclic_alkene_count
        
        return product_cyclic_alkenes > 0
