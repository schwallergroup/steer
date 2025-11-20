"""Generated evaluation code for: Late stage intramolecular ring cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageIntramolecularRingCyclization(BaseScoring):
    """
    Evaluates late-stage intramolecular ring cyclization reactions.
    Specifically looks for six-membered ring formation through intramolecular cyclization.
    """
    
    def __init__(self, config: Dict):
        self.stage = config["parameters"]["stage"]  # "late"
        self.ring_type = config["parameters"]["ring_type"]  # "six_membered" 
        self.mechanism = config["parameters"]["mechanism"]  # "intramolecular_cyclization"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclization doesn't happen
        else:
            # Late-stage cyclization is better - higher depth fraction means later
            return x * 10  # Convert to 0-10 scale, favoring late stage
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves intramolecular six-membered ring formation.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
        
        if None in reactant_mols or None in product_mols:
            return False
            
        # Check for intramolecular cyclization (single reactant -> single product with new ring)
        if len(reactant_mols) != 1 or len(product_mols) != 1:
            return False
            
        reactant = reactant_mols[0]
        product = product_mols[0]
        
        # Count six-membered rings in reactant vs product
        reactant_six_rings = self._count_six_membered_rings(reactant)
        product_six_rings = self._count_six_membered_rings(product)
        
        # Check if a new six-membered ring was formed
        if product_six_rings > reactant_six_rings:
            # Verify it's truly intramolecular by checking atom map conservation
            return self._verify_intramolecular_cyclization(reactant, product)
            
        return False
    
    def _count_six_membered_rings(self, mol):
        """Count the number of six-membered rings in a molecule."""
        ring_info = mol.GetRingInfo()
        six_rings = 0
        for ring in ring_info.AtomRings():
            if len(ring) == 6:
                six_rings += 1
        return six_rings
    
    def _verify_intramolecular_cyclization(self, reactant, product):
        """
        Verify this is an intramolecular cyclization by checking that:
        1. Atom count is conserved (or only differs by small molecules like H2O)
        2. New bonds are formed between atoms that existed in reactant
        """
        reactant_atom_maps = set(atom.GetAtomMapNum() for atom in reactant.GetAtoms() if atom.GetAtomMapNum() > 0)
        product_atom_maps = set(atom.GetAtomMapNum() for atom in product.GetAtoms() if atom.GetAtomMapNum() > 0)
        
        # Check if the mapped atoms are conserved (intramolecular)
        if len(reactant_atom_maps) > 0 and len(product_atom_maps) > 0:
            # Most mapped atoms should be conserved in intramolecular cyclization
            conserved_atoms = reactant_atom_maps.intersection(product_atom_maps)
            return len(conserved_atoms) >= min(len(reactant_atom_maps), len(product_atom_maps)) * 0.8
        
        # Fallback: check atom count conservation (allowing for loss of small molecules)
        reactant_heavy_atoms = reactant.GetNumHeavyAtoms()
        product_heavy_atoms = product.GetNumHeavyAtoms()
        
        # Allow for loss of up to 3 heavy atoms (e.g., H2O, CO2, etc.)
        return abs(reactant_heavy_atoms - product_heavy_atoms) <= 3
