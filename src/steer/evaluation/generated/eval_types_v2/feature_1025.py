"""Generated evaluation code for: Nitrene C-H insertion for carbazole formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitreneInsertionCarbazole(BaseScoring):
    """
    Evaluates synthesis routes for nitrene C-H insertion leading to carbazole formation.
    Checks for the presence of azide-to-nitrene conversion followed by intramolecular
    C-H insertion to form fused ring systems like carbazole.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # Define substrate pattern (azide precursor)
        self.substrate_pattern = Chem.MolFromSmarts("[N+]=[N-]=[N]c1ccccc1")
        
        # Define product pattern (carbazole-like fused ring)
        self.product_pattern = Chem.MolFromSmarts("c1cc2[nH]c3ccccc3c2cc1")
        
        # Additional pattern for detecting nitrene intermediate characteristics
        self.nitrene_context_pattern = Chem.MolFromSmarts("[nH]c1ccccc1")
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
            else:
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth) / 10.0)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a nitrene C-H insertion for carbazole formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0].strip()
            reactants = rxn_parts[1].strip()
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products)
            if not product_mol:
                return False
            
            reactant_mols = []
            for r_smiles in reactants.split("."):
                r_mol = Chem.MolFromSmiles(r_smiles.strip())
                if r_mol:
                    reactant_mols.append(r_mol)
            
            if not reactant_mols:
                return False
            
            # Check if product contains carbazole-like structure
            has_carbazole_product = product_mol.HasSubstructMatch(self.product_pattern)
            if not has_carbazole_product:
                return False
            
            # Check if any reactant contains azide precursor
            has_azide_reactant = any(mol.HasSubstructMatch(self.substrate_pattern) for mol in reactant_mols)
            
            # Alternative check: look for nitrene context in reactants
            has_nitrene_context = any(mol.HasSubstructMatch(self.nitrene_context_pattern) for mol in reactant_mols)
            
            # Check for ring formation (reactant has fewer rings than product)
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactant_mols)
            product_rings = product_mol.GetRingInfo().NumRings()
            
            ring_formation = product_rings > reactant_rings
            
            # Condition met if we have carbazole product and either azide reactant or nitrene context
            # plus evidence of ring formation
            return has_carbazole_product and (has_azide_reactant or has_nitrene_context) and ring_formation
            
        except Exception:
            return False
