"""Generated evaluation code for: Late stage macrocycle formation via RCM"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRCMMacrocycle(BaseScoring):
    """
    Evaluates whether a macrocycle (12-20 membered ring) is formed late in the synthesis 
    using ring-closing metathesis (RCM). Rewards late-stage formation of large rings 
    through metathesis reactions.
    """
    
    def __init__(self, config: Dict):
        self.min_ring_size = config["parameters"]["ring_size_range"][0]  # 12
        self.max_ring_size = config["parameters"]["ring_size_range"][1]  # 20
        self.formation_method = config["parameters"]["formation_method"]  # "RCM"
        self.timing = config["parameters"]["timing"]  # "late"
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score. For late-stage timing, reward higher depth values.
        x = -1 means condition not met, x >= 0 is depth fraction (0=early, 1=late)
        """
        if x < 0:
            return 0  # RCM macrocycle formation doesn't happen
        else:
            # For late-stage timing, higher depth is better (closer to 1.0)
            # Scale to 0-10 range with exponential reward for very late stages
            return 10 * (x ** 2)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves RCM macrocycle formation.
        Returns True if:
        1. Reaction involves metathesis (Ru catalyst or alkene metathesis pattern)
        2. Forms a ring in the size range 12-20
        3. Ring formation occurs (product has more rings than largest reactant)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if this is a metathesis reaction
            if not self._is_metathesis_reaction(reactant_mols, product_mol):
                return False
            
            # Check if macrocycle is formed
            return self._forms_macrocycle(reactant_mols, product_mol)
            
        except Exception:
            return False
    
    def _is_metathesis_reaction(self, reactants, product):
        """
        Detect metathesis by checking for:
        1. Presence of alkenes in reactants and product
        2. Change in alkene connectivity pattern consistent with metathesis
        """
        # Simple heuristic: look for alkene groups and connectivity changes
        alkene_pattern = Chem.MolFromSmarts("C=C")
        
        # Count alkenes in reactants vs product
        reactant_alkenes = sum(len(mol.GetSubstructMatches(alkene_pattern)) for mol in reactants)
        product_alkenes = len(product.GetSubstructMatches(alkene_pattern))
        
        # In RCM, we typically go from 2 alkenes to 1 (intramolecular) 
        # or maintain alkene count but change connectivity
        if reactant_alkenes >= 2 and product_alkenes >= 1:
            return True
            
        # Additional check: look for terminal alkenes in reactants
        terminal_alkene = Chem.MolFromSmarts("C=C([H])[H]")
        terminal_count = sum(len(mol.GetSubstructMatches(terminal_alkene)) for mol in reactants)
        
        return terminal_count >= 2
    
    def _forms_macrocycle(self, reactants, product):
        """
        Check if a macrocycle in the target size range is formed.
        Compare ring systems in reactants vs product.
        """
        # Get ring info for product
        product_rings = product.GetRingInfo()
        product_ring_sizes = [len(ring) for ring in product_rings.AtomRings()]
        
        # Check if product has a ring in our target range
        has_target_macrocycle = any(self.min_ring_size <= size <= self.max_ring_size 
                                   for size in product_ring_sizes)
        
        if not has_target_macrocycle:
            return False
        
        # Check that this ring is newly formed (not present in reactants)
        max_reactant_ring = 0
        for reactant in reactants:
            reactant_rings = reactant.GetRingInfo()
            if reactant_rings.AtomRings():
                reactant_ring_sizes = [len(ring) for ring in reactant_rings.AtomRings()]
                max_reactant_ring = max(max_reactant_ring, max(reactant_ring_sizes))
        
        # New macrocycle formed if product has larger ring than any reactant
        max_product_ring = max(product_ring_sizes) if product_ring_sizes else 0
        
        return (max_product_ring >= self.min_ring_size and 
                max_product_ring <= self.max_ring_size and
                max_product_ring > max_reactant_ring)
