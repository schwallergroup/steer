"""Generated evaluation code for: Early penicillin to cephalosporin ring expansion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyRingExpansion(BaseScoring):
    """
    Evaluates if ring expansion from 5-membered to 6-membered ring occurs early in synthesis.
    Detects penicillin-to-cephalosporin type transformations and checks timing.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"]["stage_threshold"]
        # SMARTS patterns for beta-lactam rings
        self.penicillin_pattern = "[#6]1[#6][#6][#7]1[#6](=[#8])"  # 4-membered beta-lactam with 5-membered thiazolidine
        self.cephalosporin_pattern = "[#6]1[#6][#6][#7]1[#6](=[#8])"  # 4-membered beta-lactam with 6-membered dihydrothiazine
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10). Early expansion gets higher score."""
        if x < 0:
            return 0  # Ring expansion doesn't occur
        
        if x <= self.stage_threshold:
            return 10  # Early expansion - maximum score
        else:
            # Linear decrease from 10 to 1 as depth increases beyond threshold
            score = max(1, 10 - (x - self.stage_threshold) * 12.86)
            return score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a ring expansion from 5- to 6-membered ring."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for ring expansion pattern
            return self._detect_ring_expansion(reactants, products)
            
        except Exception:
            return False
    
    def _detect_ring_expansion(self, reactants, products) -> bool:
        """Detect if ring expansion from 5 to 6 membered ring occurs."""
        # Count 5-membered and 6-membered rings in reactants and products
        reactant_5rings = sum(self._count_5membered_rings(mol) for mol in reactants)
        reactant_6rings = sum(self._count_6membered_rings(mol) for mol in reactants)
        
        product_5rings = sum(self._count_5membered_rings(mol) for mol in products)
        product_6rings = sum(self._count_6membered_rings(mol) for mol in products)
        
        # Ring expansion: decrease in 5-membered rings and increase in 6-membered rings
        five_ring_decrease = reactant_5rings > product_5rings
        six_ring_increase = product_6rings > reactant_6rings
        
        # Also check for beta-lactam context (penicillin/cephalosporin transformation)
        has_beta_lactam_context = self._has_beta_lactam_context(reactants + products)
        
        return five_ring_decrease and six_ring_increase and has_beta_lactam_context
    
    def _count_5membered_rings(self, mol) -> int:
        """Count 5-membered rings in molecule."""
        if mol is None:
            return 0
        ri = mol.GetRingInfo()
        return len([ring for ring in ri.AtomRings() if len(ring) == 5])
    
    def _count_6membered_rings(self, mol) -> int:
        """Count 6-membered rings in molecule."""
        if mol is None:
            return 0
        ri = mol.GetRingInfo()
        return len([ring for ring in ri.AtomRings() if len(ring) == 6])
    
    def _has_beta_lactam_context(self, molecules) -> bool:
        """Check if any molecule contains beta-lactam core structure."""
        beta_lactam_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#7]1[#6](=[#8])")
        if beta_lactam_pattern is None:
            return False
            
        for mol in molecules:
            if mol is not None and mol.HasSubstructMatch(beta_lactam_pattern):
                return True
        return False
