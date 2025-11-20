"""Generated evaluation code for: Late stage tandem cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTandemCyclization(BaseScoring):
    """
    Evaluates whether a tandem cyclization reaction occurs in the final stages of synthesis.
    Tandem cyclization involves multiple ring-forming reactions occurring in sequence,
    particularly Fischer indole formation followed by intramolecular cyclization.
    """
    
    def __init__(self, config: Dict):
        self.stage = config.get("parameters", {}).get("stage", "final")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Tandem cyclization doesn't occur
        
        if self.stage == "final":
            # Reward later stage cyclization more highly
            return (1 - x) * 10
        else:
            # For non-final stage, reward presence but less strongly
            return 5 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """
        Detect tandem cyclization by checking for:
        1. Formation of multiple rings in one reaction
        2. Indole formation patterns
        3. Intramolecular cyclization patterns
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Count rings in reactants vs products
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
            product_rings = sum(mol.GetRingInfo().NumRings() for mol in products)
            
            # Check if multiple rings are formed (tandem aspect)
            rings_formed = product_rings - reactant_rings
            if rings_formed < 2:
                return False
            
            # Check for Fischer indole formation pattern
            indole_pattern = Chem.MolFromSmarts("c1ccc2[nH]ccc2c1")  # Basic indole
            has_indole_formation = any(mol.HasSubstructMatch(indole_pattern) for mol in products)
            
            # Check for fused ring systems (7-membered rings fused to other systems)
            seven_ring_fused = Chem.MolFromSmarts("[R2]1~[R2]~[R2]~[R2]~[R2]~[R2]~[R2]1")  # 7-membered ring in fused system
            has_seven_fused = any(mol.HasSubstructMatch(seven_ring_fused) for mol in products)
            
            # Check for intramolecular cyclization indicators
            # Look for molecules that have become more constrained (higher ring count relative to size)
            intramolecular_cyclization = False
            for prod in products:
                if prod.GetNumAtoms() > 10:  # Reasonable size threshold
                    ring_density = prod.GetRingInfo().NumRings() / prod.GetNumAtoms()
                    if ring_density > 0.3:  # High ring density suggests complex cyclization
                        intramolecular_cyclization = True
                        break
            
            # Tandem cyclization criteria:
            # 1. Multiple rings formed AND
            # 2. Either indole formation OR seven-membered fused ring AND
            # 3. Evidence of intramolecular cyclization
            return (rings_formed >= 2 and 
                   (has_indole_formation or has_seven_fused) and 
                   intramolecular_cyclization)
            
        except Exception:
            return False
