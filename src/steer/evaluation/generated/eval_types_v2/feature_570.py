"""Generated evaluation code for: Late stage Corey-Chaykovsky cyclopropane formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CoreyChaykovsky(BaseScoring):
    """
    Evaluates late-stage Corey-Chaykovsky cyclopropane formation.
    Checks for cyclopropane ring formation via sulfoxonium ylide cyclopropanation.
    """
    
    def __init__(self, config: Dict):
        self.ring_size = config.get("ring_size", 3)
        self.ring_count = config.get("ring_count", 1)
        self.timing = config.get("timing", "late")
        
        # SMARTS pattern for sulfoxonium ylide (Me2S+CH2-)
        self.sulfoxonium_pattern = Chem.MolFromSmarts("[S+]([CH3])([CH3])[CH2-]")
        # Pattern for cyclopropane formation
        self.cyclopropane_pattern = Chem.MolFromSmarts("C1CC1")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing == "late":
            # Late-stage formation is preferred - penalize early occurrence
            return max(0, 10 * (1 - x))  # Higher score for later depth (higher x)
        else:
            # Early-stage formation preferred
            return max(0, 10 * x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents Corey-Chaykovsky cyclopropane formation
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for sulfoxonium ylide in reactants
            has_sulfoxonium = any(mol.HasSubstructMatch(self.sulfoxonium_pattern) for mol in reactants)
            
            # Count cyclopropane rings in reactants vs products
            reactant_cyclopropanes = sum(self._count_cyclopropanes(mol) for mol in reactants)
            product_cyclopropanes = sum(self._count_cyclopropanes(mol) for mol in products)
            
            # Check if cyclopropane rings are formed (increase from reactants to products)
            cyclopropanes_formed = product_cyclopropanes - reactant_cyclopropanes
            
            # Must have sulfoxonium ylide and form the expected number of cyclopropane rings
            return has_sulfoxonium and cyclopropanes_formed >= self.ring_count
            
        except Exception:
            return False
    
    def _count_cyclopropanes(self, mol) -> int:
        """Count the number of cyclopropane rings in a molecule"""
        if mol is None:
            return 0
        
        # Find all 3-membered rings
        ring_info = mol.GetRingInfo()
        cyclopropane_count = 0
        
        for ring in ring_info.AtomRings():
            if len(ring) == self.ring_size:
                # Verify it's a cyclopropane (all carbons)
                ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
                if all(atom.GetSymbol() == 'C' for atom in ring_atoms):
                    cyclopropane_count += 1
        
        return cyclopropane_count
