"""Generated evaluation code for: Early spiro-cyclopropane ring formation via cyclopropanation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySpiroCyclopropaneFormation(BaseScoring):
    """
    Checks for early spiro-cyclopropane ring formation via cyclopropanation.
    Scores routes based on when cyclopropane rings are formed through cyclopropanation reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.stage_threshold = config["parameters"]["stage_threshold"]  # 0.3
        self.formation_type = config["parameters"]["formation_type"]  # "cyclopropanation"
        self.cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropanation doesn't happen
        
        if self.timing == "early":
            # Early formation is better - higher score for smaller depth fractions
            if x <= self.stage_threshold:
                return 10  # Perfect early timing
            else:
                # Linear decay from 10 to 0 as depth increases beyond threshold
                return max(0, 10 * (1 - x) / (1 - self.stage_threshold))
        else:
            # For other timing preferences, use standard scoring
            return 10 * (1 - x)
    
    def hit_condition(self, d):
        """
        Checks if this reaction forms a cyclopropane ring via cyclopropanation.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse products and reactants
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Count cyclopropane rings in products vs reactants
            prod_cyclopropane_count = len(prod_mol.GetSubstructMatches(self.cyclopropane_pattern))
            react_cyclopropane_count = sum(
                len(mol.GetSubstructMatches(self.cyclopropane_pattern)) for mol in react_mols
            )
            
            # Check if cyclopropane rings were formed
            cyclopropane_formed = prod_cyclopropane_count > react_cyclopropane_count
            
            if not cyclopropane_formed:
                return False
            
            # Check if this is a cyclopropanation reaction type
            return self._is_cyclopropanation_reaction(prod_mol, react_mols)
            
        except Exception:
            return False
    
    def _is_cyclopropanation_reaction(self, product, reactants):
        """
        Determines if this is a cyclopropanation reaction based on reaction patterns.
        """
        # Look for common cyclopropanation patterns:
        # 1. Alkene + carbene/carbenoid -> cyclopropane
        # 2. Simmons-Smith type reactions
        # 3. Diazo compound + alkene
        
        alkene_pattern = Chem.MolFromSmarts("C=C")
        diazo_pattern = Chem.MolFromSmarts("C=[N+]=[N-]")
        zinc_carbenoid_pattern = Chem.MolFromSmarts("[CH2][Zn]")
        
        # Check if reactants contain alkene and cyclopropanation reagents
        has_alkene = any(mol.HasSubstructMatch(alkene_pattern) for mol in reactants)
        has_diazo = any(mol.HasSubstructMatch(diazo_pattern) for mol in reactants)
        has_carbenoid = any(mol.HasSubstructMatch(zinc_carbenoid_pattern) for mol in reactants)
        
        # Simple heuristic: if we have an alkene and potential cyclopropanation reagent
        if has_alkene and (has_diazo or has_carbenoid):
            return True
        
        # Additional check: look for characteristic atom count changes
        # Cyclopropanation typically involves C-C bond formation without atom loss
        total_reactant_atoms = sum(mol.GetNumAtoms() for mol in reactants)
        product_atoms = product.GetNumAtoms()
        
        # Allow for small atom count differences due to leaving groups
        atom_diff = abs(total_reactant_atoms - product_atoms)
        
        return has_alkene and atom_diff <= 4  # Flexible threshold for leaving groups
