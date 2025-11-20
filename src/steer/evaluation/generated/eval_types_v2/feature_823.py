"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage cyclopropane ring formation.
    Detects cyclopropanation reactions and scores based on their timing in the route.
    """
    
    def __init__(self, config: Dict):
        self.ring_size = config["parameters"]["ring_size"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        
    def route_scoring(self, x) -> float:
        """
        Score based on depth fraction where cyclopropane formation occurs.
        For late-stage preference, lower depth fractions (closer to target) score higher.
        """
        if x < 0:
            return 0  # No cyclopropane formation found
        
        if self.timing == "late":
            # Reward late-stage formation (low depth fraction)
            return max(0, 1 - x) * 10
        elif self.timing == "early":
            # Reward early-stage formation (high depth fraction)
            return x * 10
        else:
            # Any timing is acceptable
            return 5
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves cyclopropane ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Count cyclopropane rings in products and reactants
            cyclopropane_pattern = Chem.MolFromSmarts("[C;R1]1[C;R1][C;R1]1")
            
            prod_cyclopropanes = len(prod_mol.GetSubstructMatches(cyclopropane_pattern))
            react_cyclopropanes = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                                    for mol in react_mols)
            
            # Check if cyclopropane ring was formed
            cyclopropane_formed = prod_cyclopropanes > react_cyclopropanes
            
            # Additional check for cyclopropanation method if specified
            if cyclopropane_formed and self.formation_method == "cyclopropanation":
                return self._detect_cyclopropanation_reaction(react_mols, prod_mol)
            
            return cyclopropane_formed
            
        except Exception:
            return False
    
    def _detect_cyclopropanation_reaction(self, reactants, product):
        """
        Detect specific cyclopropanation reaction patterns.
        Looks for alkene + carbene/carbenoid -> cyclopropane transformation.
        """
        # Check for alkene in reactants
        alkene_pattern = Chem.MolFromSmarts("C=C")
        has_alkene = any(mol.HasSubstructMatch(alkene_pattern) for mol in reactants)
        
        if not has_alkene:
            return False
        
        # Check for potential carbene sources (diazo compounds, etc.)
        carbene_sources = [
            Chem.MolFromSmarts("C=[N+]=[N-]"),  # Diazo compound
            Chem.MolFromSmarts("[CH2]"),         # Methylene (diazomethane)
            Chem.MolFromSmarts("C(Br)(Br)"),    # Dibromocarbene source
            Chem.MolFromSmarts("C(Cl)(Cl)")     # Dichlorocarbene source
        ]
        
        has_carbene_source = any(
            any(mol.HasSubstructMatch(pattern) for mol in reactants)
            for pattern in carbene_sources if pattern is not None
        )
        
        return has_carbene_source
