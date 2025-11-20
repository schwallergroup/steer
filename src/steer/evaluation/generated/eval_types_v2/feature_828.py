"""Generated evaluation code for: Cyclopropanation using diazomethane reagent"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclopropanationDiazomethane(BaseScoring):
    """
    Evaluates synthesis routes for cyclopropanation reactions using diazomethane reagent.
    
    This class detects the formation of cyclopropane rings through diazomethane-mediated
    cyclopropanation reactions, which are characterized by the addition of CH2 across
    alkene bonds using diazomethane as the carbene source.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to 0-10 score.
        Earlier cyclopropanation is generally preferred due to diazomethane's reactivity.
        """
        if x < 0:
            return 0  # Condition not met
        else:
            return 1 - x  # Earlier stage is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents cyclopropanation using diazomethane.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for diazomethane reagent (C[N+]#[N-] or diazomethane pattern)
            diazomethane_patterns = [
                "[CH2][N+]#[N-]",  # Diazomethane
                "C[N+]#[N-]",      # General diazo compound
                "[CH2][N]=[N]"     # Alternative representation
            ]
            
            has_diazo_reagent = False
            for reactant in reactants:
                for pattern in diazomethane_patterns:
                    try:
                        pattern_mol = Chem.MolFromSmarts(pattern)
                        if pattern_mol and reactant.HasSubstructMatch(pattern_mol):
                            has_diazo_reagent = True
                            break
                    except:
                        continue
                if has_diazo_reagent:
                    break
            
            if not has_diazo_reagent:
                return False
            
            # Check for cyclopropane formation
            # Look for cyclopropane rings in products that weren't in reactants
            cyclopropane_pattern = Chem.MolFromSmarts("[CH2]1[CH2][CH2]1")  # Cyclopropane ring
            cyclopropyl_patterns = [
                "[CH2]1[CH2][CH2]1",     # Simple cyclopropane
                "[CH]1[CH2][CH2]1",      # Substituted cyclopropane
                "[C]1[CH2][CH2]1",       # Di-substituted cyclopropane
                "[C]1[CH][CH2]1",        # Tri-substituted cyclopropane
                "[C]1[C][CH2]1"          # Fully substituted cyclopropane
            ]
            
            # Count cyclopropane rings in reactants vs products
            reactant_cyclopropanes = 0
            product_cyclopropanes = 0
            
            for pattern_smarts in cyclopropyl_patterns:
                try:
                    pattern_mol = Chem.MolFromSmarts(pattern_smarts)
                    if pattern_mol:
                        for reactant in reactants:
                            reactant_cyclopropanes += len(reactant.GetSubstructMatches(pattern_mol))
                        for product in products:
                            product_cyclopropanes += len(product.GetSubstructMatches(pattern_mol))
                except:
                    continue
            
            # Cyclopropanation should increase the number of cyclopropane rings
            return product_cyclopropanes > reactant_cyclopropanes
            
        except Exception:
            return False
