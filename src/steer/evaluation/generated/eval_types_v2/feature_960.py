"""Generated evaluation code for: Late stage pyridine ring formation via Skraup"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSkraupPyridine(BaseScoring):
    """
    Checks for late-stage pyridine ring formation via Skraup reaction.
    
    The Skraup reaction forms pyridine rings from anilines and α,β-unsaturated 
    carbonyl compounds under harsh acidic conditions. This evaluator looks for
    pyridine ring formation and rewards it when it occurs late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ccncc1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.formation_method = config["parameters"]["formation_method"]  # "Skraup"
        self.pyridine_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyridine formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (closer to 1.0)
            else:
                return x  # Earlier formation is better
    
    def hit_condition(self, d):
        """
        Check if this reaction involves pyridine ring formation via Skraup-type conditions.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if product contains pyridine ring
            if not product_mol.HasSubstructMatch(self.pyridine_pattern):
                return False
                
            # Check if any reactant contains the pyridine ring (if so, it's not formation)
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(self.pyridine_pattern):
                    return False
                    
            # Check for Skraup-like reaction pattern
            # Look for aniline derivative in reactants (aromatic amine)
            aniline_pattern = Chem.MolFromSmarts("c1ccc(N)cc1")  # Basic aniline pattern
            aromatic_amine_pattern = Chem.MolFromSmarts("c1ccccc1N")  # More general aromatic amine
            
            has_aniline = any(reactant.HasSubstructMatch(aniline_pattern) or 
                            reactant.HasSubstructMatch(aromatic_amine_pattern) 
                            for reactant in reactant_mols)
            
            # Look for α,β-unsaturated carbonyl or related electrophile
            # Common patterns in Skraup reactions
            unsaturated_carbonyl = Chem.MolFromSmarts("C=CC=O")  # α,β-unsaturated aldehyde
            glycerol_pattern = Chem.MolFromSmarts("OCC(O)CO")  # Glycerol (classic Skraup)
            
            has_electrophile = any(reactant.HasSubstructMatch(unsaturated_carbonyl) or
                                 reactant.HasSubstructMatch(glycerol_pattern)
                                 for reactant in reactant_mols)
            
            # Return True if we have both components suggesting Skraup reaction
            return has_aniline and (has_electrophile or len(reactant_mols) >= 2)
            
        except Exception:
            return False
