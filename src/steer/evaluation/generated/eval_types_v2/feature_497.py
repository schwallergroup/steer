"""Generated evaluation code for: Late stage epoxide formation via ylide"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEpoxideFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage epoxide formation via ylide reactions.
    Detects epoxide ring formation using Corey-Chaykovsky reaction with trimethylsulfonium salt.
    """
    
    def __init__(self, config: Dict):
        self.epoxide_pattern = config.get("ring_smarts", "C1CO1")
        self.timing = config.get("timing", "late")
        self.reaction_type = config.get("reaction_type", "ylide_epoxidation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Epoxide formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later stage formation is better (closer to 1.0)
        else:
            return 1 if x >= 0 else 0  # Just presence/absence
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms an epoxide via ylide mechanism"""
        if not self._is_ylide_epoxidation(d):
            return False
            
        return self._forms_epoxide_ring(d)
    
    def _is_ylide_epoxidation(self, d) -> bool:
        """Check if reaction involves ylide-based epoxidation"""
        metadata = d.get("metadata", {})
        
        # Check for sulfonium ylide patterns in reagents/conditions
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        # Look for trimethylsulfonium or dimethylsulfonium patterns
        ylide_patterns = [
            "[S+]([CH3])([CH3])[CH2-]",  # Trimethylsulfonium ylide
            "[S+]([CH3])([CH3])C",       # Trimethylsulfonium salt
            "C[S+](C)C",                 # General sulfonium pattern
        ]
        
        for pattern in ylide_patterns:
            try:
                ylide_mol = Chem.MolFromSmarts(pattern)
                if ylide_mol:
                    reactants = rxn_smiles.split(">>")[1].split(".")
                    for reactant_smiles in reactants:
                        reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                        if reactant_mol and reactant_mol.HasSubstructMatch(ylide_mol):
                            return True
            except:
                continue
                
        return False
    
    def _forms_epoxide_ring(self, d) -> bool:
        """Check if epoxide ring is formed in this reaction"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Create epoxide pattern
            epoxide_mol = Chem.MolFromSmarts(self.epoxide_pattern)
            if not epoxide_mol:
                return False
                
            # Check if product has epoxide but reactants don't
            product_has_epoxide = product_mol.HasSubstructMatch(epoxide_mol)
            reactants_have_epoxide = any(r.HasSubstructMatch(epoxide_mol) for r in reactant_mols)
            
            return product_has_epoxide and not reactants_have_epoxide
            
        except Exception:
            return False
