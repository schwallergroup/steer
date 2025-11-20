"""Generated evaluation code for: Convergent synthesis via amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use amide coupling reactions.
    Checks if an amide formation reaction occurs at a specified depth with
    the required number of fragments.
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config["parameters"]["coupling_depth"]
        self.required_fragments = config["parameters"]["fragment_count"]
        
        # SMARTS patterns for amide bond formation
        self.amide_patterns = [
            "[C:1](=[O:2])[N:3]",  # Basic amide pattern
            "[C:1](=[O:2])[NH:3]",  # Primary amide
            "[C:1](=[O:2])[NH2:3]"  # Unsubstituted amide
        ]
        
        # Common amide coupling reaction patterns (acid + amine -> amide)
        self.coupling_precursors = [
            ("[C:1](=[O:2])[OH]", "[N:3]"),  # Carboxylic acid + amine
            ("[C:1](=[O:2])[Cl]", "[N:3]"),  # Acid chloride + amine
            ("[C:1](=[O:2])[O][C](=[O])[C]", "[N:3]")  # Anhydride + amine
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        
        # Score based on how close the coupling depth is to target
        depth_score = max(0, 10 - abs(x - self.target_depth))
        return depth_score

    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents an amide coupling reaction."""
        metadata = d.get("metadata", {})
        
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1].split(".")
        
        # Check if we have the required number of reactant fragments
        if len(reactants_smiles) != self.required_fragments:
            return False
            
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if product contains amide bond
            has_amide = any(product_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                          for pattern in self.amide_patterns)
            
            if not has_amide:
                return False
                
            # Check if reactants match amide coupling precursors
            return self._check_amide_coupling_precursors(reactant_mols, product_mol)
            
        except Exception:
            return False

    def _check_amide_coupling_precursors(self, reactants, product):
        """Check if reactants are typical amide coupling partners."""
        if len(reactants) != 2:
            return False
            
        mol1, mol2 = reactants
        
        for acid_pattern, amine_pattern in self.coupling_precursors:
            acid_smarts = Chem.MolFromSmarts(acid_pattern)
            amine_smarts = Chem.MolFromSmarts(amine_pattern)
            
            # Check both orientations (mol1=acid, mol2=amine) and (mol1=amine, mol2=acid)
            if ((mol1.HasSubstructMatch(acid_smarts) and mol2.HasSubstructMatch(amine_smarts)) or
                (mol1.HasSubstructMatch(amine_smarts) and mol2.HasSubstructMatch(acid_smarts))):
                return True
                
        return False
