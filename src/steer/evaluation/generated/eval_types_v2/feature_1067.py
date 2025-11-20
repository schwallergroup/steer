"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are coupled
    via amide formation at a specific step depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_step = config["coupling_step"]
        self.coupling_reaction = config["coupling_reaction"]
        
        # SMARTS patterns for amide formation reactants
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        self.amine_pattern = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
        self.acid_chloride_pattern = Chem.MolFromSmarts("[CX3](=O)[Cl]")
        self.ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H0]")
        
        # SMARTS pattern for amide product
        self.amide_pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't occur
        
        # Score based on how close the coupling occurs to target step
        target_depth_fraction = self.coupling_step / 10.0  # Normalize to 0-1
        
        if abs(x - target_depth_fraction) < 0.1:  # Within 10% of target
            return 10
        elif abs(x - target_depth_fraction) < 0.2:  # Within 20% of target
            return 7
        else:
            # Penalize based on distance from target
            penalty = abs(x - target_depth_fraction) * 10
            return max(0, 5 - penalty)

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent amide formation
        between two fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or len(reactant_mols) != self.fragment_count:
                return False
                
            # Check if product contains amide bond
            if not product_mol.HasSubstructMatch(self.amide_pattern):
                return False
                
            # Check if reactants are appropriate for amide formation
            has_carboxyl_component = False
            has_amine_component = False
            
            for reactant in reactant_mols:
                if not reactant:
                    continue
                    
                # Check for carboxylic acid derivatives
                if (reactant.HasSubstructMatch(self.carboxylic_acid_pattern) or
                    reactant.HasSubstructMatch(self.acid_chloride_pattern) or
                    reactant.HasSubstructMatch(self.ester_pattern)):
                    has_carboxyl_component = True
                    
                # Check for amine component
                if reactant.HasSubstructMatch(self.amine_pattern):
                    has_amine_component = True
                    
            return has_carboxyl_component and has_amine_component
            
        except Exception:
            return False
