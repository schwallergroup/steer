"""Generated evaluation code for: Azide-mediated stereoinversion for amine installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideStereoInversion(BaseScoring):
    """
    Evaluates routes for azide-mediated stereoinversion for amine installation.
    Detects azide substitution reactions that involve stereochemical inversion,
    typically from mesylate/tosylate displacement followed by reduction.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)
    
    def hit_condition(self, d):
        """Check if reaction involves azide substitution with stereoinversion potential"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".") if r.strip()]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check for azide introduction
            azide_pattern = Chem.MolFromSmarts("[N-]=[N+]=[N-]")  # Azide functional group
            azide_alt_pattern = Chem.MolFromSmarts("N=[N+]=[N-]")  # Alternative azide representation
            
            # Check if product contains azide
            has_azide_product = (product_mol.HasSubstructMatch(azide_pattern) or 
                               product_mol.HasSubstructMatch(azide_alt_pattern))
            
            if not has_azide_product:
                return False
                
            # Check if reactants contain leaving group (mesylate, tosylate, halide)
            leaving_groups = [
                Chem.MolFromSmarts("OS(=O)(=O)C"),  # Mesylate
                Chem.MolFromSmarts("OS(=O)(=O)c1ccc(C)cc1"),  # Tosylate
                Chem.MolFromSmarts("[C,c][Cl,Br,I]"),  # Halides
                Chem.MolFromSmarts("OS(=O)(=O)[C,c]")  # General sulfonate
            ]
            
            has_leaving_group = any(
                any(reactant.HasSubstructMatch(lg_pattern) for reactant in reactant_mols)
                for lg_pattern in leaving_groups
            )
            
            # Check for azide nucleophile in reactants
            azide_nucleophile_patterns = [
                Chem.MolFromSmarts("[Na+].[N-]=[N+]=[N-]"),  # Sodium azide
                Chem.MolFromSmarts("N=[N+]=[N-]"),  # Azide anion
                Chem.MolFromSmarts("[N-]=[N+]=[N-]")  # Azide anion alt
            ]
            
            has_azide_nucleophile = any(
                any(reactant.HasSubstructMatch(az_pattern) for reactant in reactant_mols)
                for az_pattern in azide_nucleophile_patterns
            )
            
            # Check for stereocenter involvement
            chiral_carbon_pattern = Chem.MolFromSmarts("[C@,C@@]")
            
            # Check if stereocenter is near the reaction site
            product_stereo = product_mol.HasSubstructMatch(chiral_carbon_pattern)
            reactant_stereo = any(reactant.HasSubstructMatch(chiral_carbon_pattern) 
                                for reactant in reactant_mols)
            
            return (has_leaving_group and 
                   (has_azide_nucleophile or has_azide_product) and
                   (product_stereo or reactant_stereo))
                   
        except Exception:
            return False
