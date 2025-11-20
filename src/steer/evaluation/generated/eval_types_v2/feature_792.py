"""Generated evaluation code for: Acyl chloride formation before nitro reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcylChlorideBeforeNitroReduction(MultiRxnCondBase):
    """
    Detects if acyl chloride formation occurs before nitro group reduction,
    creating functional group incompatibility issues.
    """
    
    def __init__(self, config):
        self.penalize_incompatibility = config.get("penalize_incompatibility", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track the order of reactions
        acyl_chloride_depth = -1
        nitro_reduction_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_acyl_chloride_formation(rxn):
                if acyl_chloride_depth == -1:  # First occurrence
                    acyl_chloride_depth = i
            
            if self.detect_nitro_reduction(rxn):
                if nitro_reduction_depth == -1:  # First occurrence
                    nitro_reduction_depth = i
        
        # Check if both reactions occur and acyl chloride comes before nitro reduction
        if acyl_chloride_depth >= 0 and nitro_reduction_depth >= 0:
            incompatible_sequence = acyl_chloride_depth < nitro_reduction_depth
            condition = incompatible_sequence == self.penalize_incompatibility
            return condition, len(reactions)
        
        # If only one or neither reaction type is present, no incompatibility
        return not self.penalize_incompatibility, len(reactions)
    
    def detect_acyl_chloride_formation(self, rxn):
        """Detect formation of acyl chlorides (C(=O)Cl)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1].split(".")
        
        # Look for acyl chloride pattern in products
        acyl_chloride_pattern = Chem.MolFromSmarts("C(=O)Cl")
        
        for product_smiles in products:
            try:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol and product_mol.HasSubstructMatch(acyl_chloride_pattern):
                    # Verify it wasn't already present in reactants
                    reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
                    reactant_has_acyl_chloride = any(
                        mol and mol.HasSubstructMatch(acyl_chloride_pattern) 
                        for mol in reactant_mols
                    )
                    if not reactant_has_acyl_chloride:
                        return True
            except:
                continue
        
        return False
    
    def detect_nitro_reduction(self, rxn):
        """Detect reduction of nitro groups to amines"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")  # Nitro group
        amine_pattern = Chem.MolFromSmarts("N")  # Amine nitrogen
        
        # Check if reactants have nitro groups
        reactant_has_nitro = False
        for reactant_smiles in reactants:
            try:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles.strip())
                if reactant_mol and reactant_mol.HasSubstructMatch(nitro_pattern):
                    reactant_has_nitro = True
                    break
            except:
                continue
        
        if not reactant_has_nitro:
            return False
        
        # Check if products have corresponding amines (simplified check)
        product_has_amine = False
        for product_smiles in products:
            try:
                product_mol = Chem.MolFromSmiles(product_smiles.strip())
                if product_mol and product_mol.HasSubstructMatch(amine_pattern):
                    # Additional check: fewer nitro groups in products than reactants
                    product_nitro_count = len(product_mol.GetSubstructMatches(nitro_pattern))
                    total_reactant_nitro_count = sum(
                        len(Chem.MolFromSmiles(r.strip()).GetSubstructMatches(nitro_pattern))
                        for r in reactants
                        if Chem.MolFromSmiles(r.strip())
                    )
                    if product_nitro_count < total_reactant_nitro_count:
                        product_has_amine = True
                        break
            except:
                continue
        
        return reactant_has_nitro and product_has_amine
