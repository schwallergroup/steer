"""Generated evaluation code for: Late stage Wittig olefination for Z-alkene"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWittigZAlkene(BaseScoring):
    """
    Evaluates if a Wittig olefination reaction occurs late in the synthesis route
    and forms a Z-alkene product. Returns higher scores for later-stage Wittig reactions.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
        self.require_z_alkene = config.get("stereochemistry", "Z") == "Z"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Wittig reaction doesn't happen
        else:
            # Late-stage Wittig is preferred, so higher depth fraction is better
            return 10 * x  # Scale to 0-10 range
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction is a Wittig olefination that forms a Z-alkene
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        # Split reaction SMILES
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check for Wittig reaction pattern: phosphonium ylide + carbonyl -> alkene + phosphine oxide
            if not self._is_wittig_reaction(reactant_mols, product_mol):
                return False
            
            # If Z-alkene is required, check stereochemistry
            if self.require_z_alkene:
                return self._has_z_alkene_product(product_mol)
            
            return True
            
        except Exception:
            return False
    
    def _is_wittig_reaction(self, reactants, product):
        """
        Check if reaction involves Wittig reagents and forms alkene
        """
        # Pattern for phosphonium ylide (simplified)
        ylide_pattern = Chem.MolFromSmarts("[P+]([C-])")
        # Pattern for carbonyl (aldehyde or ketone)
        carbonyl_pattern = Chem.MolFromSmarts("[C]=[O]")
        # Pattern for alkene in product
        alkene_pattern = Chem.MolFromSmarts("C=C")
        
        has_ylide = False
        has_carbonyl = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(ylide_pattern):
                has_ylide = True
            if reactant.HasSubstructMatch(carbonyl_pattern):
                has_carbonyl = True
        
        has_alkene_product = product.HasSubstructMatch(alkene_pattern)
        
        return has_ylide and has_carbonyl and has_alkene_product
    
    def _has_z_alkene_product(self, product_mol):
        """
        Check if product contains Z-alkene stereochemistry
        """
        # Look for double bonds with Z stereochemistry
        for bond in product_mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetStereo() == Chem.BondStereo.STEREOZ:
                    return True
                # Also check for cis configuration in rings
                if bond.IsInRing():
                    # In rings, double bonds are typically cis (Z-like)
                    return True
        
        return False
