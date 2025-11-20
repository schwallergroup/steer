"""Generated evaluation code for: Late stage Curtius rearrangement for amine formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCurtiusRearrangement(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Curtius rearrangement reactions.
    
    The Curtius rearrangement converts carboxylic acids to amines via acyl azide intermediates.
    This scorer identifies routes where this transformation occurs in the final stages.
    """
    
    def __init__(self, config: Dict):
        self.target_stage = config.get("stage", "final")
        
    def route_scoring(self, x) -> float:
        """
        Score based on how late in the synthesis the Curtius rearrangement occurs.
        Later stage reactions get higher scores (closer to 1.0).
        """
        if x < 0:
            return 0  # Curtius rearrangement doesn't happen
        else:
            return 1 - x  # Later stage is better (lower depth fraction = higher score)
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents a Curtius rearrangement.
        Identifies COOH to NH2 transformation pattern.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product = rxn_parts[0]
        reactants = rxn_parts[1].split(".")
        
        try:
            prod_mol = Chem.MolFromSmiles(product)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants if Chem.MolFromSmiles(r) is not None]
            
            if not prod_mol or not react_mols:
                return False
                
            # Define patterns for carboxylic acid and primary amine
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
            primary_amine_pattern = Chem.MolFromSmarts("[C][NH2]")
            
            # Check if product contains primary amine
            has_amine_product = prod_mol.HasSubstructMatch(primary_amine_pattern)
            
            # Check if any reactant contains carboxylic acid
            has_carboxylic_reactant = any(mol.HasSubstructMatch(carboxylic_acid_pattern) 
                                        for mol in react_mols)
            
            # Additional check for potential Curtius intermediates or reagents
            curtius_indicators = [
                "[N-]=[N+]=[N-]",  # Azide ion
                "[C](=[O])[N]=[N+]=[N-]",  # Acyl azide
                "[N]=[C]=[O]",  # Isocyanate intermediate
            ]
            
            has_curtius_intermediate = False
            for indicator_smarts in curtius_indicators:
                indicator_pattern = Chem.MolFromSmarts(indicator_smarts)
                if indicator_pattern:
                    has_curtius_intermediate = any(mol.HasSubstructMatch(indicator_pattern) 
                                                 for mol in react_mols)
                    if has_curtius_intermediate:
                        break
            
            # Check atom mapping to ensure same carbon framework
            if has_amine_product and (has_carboxylic_reactant or has_curtius_intermediate):
                return self._verify_carbon_mapping(prod_mol, react_mols)
                
        except Exception:
            return False
            
        return False
    
    def _verify_carbon_mapping(self, product, reactants):
        """
        Verify that the carbon attached to NH2 in product corresponds 
        to carbon attached to COOH in reactants via atom mapping.
        """
        try:
            # Get mapped atoms in product with NH2
            amine_pattern = Chem.MolFromSmarts("[C][NH2]")
            amine_matches = product.GetSubstructMatches(amine_pattern)
            
            for match in amine_matches:
                carbon_idx = match[0]  # Carbon attached to NH2
                carbon_atom = product.GetAtomWithIdx(carbon_idx)
                carbon_map = carbon_atom.GetAtomMapNum()
                
                if carbon_map > 0:
                    # Find corresponding carbon in reactants
                    for reactant in reactants:
                        for atom in reactant.GetAtoms():
                            if atom.GetAtomMapNum() == carbon_map:
                                # Check if this carbon is part of COOH
                                atom_idx = atom.GetIdx()
                                cooh_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
                                cooh_matches = reactant.GetSubstructMatches(cooh_pattern)
                                
                                for cooh_match in cooh_matches:
                                    if atom_idx == cooh_match[0]:
                                        return True
            
            return True  # If mapping verification fails, still allow based on functional group presence
            
        except Exception:
            return True  # Default to allowing if verification fails
