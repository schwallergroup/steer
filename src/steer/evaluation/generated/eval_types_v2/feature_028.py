"""Generated evaluation code for: Late stage bromination of methyl group"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageBromination(BaseScoring):
    """
    Evaluates late-stage allylic bromination of cephem methyl groups using NBS.
    Checks for NBS-mediated bromination reactions and rewards when they occur late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Prefer late stage
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Bromination doesn't happen
        else:
            # Reward late-stage bromination (higher depth fraction is better)
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Continuous scoring - penalize early bromination
                return max(0, 10 * (x - 0.2))  # Scale so late stage gets high scores
    
    def hit_condition(self, d):
        """Check if this reaction is an allylic bromination using NBS on cephem methyl."""
        metadata = d.get("metadata", {})
        
        # Check for NBS reagent
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        if not rxn_smiles or "NBS" not in rxn_smiles.upper():
            # Also check for bromine patterns in products
            if not self._has_bromination_pattern(rxn_smiles):
                return False
        
        # Check if this involves cephem core and methyl bromination
        return self._is_cephem_methyl_bromination(rxn_smiles)
    
    def _has_bromination_pattern(self, rxn_smiles):
        """Check for bromination by looking for Br introduction."""
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants, products = rxn_smiles.split(">>")
        
        try:
            # Count bromine atoms in reactants vs products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            reactant_br = sum(len([a for a in mol.GetAtoms() if a.GetSymbol() == "Br"]) 
                             for mol in reactant_mols if mol is not None)
            product_br = sum(len([a for a in mol.GetAtoms() if a.GetSymbol() == "Br"]) 
                            for mol in product_mols if mol is not None)
            
            return product_br > reactant_br
        except:
            return False
    
    def _is_cephem_methyl_bromination(self, rxn_smiles):
        """Check if reaction involves bromination of methyl group on cephem core."""
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants, products = rxn_smiles.split(">>")
        
        try:
            # Define cephem core pattern
            cephem_pattern = Chem.MolFromSmarts("[#6]1[#6][#6]2[#7][#6](=[#8])[#6]([#7]2[#6]1=[#8])")
            
            # Pattern for methyl group that could be brominated
            methyl_pattern = Chem.MolFromSmarts("[#6][CH3]")
            bromomethyl_pattern = Chem.MolFromSmarts("[#6][CH2][Br]")
            
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Check for cephem core in both reactants and products
            has_cephem_reactant = any(mol and cephem_pattern and mol.HasSubstructMatch(cephem_pattern) 
                                    for mol in reactant_mols)
            has_cephem_product = any(mol and cephem_pattern and mol.HasSubstructMatch(cephem_pattern) 
                                   for mol in product_mols)
            
            if not (has_cephem_reactant and has_cephem_product):
                return False
            
            # Check for methyl -> bromomethyl transformation
            has_methyl_reactant = any(mol and methyl_pattern and mol.HasSubstructMatch(methyl_pattern) 
                                    for mol in reactant_mols)
            has_bromomethyl_product = any(mol and bromomethyl_pattern and mol.HasSubstructMatch(bromomethyl_pattern) 
                                        for mol in product_mols)
            
            return has_methyl_reactant and has_bromomethyl_product
            
        except:
            return False
