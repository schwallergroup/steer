"""Generated evaluation code for: Suzuki coupling with chemoselectivity challenge"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SuzukiChemoselectivity(BaseScoring):
    """
    Evaluates Suzuki-Miyaura coupling reactions with chemoselectivity challenges.
    Specifically checks for Suzuki coupling at C-Cl sites when more reactive 
    vinyl C-Br sites are present, creating selectivity issues.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        else:
            # Earlier occurrence (lower depth) is more challenging, hence better score
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction is a Suzuki coupling with chemoselectivity challenge
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is a Suzuki coupling (formation of C-C bond from organoborane)
            has_boron_reactant = any(self._contains_boron(mol) for mol in reactants)
            has_halogen_reactant = any(self._contains_halogen(mol) for mol in reactants)
            
            if not (has_boron_reactant and has_halogen_reactant):
                return False
            
            # Check for chemoselectivity challenge: C-Cl coupling with vinyl C-Br present
            return self._has_chemoselectivity_challenge(product, reactants)
            
        except Exception:
            return False
    
    def _contains_boron(self, mol) -> bool:
        """Check if molecule contains boron (organoborane coupling partner)"""
        if not mol:
            return False
        return any(atom.GetSymbol() == 'B' for atom in mol.GetAtoms())
    
    def _contains_halogen(self, mol) -> bool:
        """Check if molecule contains halogen (Cl, Br, I)"""
        if not mol:
            return False
        halogens = {'Cl', 'Br', 'I'}
        return any(atom.GetSymbol() in halogens for atom in mol.GetAtoms())
    
    def _has_chemoselectivity_challenge(self, product, reactants) -> bool:
        """
        Check if reaction involves C-Cl coupling while vinyl C-Br is present
        """
        # Find halogenated reactant
        halogen_reactant = None
        for mol in reactants:
            if self._contains_halogen(mol):
                halogen_reactant = mol
                break
        
        if not halogen_reactant:
            return False
        
        # Check for vinyl bromide pattern (C=C-Br)
        vinyl_br_pattern = Chem.MolFromSmarts("[C]=[C]-Br")
        has_vinyl_br = halogen_reactant.HasSubstructMatch(vinyl_br_pattern)
        
        # Check for chloride that gets coupled (disappears in product)
        cl_pattern = Chem.MolFromSmarts("[C]-Cl")
        reactant_has_cl = halogen_reactant.HasSubstructMatch(cl_pattern)
        product_has_cl = product.HasSubstructMatch(cl_pattern)
        
        # Chemoselectivity challenge: vinyl Br present, Cl was coupled (consumed)
        cl_was_coupled = reactant_has_cl and not product_has_cl
        
        return has_vinyl_br and cl_was_coupled
