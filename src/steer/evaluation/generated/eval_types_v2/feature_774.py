"""Generated evaluation code for: Late stage ketone reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageKetoneReduction(BaseScoring):
    """
    Evaluates whether ketone reduction occurs at a late stage in the synthesis.
    Detects reactions where a ketone is reduced to an alcohol and scores based on timing.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")  # "early" or "late"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ketone reduction doesn't happen
        else:
            if self.timing_preference == "late":
                return 1 - x  # Late-stage reduction is better (lower depth fraction)
            else:
                return x  # Early-stage reduction is better (higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves ketone reduction.
        Looks for C=O -> C-OH transformation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants_smiles = rxn_parts[1].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not product or not reactants:
                return False
            
            # Find ketone pattern in reactants: C=O (not aldehyde, not amide, not ester)
            ketone_pattern = Chem.MolFromSmarts("[CH3,CH2,CH1,C][C](=[O])[CH3,CH2,CH1,C]")
            # Find alcohol pattern in product: C-OH
            alcohol_pattern = Chem.MolFromSmarts("[CH3,CH2,CH1,C][CH]([OH])[CH3,CH2,CH1,C]")
            
            # Check if any reactant has ketone and product has corresponding alcohol
            has_ketone_reactant = any(reactant.HasSubstructMatch(ketone_pattern) for reactant in reactants)
            has_alcohol_product = product.HasSubstructMatch(alcohol_pattern)
            
            if not (has_ketone_reactant and has_alcohol_product):
                return False
            
            # Verify the transformation by checking atom mapping
            return self._verify_ketone_to_alcohol_mapping(product, reactants)
            
        except Exception:
            return False
    
    def _verify_ketone_to_alcohol_mapping(self, product, reactants) -> bool:
        """
        Verify that mapped atoms show ketone C=O -> alcohol C-OH transformation.
        """
        try:
            # Get atom map numbers for carbons bonded to OH in product
            alcohol_carbons = set()
            for atom in product.GetAtoms():
                if atom.GetSymbol() == 'C' and atom.GetAtomMapNum() > 0:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'O':
                            # Check if this oxygen is an OH (single bond, not double)
                            bond = product.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                            if bond.GetBondType() == Chem.BondType.SINGLE:
                                alcohol_carbons.add(atom.GetAtomMapNum())
            
            # Check if corresponding atoms in reactants are ketone carbons
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if (atom.GetSymbol() == 'C' and 
                        atom.GetAtomMapNum() in alcohol_carbons):
                        # Check if this carbon has a double bond to oxygen (ketone)
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetSymbol() == 'O':
                                bond = reactant.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                                if bond.GetBondType() == Chem.BondType.DOUBLE:
                                    return True
            
            return False
            
        except Exception:
            return False
