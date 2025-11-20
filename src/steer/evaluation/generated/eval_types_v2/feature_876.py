"""Generated evaluation code for: Late stage Suzuki coupling for biphenyl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiBiphenyl(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction occurs late in the synthesis
    to form a biphenyl (biaryl) bond. Returns higher scores for later occurrence
    of the reaction.
    """
    
    def __init__(self, config: Dict):
        # Configuration for what constitutes "late stage"
        self.target_timing = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score where later Suzuki coupling gets higher score.
        x is the depth fraction (0.0 = root, 1.0 = leaves, -1 = not found)
        """
        if x < 0:
            return 0  # Suzuki coupling not found
        
        # For late-stage preference, higher depth fraction is better
        # Score ranges from 0-10, with latest reactions scoring highest
        return x * 10
        
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a Suzuki coupling forming biphenyl.
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
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if this is a Suzuki coupling pattern
            if not self._is_suzuki_coupling(reactant_mols, product_mol):
                return False
                
            # Check if biphenyl bond is formed
            return self._forms_biphenyl_bond(reactant_mols, product_mol)
            
        except:
            return False
    
    def _is_suzuki_coupling(self, reactants, product):
        """
        Identify Suzuki coupling by presence of organoborane and haloarene reactants.
        """
        has_boron = False
        has_halogen = False
        
        for mol in reactants:
            # Check for boron-containing reactant (boronic acid/ester)
            if any(atom.GetSymbol() == 'B' for atom in mol.GetAtoms()):
                has_boron = True
                
            # Check for halogen on aromatic carbon
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in ['Br', 'I', 'Cl'] and atom.IsInRing():
                    # Check if attached to aromatic carbon
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetIsAromatic():
                            has_halogen = True
                            break
                            
        return has_boron and has_halogen
    
    def _forms_biphenyl_bond(self, reactants, product):
        """
        Check if the reaction forms a biphenyl (aryl-aryl) bond.
        """
        # SMARTS pattern for biphenyl core (two connected aromatic rings)
        biphenyl_pattern = Chem.MolFromSmarts("c1ccccc1-c2ccccc2")
        
        if not biphenyl_pattern:
            return False
            
        # Check if product contains biphenyl but reactants don't
        product_has_biphenyl = product.HasSubstructMatch(biphenyl_pattern)
        
        if not product_has_biphenyl:
            return False
            
        # Verify that no single reactant already contains the biphenyl
        reactants_have_biphenyl = any(mol.HasSubstructMatch(biphenyl_pattern) for mol in reactants)
        
        # True if product has biphenyl but no single reactant does (bond formation)
        return product_has_biphenyl and not reactants_have_biphenyl
