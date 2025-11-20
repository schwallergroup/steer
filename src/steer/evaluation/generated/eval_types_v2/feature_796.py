"""Generated evaluation code for: Sandmeyer reaction for halogen installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SandmeyerReaction(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Sandmeyer reactions.
    
    The Sandmeyer reaction converts aromatic amines to halides via diazonium salt intermediates.
    This class detects reactions where an aromatic amine is replaced by a halogen (Cl, Br, I).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            return 1 - abs(x - self.target_depth) / 10  # Normalize to 0-1 range
    
    def hit_condition(self, d):
        """
        Detects Sandmeyer reaction by checking if an aromatic amine is converted to a halide.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles")
            if not rxn_smiles:
                return False
                
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product has aromatic amine
            aromatic_amine_pattern = Chem.MolFromSmarts("[c:1][NH2:2]")
            if not product_mol.HasSubstructMatch(aromatic_amine_pattern):
                return False
            
            # Get atom mapping for the amine in product
            amine_matches = product_mol.GetSubstructMatches(aromatic_amine_pattern)
            
            for match in amine_matches:
                carbon_idx, nitrogen_idx = match
                carbon_atom = product_mol.GetAtomWithIdx(carbon_idx)
                nitrogen_atom = product_mol.GetAtomWithIdx(nitrogen_idx)
                
                carbon_mapnum = carbon_atom.GetAtomMapNum()
                nitrogen_mapnum = nitrogen_atom.GetAtomMapNum()
                
                if carbon_mapnum == 0 or nitrogen_mapnum == 0:
                    continue
                
                # Check if in any reactant, the carbon has a halogen instead of amine
                for reactant in reactant_mols:
                    carbon_in_reactant = None
                    nitrogen_in_reactant = None
                    
                    for atom in reactant.GetAtoms():
                        if atom.GetAtomMapNum() == carbon_mapnum:
                            carbon_in_reactant = atom
                        elif atom.GetAtomMapNum() == nitrogen_mapnum:
                            nitrogen_in_reactant = atom
                    
                    # If carbon is present but nitrogen is not, check if carbon has halogen
                    if carbon_in_reactant and not nitrogen_in_reactant:
                        for neighbor in carbon_in_reactant.GetNeighbors():
                            if neighbor.GetSymbol() in ['Cl', 'Br', 'I']:
                                return True
            
            return False
            
        except Exception:
            return False
