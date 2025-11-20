"""Generated evaluation code for: Chiral auxiliary attachment without stereochemical purpose"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChiralAuxiliaryMisuse(BaseScoring):
    """
    Detects chiral auxiliary attachment without stereochemical purpose.
    Identifies cases where Evans oxazolidinone is attached and removed from
    molecules that already contain required stereocenters.
    """
    
    def __init__(self, config: Dict):
        self.auxiliary_type = config["parameters"]["auxiliary_type"]
        self.purpose = config["parameters"]["purpose"]
        self.existing_stereocenters = config["parameters"]["existing_stereocenters"]
        
        # Evans oxazolidinone SMARTS pattern
        self.evans_pattern = Chem.MolFromSmarts("[#6]1[#6][#7][#6](=O)[#8][#6]1")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 10  # Good - no misuse detected
        else:
            # Penalty for misuse, worse if it happens early
            return max(0, 10 - (8 * (1 - x)))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Evans auxiliary misuse"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r.strip()) 
                         for r in react_smiles.split(".") if r.strip()]
            
            if not prod_mol or not react_mols:
                return False
                
            # Check if Evans auxiliary is being attached or removed
            prod_has_auxiliary = self._has_evans_auxiliary(prod_mol)
            react_has_auxiliary = any(self._has_evans_auxiliary(mol) for mol in react_mols)
            
            # Auxiliary attachment or removal detected
            if prod_has_auxiliary != react_has_auxiliary:
                # Check if molecule already has stereocenters (excluding auxiliary)
                if self._has_existing_stereocenters(prod_mol, react_mols):
                    return True
                    
        except Exception:
            pass
            
        return False
    
    def _has_evans_auxiliary(self, mol) -> bool:
        """Check if molecule contains Evans oxazolidinone auxiliary"""
        if mol is None:
            return False
        return mol.HasSubstructMatch(self.evans_pattern)
    
    def _has_existing_stereocenters(self, prod_mol, react_mols) -> bool:
        """Check if molecule has stereocenters outside of the auxiliary"""
        all_mols = [prod_mol] + react_mols
        
        for mol in all_mols:
            if mol is None:
                continue
                
            # Remove auxiliary substructure temporarily to check remaining stereocenters
            mol_copy = Chem.Mol(mol)
            
            # Count stereocenters in original molecule
            stereo_centers = []
            for atom in mol_copy.GetAtoms():
                if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                    stereo_centers.append(atom.GetIdx())
            
            # If auxiliary is present, check if stereocenters exist outside auxiliary
            if self._has_evans_auxiliary(mol_copy) and len(stereo_centers) > 0:
                # Get auxiliary atom indices
                auxiliary_matches = mol_copy.GetSubstructMatches(self.evans_pattern)
                if auxiliary_matches:
                    auxiliary_atoms = set(auxiliary_matches[0])
                    # Check if any stereocenters are outside auxiliary
                    external_stereo = [idx for idx in stereo_centers 
                                     if idx not in auxiliary_atoms]
                    if external_stereo:
                        return True
            elif len(stereo_centers) > 0:
                return True
                
        return False
