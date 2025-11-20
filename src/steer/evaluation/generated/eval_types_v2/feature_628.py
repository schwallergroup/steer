"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates routes for late-stage nucleophilic aromatic substitution (SNAr) reactions.
    
    This scorer identifies SNAr reactions by detecting the formation of C-O, C-N, or C-S bonds
    to aromatic carbons, typically involving electron-withdrawing groups that activate the
    aromatic ring towards nucleophilic attack.
    """
    
    def __init__(self, config: Dict):
        # No additional configuration needed for this specific reaction type
        pass
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score where late-stage reactions score higher.
        
        Args:
            x: Depth fraction where SNAr reaction occurs (-1 if not found)
            
        Returns:
            Score from 0-10, with higher scores for later-stage reactions
        """
        if x < 0:
            return 0  # SNAr reaction not found
        else:
            # Late-stage reactions (higher x values) get higher scores
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents a nucleophilic aromatic substitution.
        
        Args:
            d: Reaction node dictionary containing metadata
            
        Returns:
            True if the reaction is identified as SNAr
        """
        try:
            mapped_rxn = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Look for nucleophile + electrophile pattern
            nucleophile, electrophile = self._identify_nucleophile_electrophile(reactants)
            
            if nucleophile is None or electrophile is None:
                return False
            
            return self._is_snar_reaction(nucleophile, electrophile, product)
            
        except (KeyError, ValueError, AttributeError):
            return False
    
    def _identify_nucleophile_electrophile(self, reactants):
        """Identify which reactant is the nucleophile and which is the electrophile."""
        nucleophile = None
        electrophile = None
        
        for mol in reactants:
            if self._has_leaving_group_on_aromatic(mol):
                electrophile = mol
            elif self._has_nucleophilic_center(mol):
                nucleophile = mol
        
        return nucleophile, electrophile
    
    def _has_leaving_group_on_aromatic(self, mol):
        """Check if molecule has a leaving group attached to an aromatic carbon."""
        # Common leaving groups in SNAr: F, Cl, Br, I, NO2, SO2R
        leaving_group_patterns = [
            "[cH0:1][F,Cl,Br,I]",  # Halogen on aromatic carbon
            "[cH0:1][N+](=O)[O-]",  # Nitro group
            "[cH0:1]S(=O)(=O)",     # Sulfonyl group
        ]
        
        for pattern in leaving_group_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                # Check for electron-withdrawing groups that activate the ring
                if self._has_electron_withdrawing_groups(mol):
                    return True
        
        return False
    
    def _has_electron_withdrawing_groups(self, mol):
        """Check for electron-withdrawing groups that activate aromatic rings towards SNAr."""
        ewg_patterns = [
            "[N+](=O)[O-]",     # Nitro
            "C(=O)",            # Carbonyl
            "C#N",              # Cyano
            "S(=O)(=O)",        # Sulfonyl
            "C(F)(F)F",         # Trifluoromethyl
        ]
        
        for pattern in ewg_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        return False
    
    def _has_nucleophilic_center(self, mol):
        """Check if molecule has a nucleophilic center suitable for SNAr."""
        nucleophile_patterns = [
            "[OH]",             # Hydroxide/alcohol
            "[NH2,NH]",         # Primary/secondary amine
            "[SH,S]",           # Thiol/sulfide
            "[O-]",             # Alkoxide
            "[N-]",             # Amide anion
        ]
        
        for pattern in nucleophile_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        return False
    
    def _is_snar_reaction(self, nucleophile, electrophile, product):
        """Verify that the reaction represents bond formation between nucleophile and aromatic carbon."""
        # Get atom mapping to track bond formation
        nuc_atoms = set(atom.GetAtomMapNum() for atom in nucleophile.GetAtoms() if atom.GetAtomMapNum() > 0)
        elec_atoms = set(atom.GetAtomMapNum() for atom in electrophile.GetAtoms() if atom.GetAtomMapNum() > 0)
        
        # Look for new bonds in product between nucleophile and electrophile atoms
        for bond in product.GetBonds():
            atom1_map = bond.GetBeginAtom().GetAtomMapNum()
            atom2_map = bond.GetEndAtom().GetAtomMapNum()
            
            # Check if bond connects nucleophile atom to aromatic carbon from electrophile
            if ((atom1_map in nuc_atoms and atom2_map in elec_atoms) or 
                (atom2_map in nuc_atoms and atom1_map in elec_atoms)):
                
                # Verify the electrophile atom is aromatic
                for atom in electrophile.GetAtoms():
                    if atom.GetAtomMapNum() in [atom1_map, atom2_map] and atom.GetIsAromatic():
                        return True
        
        return False
