"""Generated evaluation code for: Evans auxiliary attachment to pre-formed stereocenter"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EvansAuxiliaryAttachment(BaseScoring):
    """
    Evaluates Evans auxiliary attachment to molecules with pre-existing stereocenters.
    Checks if an Evans oxazolidinone auxiliary is attached at the specified depth
    when the molecule already contains stereochemistry.
    """
    
    def __init__(self, config: Dict):
        self.auxiliary_type = config["parameters"]["auxiliary_type"]
        self.stereocenter_preexisting = config["parameters"]["stereocenter_preexisting"]
        self.target_depth = config["parameters"]["attachment_depth"]
        
        # Evans oxazolidinone SMARTS pattern
        self.evans_pattern = Chem.MolFromSmarts("[C,N]-C(=O)-N1-C(=O)-O-C-C-1")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Auxiliary attachment doesn't occur
        else:
            # Closer to target depth is better
            depth_score = max(0, 1 - abs(x - self.target_depth) / 10)
            return depth_score * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Evans auxiliary attachment to a molecule 
        with pre-existing stereocenters.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains Evans auxiliary
            if not product.HasSubstructMatch(self.evans_pattern):
                return False
            
            # Check if any reactant has the auxiliary (would be removal, not attachment)
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.evans_pattern):
                    return False
            
            # Check for pre-existing stereocenters in reactants
            if self.stereocenter_preexisting:
                has_preexisting_stereo = False
                for reactant in reactants:
                    stereo_centers = Chem.FindMolChiralCenters(reactant, includeUnassigned=False)
                    if len(stereo_centers) > 0:
                        has_preexisting_stereo = True
                        break
                
                if not has_preexisting_stereo:
                    return False
            
            # Check if one reactant contains the auxiliary precursor and another is the main substrate
            auxiliary_reactant = None
            substrate_reactant = None
            
            for reactant in reactants:
                # Look for oxazolidinone precursor patterns
                oxazolidinone_pattern = Chem.MolFromSmarts("N1-C(=O)-O-C-C-1")
                if reactant.HasSubstructMatch(oxazolidinone_pattern):
                    auxiliary_reactant = reactant
                elif len([atom for atom in reactant.GetAtoms() if atom.GetSymbol() == 'C']) > 3:
                    substrate_reactant = reactant
            
            return auxiliary_reactant is not None and substrate_reactant is not None
            
        except Exception:
            return False
