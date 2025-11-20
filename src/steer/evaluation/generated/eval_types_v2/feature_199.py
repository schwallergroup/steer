"""Generated evaluation code for: TMS group as regiochemical placeholder"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TMSBromodesilylation(BaseScoring):
    """
    Detects bromodesilylation reactions where TMS (trimethylsilyl) groups are used as 
    regiochemical placeholders and replaced by bromine for regioselective synthesis.
    
    Identifies reactions where:
    1. Product contains TMS group [Si(CH3)3]
    2. Reactant has TMS replaced by Br at the same position
    3. This indicates TMS was used as a directing group for regioselective bromination
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score"""
        if x < 0:
            return 0  # Reaction not found
        else:
            # Early use of TMS protection is better (lower depth)
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves bromodesilylation with TMS as placeholder"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Define TMS pattern: Si with 3 methyl groups
            tms_pattern = Chem.MolFromSmarts("[Si]([CH3])([CH3])[CH3]")
            if not tms_pattern:
                return False
            
            # Check for bromodesilylation pattern
            return self._detect_bromodesilylation(reactants, products, tms_pattern)
            
        except Exception:
            return False
    
    def _detect_bromodesilylation(self, reactants, products, tms_pattern):
        """
        Detect if TMS group in reactant is replaced by Br in product at same position
        """
        # Find reactant with TMS group
        tms_reactant = None
        tms_atoms = []
        
        for reactant in reactants:
            matches = reactant.GetSubstructMatches(tms_pattern)
            if matches:
                tms_reactant = reactant
                # Get the silicon atom indices
                tms_atoms.extend([match[0] for match in matches])  # Si is first atom in pattern
        
        if not tms_reactant or not tms_atoms:
            return False
        
        # Find the carbon atoms bonded to Si (attachment points)
        attachment_maps = []
        for si_idx in tms_atoms:
            si_atom = tms_reactant.GetAtomWithIdx(si_idx)
            si_map = si_atom.GetAtomMapNum()
            
            # Find carbon attached to Si (not the methyl carbons)
            for neighbor in si_atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    # Check if this carbon has other neighbors (not just Si and H)
                    non_si_neighbors = [n for n in neighbor.GetNeighbors() if n.GetSymbol() != 'Si']
                    if len(non_si_neighbors) > 0:  # This is the attachment carbon
                        attachment_maps.append(neighbor.GetAtomMapNum())
                        break
        
        if not attachment_maps:
            return False
        
        # Check if any product has Br at the same mapped position
        for product in products:
            for atom in product.GetAtoms():
                if (atom.GetSymbol() == 'Br' and 
                    atom.GetAtomMapNum() in attachment_maps):
                    return True
                    
                # Also check if Br is bonded to the carbon that was attached to Si
                if atom.GetSymbol() == 'C' and atom.GetAtomMapNum() in attachment_maps:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'Br':
                            return True
        
        return False
