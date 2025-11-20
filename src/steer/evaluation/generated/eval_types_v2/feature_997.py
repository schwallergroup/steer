"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when a specified number
    of major fragments are coupled together at a target depth.
    
    Checks for reactions where multiple substantial fragments (non-trivial molecules)
    are combined, indicating a convergent approach rather than linear synthesis.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_step = config["coupling_step"]
        self.min_heavy_atoms = config.get("min_heavy_atoms", 5)  # Minimum atoms to be considered a major fragment
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling not found
        else:
            # Perfect score if coupling happens at target step, penalty for deviation
            depth_penalty = abs(x - (self.coupling_step / 10.0))  # Normalize to 0-1 scale
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent coupling of major fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0].split(".")
            
            # Need at least the specified number of fragments
            if len(reactants_smiles) < self.fragment_count:
                return False
                
            # Check that we have the right number of major fragments
            major_fragments = []
            for smi in reactants_smiles:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol and mol.GetNumHeavyAtoms() >= self.min_heavy_atoms:
                        major_fragments.append(mol)
                except:
                    continue
            
            # Check if we have exactly the target number of major fragments
            if len(major_fragments) == self.fragment_count:
                # Additional check: ensure fragments are structurally distinct
                return self._fragments_are_distinct(major_fragments)
                
        except Exception:
            pass
            
        return False
    
    def _fragments_are_distinct(self, fragments) -> bool:
        """
        Check that fragments are structurally distinct (not just different conformers).
        """
        if len(fragments) < 2:
            return False
            
        try:
            # Generate fingerprints for comparison
            fps = []
            for mol in fragments:
                fp = Chem.RDKFingerprint(mol)
                fps.append(fp)
            
            # Check pairwise similarity - fragments should be reasonably different
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    if similarity > 0.85:  # Too similar, likely not convergent
                        return False
                        
            return True
            
        except:
            return False
