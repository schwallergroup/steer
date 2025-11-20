"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if two substantial fragments
    are coupled in the final step of the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_step = config.get("coupling_step", "final")
        self.min_fragment_size = config.get("min_fragment_size", 5)  # minimum atoms per fragment
    
    def route_scoring(self, x) -> float:
        """
        Scoring function for convergent synthesis.
        x is the depth fraction where convergent coupling occurs.
        """
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_step == "final":
            # Reward coupling that happens late in the synthesis (closer to final product)
            return 10 * (1 - x)  # Higher score for later coupling
        else:
            # For other coupling preferences, reward based on target position
            return 10 * max(0, 1 - abs(x - 0.5))  # Reward mid-route coupling
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction represents convergent coupling of substantial fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0].strip()
            reactants_smiles = rxn_parts[1].strip()
            
            if not reactants_smiles or "." not in reactants_smiles:
                return False  # Need multiple reactants for convergent coupling
            
            # Parse reactants
            reactant_smiles_list = [r.strip() for r in reactants_smiles.split(".")]
            reactants = []
            
            for smiles in reactant_smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    reactants.append(mol)
            
            # Filter for substantial fragments (not small reagents)
            substantial_fragments = []
            for mol in reactants:
                atom_count = mol.GetNumAtoms()
                # Count non-hydrogen, non-trivial atoms
                heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
                
                if heavy_atoms >= self.min_fragment_size:
                    substantial_fragments.append(mol)
            
            # Check if we have the required number of substantial fragments
            if len(substantial_fragments) >= self.fragment_count:
                # Additional check: ensure fragments are not just protecting group removals
                # by checking structural diversity
                return self._check_structural_diversity(substantial_fragments)
            
            return False
            
        except Exception:
            return False
    
    def _check_structural_diversity(self, fragments) -> bool:
        """
        Check if fragments are structurally diverse enough to constitute convergent synthesis.
        """
        if len(fragments) < 2:
            return False
        
        # Generate Morgan fingerprints for structural comparison
        fps = []
        for mol in fragments:
            try:
                fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps.append(fp)
            except:
                continue
        
        if len(fps) < 2:
            return False
        
        # Calculate Tanimoto similarity between fragments
        from rdkit import DataStructs
        max_similarity = 0.0
        
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                max_similarity = max(max_similarity, similarity)
        
        # Fragments should be sufficiently different (similarity < 0.7)
        return max_similarity < 0.7
