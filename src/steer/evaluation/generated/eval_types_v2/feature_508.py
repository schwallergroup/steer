"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two major fragments are coupled.
    Checks if the target molecule is formed by combining two distinct fragments
    of specified types in a coupling reaction at a given depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_step = config["coupling_step"]
        self.fragment_types = config["fragment_types"]
        
        # Define SMARTS patterns for fragment types
        self.fragment_patterns = {
            "heterocyclic_alcohol": "[#7,#8,#16]~1~*~*~*~*~1[OH]",  # Heterocycle with alcohol
            "piperidine_bromide": "N1CCCCC1[Br]",  # Piperidine with bromide
            "aromatic_halide": "c[F,Cl,Br,I]",  # Aromatic halide
            "alkyl_chain": "CCCCC",  # Alkyl chain (5+ carbons)
            "ester_group": "C(=O)OC",  # Ester functionality
            "amine_group": "[NH2,NH1]"  # Primary or secondary amine
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Earlier convergent coupling is better (closer to target)
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling of the specified fragments."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            if "." not in reactants_smiles:
                return False  # Not a coupling reaction
                
            reactant_smiles_list = reactants_smiles.split(".")
            
            # Check if we have the expected number of fragments
            if len(reactant_smiles_list) < self.fragment_count:
                return False
                
            # Convert to RDKit molecules
            reactant_mols = []
            for smiles in reactant_smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    reactant_mols.append(mol)
                    
            if len(reactant_mols) < self.fragment_count:
                return False
                
            # Check if fragments match the specified types
            fragment_matches = self._match_fragments(reactant_mols)
            
            # Verify we found matches for all required fragment types
            matched_types = set()
            for fragment_type in fragment_matches:
                matched_types.add(fragment_type)
                
            required_types = set(self.fragment_types)
            return required_types.issubset(matched_types)
            
        except Exception:
            return False
    
    def _match_fragments(self, molecules) -> List[str]:
        """Match molecules to fragment types based on SMARTS patterns."""
        matched_types = []
        
        for mol in molecules:
            for fragment_type in self.fragment_types:
                if fragment_type in self.fragment_patterns:
                    pattern = self.fragment_patterns[fragment_type]
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    
                    if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                        matched_types.append(fragment_type)
                        break  # Each molecule matches at most one fragment type
                        
        return matched_types
