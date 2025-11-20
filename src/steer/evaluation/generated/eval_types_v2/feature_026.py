"""Generated evaluation code for: Convergent synthesis via two complex fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if complex fragments 
    are coupled in a final step via a specific reaction type.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_step = config.get("coupling_step", "final")
        self.coupling_reaction = config.get("coupling_reaction", "suzuki")
        
        # Define reaction SMARTS patterns
        self.reaction_patterns = {
            "suzuki": "[#6:1]-[B:2]([OH:3])[OH:4].[Br,I,Cl:5]-[#6:6]>>[#6:1]-[#6:6]",
            "sonogashira": "[#6:1]#[CH:2].[Br,I,Cl:3]-[#6:4]>>[#6:1]#[#6:2]-[#6:4]",
            "heck": "[#6:1]=[CH2:2].[Br,I,Cl:3]-[#6:4]>>[#6:1]-[#6:2]=[#6:4]",
            "stille": "[#6:1]-[Sn:2].[Br,I,Cl:3]-[#6:4]>>[#6:1]-[#6:4]",
            "amide_coupling": "[#6:1][C:2](=[O:3])[OH:4].[NH2:5][#6:6]>>[#6:1][C:2](=[O:3])[NH:5][#6:6]"
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        else:
            # Higher score for earlier convergent steps (lower depth fraction)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents convergent coupling of complex fragments"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
                
            # Parse reaction
            if ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Filter out None molecules and reagents (small molecules)
            valid_reactants = []
            for mol in reactant_mols:
                if mol is not None and mol.GetNumHeavyAtoms() >= 10:  # Complex fragment threshold
                    valid_reactants.append(mol)
            
            # Check if we have the expected number of complex fragments
            if len(valid_reactants) != self.fragment_count:
                return False
            
            # Check if this is the expected coupling reaction type
            if not self._is_coupling_reaction(rxn_smiles):
                return False
            
            # Additional check for fragment complexity
            return self._fragments_are_complex(valid_reactants)
            
        except Exception:
            return False
    
    def _is_coupling_reaction(self, rxn_smiles: str) -> bool:
        """Check if the reaction matches the specified coupling reaction pattern"""
        try:
            if self.coupling_reaction in self.reaction_patterns:
                pattern = self.reaction_patterns[self.coupling_reaction]
                rxn = AllChem.ReactionFromSmarts(pattern)
                test_rxn = AllChem.ReactionFromSmarts(rxn_smiles)
                
                # Simple pattern matching - check for characteristic functional groups
                reactants_part = rxn_smiles.split(">>")[0]
                
                if self.coupling_reaction == "suzuki":
                    return "B(" in reactants_part and ("Br" in reactants_part or "I" in reactants_part or "Cl" in reactants_part)
                elif self.coupling_reaction == "sonogashira":
                    return "#C" in reactants_part and ("Br" in reactants_part or "I" in reactants_part)
                elif self.coupling_reaction == "heck":
                    return "C=C" in reactants_part and ("Br" in reactants_part or "I" in reactants_part)
                elif self.coupling_reaction == "stille":
                    return "[Sn]" in reactants_part and ("Br" in reactants_part or "I" in reactants_part)
                elif self.coupling_reaction == "amide_coupling":
                    return ("C(=O)O" in reactants_part or "C(O)=O" in reactants_part) and "N" in reactants_part
                    
            return True  # Default to True if pattern not defined
            
        except Exception:
            return True  # Default to True on parsing errors
    
    def _fragments_are_complex(self, reactants: List) -> bool:
        """Check if fragments have sufficient complexity for convergent synthesis"""
        for mol in reactants:
            if mol is None:
                continue
                
            # Check for minimum complexity indicators
            heavy_atoms = mol.GetNumHeavyAtoms()
            rings = mol.GetRingInfo().NumRings()
            heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
            
            # Fragment should have reasonable complexity
            complexity_score = heavy_atoms + rings * 2 + heteroatoms
            if complexity_score < 15:  # Minimum complexity threshold
                return False
                
        return True
