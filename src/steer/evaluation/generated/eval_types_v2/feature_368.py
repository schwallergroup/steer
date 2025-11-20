"""Generated evaluation code for: Convergent synthesis via two fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two major fragments are coupled.
    Checks for fragment coupling reactions (like SNAr) at a specific depth and
    validates that fragments meet complexity requirements.
    """
    
    def __init__(self, config: Dict):
        self.target_coupling_depth = config["coupling_depth"]
        self.fragment_complexity = config["fragment_complexity"]
        
        # Define coupling reaction patterns (SNAr and other common coupling reactions)
        self.coupling_patterns = [
            "[c:1][F,Cl,Br,I:2]>>[c:1][N,O,S:3]",  # SNAr pattern
            "[C:1]=[O:2].[N:3][H:4]>>[C:1]([N:3])[O:2]",  # Amide coupling
            "[c:1][B:2].[c:3][X:4]>>[c:1][c:3]",  # Suzuki coupling
            "[C:1][B:2].[c:3][X:4]>>[C:1][c:3]",  # Suzuki sp3-sp2
            "[c:1][Sn:2].[c:3][X:4]>>[c:1][c:3]"   # Stille coupling
        ]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        
        # Calculate depth-based score (closer to target depth is better)
        depth_score = max(0, 10 - abs(x * 10 - self.target_coupling_depth))
        return depth_score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent fragment coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        try:
            # Parse reaction
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            # Must have exactly 2 major reactants for convergent synthesis
            if len(reactants_smiles) < 2:
                return False
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not product or len(reactants) < 2:
                return False
            
            # Check if this looks like a coupling reaction
            if not self._is_coupling_reaction(product, reactants):
                return False
            
            # Check fragment complexity
            major_fragments = self._identify_major_fragments(reactants)
            if len(major_fragments) < 2:
                return False
                
            return self._fragments_meet_complexity(major_fragments)
            
        except Exception:
            return False
    
    def _is_coupling_reaction(self, product, reactants):
        """Check if reaction matches coupling patterns"""
        # Simple heuristic: product should be significantly larger than largest reactant
        product_size = product.GetNumAtoms()
        max_reactant_size = max(r.GetNumAtoms() for r in reactants)
        
        # Product should be combination of at least two substantial fragments
        if product_size < max_reactant_size * 1.5:
            return False
            
        # Check for presence of heteroatoms or aromatic systems (common in coupling)
        has_aromatic = any(atom.GetIsAromatic() for atom in product.GetAtoms())
        has_heteroatom = any(atom.GetSymbol() not in ['C', 'H'] for atom in product.GetAtoms())
        
        return has_aromatic or has_heteroatom
    
    def _identify_major_fragments(self, reactants):
        """Identify fragments that are substantial (not just reagents/catalysts)"""
        # Sort by size and take the two largest that meet minimum size
        min_fragment_size = 8  # Minimum atoms for a "major" fragment
        
        major_fragments = []
        for reactant in sorted(reactants, key=lambda x: x.GetNumAtoms(), reverse=True):
            if reactant.GetNumAtoms() >= min_fragment_size:
                # Skip simple reagents/solvents
                if not self._is_simple_reagent(reactant):
                    major_fragments.append(reactant)
                    
        return major_fragments[:2]  # Return top 2 major fragments
    
    def _is_simple_reagent(self, mol):
        """Check if molecule is likely a simple reagent rather than a major fragment"""
        simple_reagents = [
            "[Li]", "[Na]", "[K]",  # Metals
            "O", "N", "S",          # Simple heteroatoms
            "C(=O)O", "CC(=O)O",    # Simple acids
            "CCO", "CO"             # Simple alcohols
        ]
        
        mol_smiles = Chem.MolToSmiles(mol)
        return any(Chem.MolFromSmarts(pattern) and 
                  mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                  for pattern in simple_reagents if Chem.MolFromSmarts(pattern))
    
    def _fragments_meet_complexity(self, fragments):
        """Check if fragments meet the specified complexity requirements"""
        if len(fragments) < 2:
            return False
            
        complexity_scores = [self._calculate_fragment_complexity(frag) for frag in fragments]
        
        if self.fragment_complexity == "high":
            return all(score >= 0.7 for score in complexity_scores)
        elif self.fragment_complexity == "medium":
            return all(score >= 0.4 for score in complexity_scores)
        else:  # low complexity
            return all(score >= 0.2 for score in complexity_scores)
    
    def _calculate_fragment_complexity(self, fragment):
        """Calculate normalized complexity score for a fragment"""
        if not fragment:
            return 0
            
        # Complexity factors
        num_atoms = fragment.GetNumAtoms()
        num_rings = fragment.GetRingInfo().NumRings()
        num_aromatic = sum(1 for atom in fragment.GetAtoms() if atom.GetIsAromatic())
        num_heteroatoms = sum(1 for atom in fragment.GetAtoms() if atom.GetSymbol() not in ['C', 'H'])
        num_stereocenters = len(Chem.FindMolChiralCenters(fragment))
        
        # Weighted complexity score (normalized to 0-1 range)
        complexity = (
            min(num_atoms / 20, 1) * 0.3 +          # Size component
            min(num_rings / 3, 1) * 0.3 +           # Ring component
            min(num_aromatic / 10, 1) * 0.2 +       # Aromaticity component
            min(num_heteroatoms / 5, 1) * 0.1 +     # Heteroatom component
            min(num_stereocenters / 2, 1) * 0.1     # Stereochemistry component
        )
        
        return complexity
