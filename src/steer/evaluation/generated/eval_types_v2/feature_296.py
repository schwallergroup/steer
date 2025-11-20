"""Generated evaluation code for: Convergent synthesis via two distinct fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two distinct molecular 
    fragments are coupled together, specifically targeting Suzuki coupling reactions.
    Rewards routes where complex fragments are assembled separately before coupling.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "suzuki_coupling")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Earlier convergent coupling is better (more synthetic efficiency)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of two distinct fragments
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Must have exactly the specified number of fragments
            if len(reactants) != self.fragment_count:
                return False
                
            # Check if it's a Suzuki-type coupling
            if not self._is_suzuki_coupling(reactants, product_smiles):
                return False
                
            # Check that fragments are sufficiently complex and distinct
            return self._are_distinct_complex_fragments(reactants)
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, reactants, product_smiles) -> bool:
        """
        Detect Suzuki coupling by looking for boronic acid/ester + aryl halide pattern
        """
        try:
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            if not all(mol for mol in reactant_mols):
                return False
                
            # Boronic acid/ester patterns
            boronic_patterns = [
                Chem.MolFromSmarts("[#6]-B(-O)-O"),  # Boronic ester
                Chem.MolFromSmarts("[#6]-B(O)O"),    # Boronic acid
                Chem.MolFromSmarts("[#6]-B(-[OH])-[OH]")  # Alternative boronic acid
            ]
            
            # Aryl halide patterns
            halide_patterns = [
                Chem.MolFromSmarts("c[Cl,Br,I]"),    # Aryl halides
                Chem.MolFromSmarts("[$(c1ccccc1)][Cl,Br,I]")  # Aromatic halides
            ]
            
            has_boronic = False
            has_halide = False
            
            for mol in reactant_mols:
                # Check for boronic acid/ester
                for pattern in boronic_patterns:
                    if mol.HasSubstructMatch(pattern):
                        has_boronic = True
                        break
                        
                # Check for aryl halide
                for pattern in halide_patterns:
                    if mol.HasSubstructMatch(pattern):
                        has_halide = True
                        break
            
            return has_boronic and has_halide
            
        except Exception:
            return False
    
    def _are_distinct_complex_fragments(self, reactants) -> bool:
        """
        Check that reactants are sufficiently complex and structurally distinct
        """
        try:
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            if not all(mol for mol in reactant_mols):
                return False
            
            # Check minimum complexity (atom count)
            min_atoms = 8
            if not all(mol.GetNumAtoms() >= min_atoms for mol in reactant_mols):
                return False
            
            # Check structural diversity using Tanimoto similarity
            fps = [Chem.RDKFingerprint(mol) for mol in reactant_mols]
            
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    # Fragments should be sufficiently different
                    if similarity > 0.6:
                        return False
            
            return True
            
        except Exception:
            return False
