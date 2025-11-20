"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are built separately
    and then coupled via a specific reaction type (e.g., SNAr for diaryl ether formation).
    
    Checks for convergent coupling reactions at appropriate depths in the synthesis tree.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction_type = config.get("coupling_reaction_type", "SNAr")
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)  # Early convergence preferred
    
    def route_scoring(self, x) -> float:
        """
        Score the route based on when convergent coupling occurs.
        Earlier convergence (lower depth fraction) gets higher score.
        """
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.condition_type == "bool":
            return 1  # Found convergent coupling
        else:
            # Prefer earlier convergence - penalize late coupling
            return max(0, 1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step.
        """
        metadata = d.get("metadata", {})
        
        # Check if we have the expected number of reactants (fragments)
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        if ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        reactants = reactants_smiles.split(".")
        
        # Must have exactly the specified number of fragments
        if len(reactants) != self.fragment_count:
            return False
        
        # Check if this is the specified coupling reaction type
        if not self._is_coupling_reaction_type(rxn_smiles):
            return False
        
        # Verify fragments have reasonable complexity (not just simple starting materials)
        return self._fragments_have_complexity(reactants)
    
    def _is_coupling_reaction_type(self, rxn_smiles: str) -> bool:
        """
        Check if the reaction matches the specified coupling type.
        """
        if self.coupling_reaction_type.lower() == "snar":
            return self._is_snar_reaction(rxn_smiles)
        elif self.coupling_reaction_type.lower() == "suzuki":
            return self._is_suzuki_reaction(rxn_smiles)
        elif self.coupling_reaction_type.lower() == "buchwald":
            return self._is_buchwald_hartwig_reaction(rxn_smiles)
        else:
            # Generic coupling - just check for C-C or C-N bond formation
            return self._is_generic_coupling(rxn_smiles)
    
    def _is_snar_reaction(self, rxn_smiles: str) -> bool:
        """
        Detect SNAr reaction pattern (nucleophilic aromatic substitution).
        Look for diaryl ether formation or similar patterns.
        """
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
        product = Chem.MolFromSmiles(product_smiles)
        
        if not all([r for r in reactants] + [product]):
            return False
        
        # Look for aromatic rings with electron-withdrawing groups in reactants
        # and diaryl ether formation in product
        ether_pattern = Chem.MolFromSmarts("c-O-c")  # Diaryl ether
        fluoride_pattern = Chem.MolFromSmarts("c-F")  # Leaving group
        
        has_fluoride = any(r.HasSubstructMatch(fluoride_pattern) for r in reactants)
        forms_ether = product.HasSubstructMatch(ether_pattern)
        
        return has_fluoride and forms_ether
    
    def _is_suzuki_reaction(self, rxn_smiles: str) -> bool:
        """
        Detect Suzuki coupling pattern.
        """
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
        
        if not all([r for r in reactants]):
            return False
        
        # Look for boronic acid/ester and halide patterns
        boronic_pattern = Chem.MolFromSmarts("[#6]-[#5]")  # C-B bond
        halide_pattern = Chem.MolFromSmarts("c-[Br,I,Cl]")  # Aryl halide
        
        has_boron = any(r.HasSubstructMatch(boronic_pattern) for r in reactants)
        has_halide = any(r.HasSubstructMatch(halide_pattern) for r in reactants)
        
        return has_boron and has_halide
    
    def _is_buchwald_hartwig_reaction(self, rxn_smiles: str) -> bool:
        """
        Detect Buchwald-Hartwig amination pattern.
        """
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
        product = Chem.MolFromSmiles(product_smiles)
        
        if not all([r for r in reactants] + [product]):
            return False
        
        # Look for amine and aryl halide forming C-N bond
        amine_pattern = Chem.MolFromSmarts("[NX3;!$(N-[#6]=[O,N,S])]")
        halide_pattern = Chem.MolFromSmarts("c-[Br,I,Cl]")
        aryl_amine_pattern = Chem.MolFromSmarts("c-N")
        
        has_amine = any(r.HasSubstructMatch(amine_pattern) for r in reactants)
        has_halide = any(r.HasSubstructMatch(halide_pattern) for r in reactants)
        forms_aryl_amine = product.HasSubstructMatch(aryl_amine_pattern)
        
        return has_amine and has_halide and forms_aryl_amine
    
    def _is_generic_coupling(self, rxn_smiles: str) -> bool:
        """
        Generic coupling reaction detection.
        """
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        reactants = reactants_smiles.split(".")
        
        # Simple heuristic: multiple reactants combining to one product
        # with reasonable molecular weight increase
        if len(reactants) == self.fragment_count:
            return True
        
        return False
    
    def _fragments_have_complexity(self, reactant_smiles: List[str]) -> bool:
        """
        Check if fragments have sufficient complexity to be considered
        meaningful synthetic intermediates rather than simple starting materials.
        """
        for smiles in reactant_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
                
            # Simple complexity metrics
            num_atoms = mol.GetNumAtoms()
            num_rings = mol.GetRingInfo().NumRings()
            
            # At least one fragment should have reasonable complexity
            if num_atoms > 8 or nu
