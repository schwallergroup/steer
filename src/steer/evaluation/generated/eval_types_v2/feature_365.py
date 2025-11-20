"""Generated evaluation code for: Convergent synthesis via two complex fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two complex fragments are coupled.
    Checks for nucleophilic aromatic substitution coupling of piperidine derivative 
    and carbazole lactam fragments at a specific depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.target_coupling_step = config["parameters"]["coupling_step"]
        self.fragment_complexity = config["parameters"]["fragment_complexity"]
        
        # Define SMARTS patterns for complex fragments
        self.piperidine_pattern = Chem.MolFromSmarts("[#6]1[#6][#7][#6][#6][#6]1")  # Piperidine core
        self.carbazole_pattern = Chem.MolFromSmarts("c1ccc2c(c1)[nH]c1ccccc12")  # Carbazole core
        self.lactam_pattern = Chem.MolFromSmarts("[#6](=[#8])[#7]")  # Lactam functional group
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        
        # Score based on how close the coupling step is to target depth
        depth_score = max(0, 1 - abs(x - (self.target_coupling_step / 10.0)))
        return depth_score * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents convergent coupling of complex fragments"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            if len(reactants_smiles) != self.fragment_count:
                return False
                
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if reactants are complex fragments
            fragment_matches = []
            for reactant in reactant_mols:
                complexity_score = self._assess_fragment_complexity(reactant)
                fragment_matches.append(complexity_score >= 2)  # At least 2 complex features
            
            # Must have exactly 2 complex fragments
            if sum(fragment_matches) != 2:
                return False
            
            # Check for nucleophilic aromatic substitution pattern
            # Look for aromatic carbon with leaving group being replaced
            nas_pattern = Chem.MolFromSmarts("c[F,Cl,Br,I,N+]")  # Aromatic carbon with leaving group
            
            nas_in_reactants = any(reactant.HasSubstructMatch(nas_pattern) for reactant in reactant_mols)
            nucleophile_present = any(self._has_nucleophile(reactant) for reactant in reactant_mols)
            
            return nas_in_reactants and nucleophile_present
            
        except Exception:
            return False
    
    def _assess_fragment_complexity(self, mol) -> int:
        """Assess complexity of a molecular fragment"""
        complexity_features = 0
        
        # Check for piperidine derivative
        if mol.HasSubstructMatch(self.piperidine_pattern):
            complexity_features += 1
            
        # Check for carbazole core
        if mol.HasSubstructMatch(self.carbazole_pattern):
            complexity_features += 1
            
        # Check for lactam functionality
        if mol.HasSubstructMatch(self.lactam_pattern):
            complexity_features += 1
            
        # Check for additional complexity indicators
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() >= 2:  # Multiple rings
            complexity_features += 1
            
        # Check for heteroatoms
        heteroatom_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        if heteroatom_count >= 2:
            complexity_features += 1
            
        return complexity_features
    
    def _has_nucleophile(self, mol) -> bool:
        """Check if molecule contains nucleophilic centers"""
        # Common nucleophile patterns
        amine_pattern = Chem.MolFromSmarts("[N;!$(N=*);!$(N#*)]")  # Amine nitrogen
        alkoxide_pattern = Chem.MolFromSmarts("[O-]")  # Alkoxide oxygen
        enolate_pattern = Chem.MolFromSmarts("[C-][C]=O")  # Enolate carbon
        
        return (mol.HasSubstructMatch(amine_pattern) or 
                mol.HasSubstructMatch(alkoxide_pattern) or 
                mol.HasSubstructMatch(enolate_pattern))
