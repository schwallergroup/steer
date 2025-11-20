"""Generated evaluation code for: Convergent synthesis via two distinct fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy via coupling of distinct fragments.
    
    Checks for nucleophilic aromatic substitution reactions that combine 
    two separate synthetic fragments at a specified stage of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.coupling_reaction = config["parameters"]["coupling_reaction"]
        self.coupling_stage = config["parameters"]["coupling_stage"]
        
        # SMARTS pattern for nucleophilic aromatic substitution
        # Aromatic carbon with electron-withdrawing group being attacked by nucleophile
        self.snar_pattern = "[c:1][N,O,S:2]"
        self.aromatic_halide_pattern = "[c:1][F,Cl,Br,I:2]"
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10)"""
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        
        if self.coupling_stage == "late":
            # Reward later stage coupling (lower depth fraction is better)
            return max(0, 10 * (1 - x))
        elif self.coupling_stage == "early":
            # Reward earlier stage coupling
            return max(0, 10 * x)
        else:  # "middle"
            # Reward coupling around middle of synthesis
            return max(0, 10 * (1 - 2 * abs(x - 0.5)))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents convergent coupling"""
        metadata = d.get("metadata", {})
        
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        
        # Check if this is a convergent step (multiple reactants -> single product)
        if not self._is_convergent_reaction(rxn_smiles):
            return False
            
        # Check if reaction matches SNAr pattern
        if not self._is_snar_reaction(rxn_smiles):
            return False
            
        # Check if reactants represent distinct synthetic fragments
        return self._are_distinct_fragments(rxn_smiles)
    
    def _is_convergent_reaction(self, rxn_smiles: str) -> bool:
        """Check if reaction combines multiple reactants into single product"""
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_list = reactants.split(".")
            product_list = products.split(".")
            
            # Convergent: multiple reactants, single major product
            return len(reactant_list) >= self.fragment_count and len(product_list) == 1
        except:
            return False
    
    def _is_snar_reaction(self, rxn_smiles: str) -> bool:
        """Check if reaction matches nucleophilic aromatic substitution pattern"""
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            product_mol = Chem.MolFromSmiles(products.split(".")[0])
            
            if not all(reactant_mols) or not product_mol:
                return False
            
            # Look for aromatic halide in reactants and C-N/C-O/C-S bond formation in product
            has_ar_halide = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.aromatic_halide_pattern)) 
                              for mol in reactant_mols if mol)
            
            has_snar_product = product_mol.HasSubstructMatch(Chem.MolFromSmarts(self.snar_pattern))
            
            return has_ar_halide and has_snar_product
            
        except:
            return False
    
    def _are_distinct_fragments(self, rxn_smiles: str) -> bool:
        """Check if reactants represent structurally distinct fragments"""
        try:
            reactants, _ = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if len(reactant_mols) < self.fragment_count:
                return False
            
            # Calculate molecular fingerprints to assess structural diversity
            fps = []
            for mol in reactant_mols:
                if mol and mol.GetNumHeavyAtoms() > 3:  # Ignore small molecules
                    fp = Chem.RDKFingerprint(mol)
                    fps.append(fp)
            
            if len(fps) < self.fragment_count:
                return False
            
            # Check that fragments are sufficiently different (Tanimoto < 0.7)
            from rdkit import DataStructs
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    if similarity > 0.7:  # Too similar
                        return False
            
            return True
            
        except:
            return False
