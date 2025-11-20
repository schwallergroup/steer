"""Generated evaluation code for: Convergent synthesis via three major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates if a synthesis route follows a convergent strategy by coupling
    a specified number of major fragments using specific coupling reactions.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.coupling_reactions = config["parameters"]["coupling_reactions"]
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
        # Define SMARTS patterns for coupling reactions
        self.reaction_patterns = {
            "Buchwald-Hartwig": {
                "C-N": "[c:1][N:2]",  # Aromatic C-N bond formation
                "breaking": "[c:1][Br,I,Cl].[N:2][H]"  # Aryl halide + amine
            },
            "Arbuzov": {
                "P-C": "[P:1][C:2]",  # P-C bond formation
                "breaking": "[P:1]([O])([O])[O].[C:2][Br,I,Cl]"  # Phosphite + alkyl halide
            }
        }
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10)"""
        if x < 0:
            return 0  # Convergent coupling not found
        
        if self.condition_type == "bool":
            return 10 if x >= 0 else 0
        else:
            # Earlier convergent coupling is better (lower depth)
            if x <= self.target_depth:
                return 10
            else:
                return max(0, 10 - (x - self.target_depth) * 20)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling step"""
        metadata = d.get("metadata", {})
        
        # Check if we have mapped reaction SMILES
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1].split(".")
        
        # Must have at least the specified number of fragments as reactants
        if len(reactants_smiles) < self.fragment_count:
            return False
            
        # Check if this is one of the specified coupling reactions
        return self._is_coupling_reaction(product_smiles, reactants_smiles)
    
    def _is_coupling_reaction(self, product_smiles: str, reactants_smiles: list) -> bool:
        """Check if the reaction matches one of the specified coupling patterns"""
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            for reaction_type in self.coupling_reactions:
                if reaction_type in self.reaction_patterns:
                    if self._matches_coupling_pattern(product_mol, reactant_mols, reaction_type):
                        return True
                        
            return False
            
        except Exception:
            return False
    
    def _matches_coupling_pattern(self, product_mol, reactant_mols, reaction_type: str) -> bool:
        """Check if the reaction matches a specific coupling pattern"""
        patterns = self.reaction_patterns.get(reaction_type, {})
        
        if reaction_type == "Buchwald-Hartwig":
            # Look for C-N bond formation between aromatic carbon and nitrogen
            bond_pattern = Chem.MolFromSmarts(patterns["C-N"])
            if not product_mol.HasSubstructMatch(bond_pattern):
                return False
                
            # Check reactants have aryl halide and amine components
            has_aryl_halide = any(mol.HasSubstructMatch(Chem.MolFromSmarts("[c][Br,I,Cl]")) 
                                for mol in reactant_mols)
            has_amine = any(mol.HasSubstructMatch(Chem.MolFromSmarts("[N][H]")) 
                          for mol in reactant_mols)
            
            return has_aryl_halide and has_amine
            
        elif reaction_type == "Arbuzov":
            # Look for P-C bond formation
            bond_pattern = Chem.MolFromSmarts(patterns["P-C"])
            if not product_mol.HasSubstructMatch(bond_pattern):
                return False
                
            # Check reactants have phosphite and alkyl halide
            has_phosphite = any(mol.HasSubstructMatch(Chem.MolFromSmarts("[P]([O])([O])[O]")) 
                              for mol in reactant_mols)
            has_alkyl_halide = any(mol.HasSubstructMatch(Chem.MolFromSmarts("[C][Br,I,Cl]")) 
                                 for mol in reactant_mols)
            
            return has_phosphite and has_alkyl_halide
            
        return False
