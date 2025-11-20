"""Generated evaluation code for: Convergent synthesis via amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use amide coupling reactions.
    Checks if an amide formation reaction occurs at a specified depth with
    the required number of fragments.
    """
    
    def __init__(self, config: Dict):
        self.coupling_reaction = config["parameters"]["coupling_reaction"]
        self.fragment_count = config["parameters"]["fragment_count"]
        self.target_depth = config["parameters"]["coupling_depth"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        else:
            # Score based on how close the coupling is to target depth
            # Late-stage coupling (closer to target depth) scores higher
            depth_penalty = abs(x - self.target_depth / 10.0)  # Normalize depth
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents an amide coupling with correct fragment count
        """
        metadata = d.get("metadata", {})
        
        # Check if reaction SMILES exists
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        # Parse reactants
        reactants = [r.strip() for r in reactant_smiles.split(".") if r.strip()]
        
        # Check if we have the expected number of fragments
        if len(reactants) != self.fragment_count:
            return False
            
        # Check for amide formation
        if not self._is_amide_formation(product_smiles, reactants):
            return False
            
        return True
    
    def _is_amide_formation(self, product_smiles: str, reactant_smiles_list: List[str]) -> bool:
        """
        Detect if this is an amide bond formation reaction
        """
        try:
            from rdkit import Chem
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles_list]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Count amide bonds in product vs reactants
            amide_pattern = Chem.MolFromSmarts("[C](=O)[NH,N]")
            if not amide_pattern:
                return False
                
            product_amides = len(product_mol.GetSubstructMatches(amide_pattern))
            reactant_amides = sum(len(mol.GetSubstructMatches(amide_pattern)) for mol in reactant_mols)
            
            # Check if new amide bond was formed
            if product_amides <= reactant_amides:
                return False
            
            # Additional check: look for typical amide coupling patterns
            # Carboxylic acid + amine patterns
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            
            has_carboxylic_acid = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in reactant_mols)
            has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactant_mols)
            
            # Also check for activated esters (common in amide coupling)
            activated_ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C,N,S]")
            has_activated_ester = any(mol.HasSubstructMatch(activated_ester_pattern) for mol in reactant_mols)
            
            return (has_carboxylic_acid or has_activated_ester) and has_amine
            
        except Exception:
            return False
