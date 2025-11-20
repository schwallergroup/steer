"""Generated evaluation code for: Late stage aldehyde formation via ester reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAldehydeFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage aldehyde formation via ester reduction.
    Checks if an ester is reduced to an aldehyde, preferably as the final step.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "final_step")
        self.reagent_type = config.get("reagent_type", "DIBAL-H")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing == "final_step":
            # Reward reactions closer to the end (depth 0 is best)
            return 1 - x if x >= 0 else 0
        else:
            # For general late-stage, prefer depth < 0.3
            if x <= 0.3:
                return 1.0
            else:
                return max(0, 1 - (x - 0.3) * 2)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents ester to aldehyde reduction.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return False
            
            # Define SMARTS patterns
            ester_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3][C:4]")  # Ester group
            aldehyde_pattern = Chem.MolFromSmarts("[C:1]=[O:2]")  # Aldehyde group
            
            # Check if reactant contains ester and product contains aldehyde
            has_ester_reactant = any(mol.HasSubstructMatch(ester_pattern) for mol in reactant_mols)
            has_aldehyde_product = any(mol.HasSubstructMatch(aldehyde_pattern) for mol in product_mols)
            
            if not (has_ester_reactant and has_aldehyde_product):
                return False
            
            # Additional check: verify the transformation by comparing atom maps
            return self._verify_ester_to_aldehyde_transformation(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _verify_ester_to_aldehyde_transformation(self, reactants, products):
        """
        Verify that an ester carbon becomes an aldehyde carbon by checking atom maps.
        """
        try:
            ester_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3][C:4]")
            aldehyde_pattern = Chem.MolFromSmarts("[C:1]=[O:2]")
            
            # Find ester carbons in reactants
            ester_carbons = set()
            for mol in reactants:
                matches = mol.GetSubstructMatches(ester_pattern)
                for match in matches:
                    ester_carbon_idx = match[0]  # First atom in pattern is the carbonyl carbon
                    atom = mol.GetAtomWithIdx(ester_carbon_idx)
                    if atom.GetAtomMapNum() > 0:
                        ester_carbons.add(atom.GetAtomMapNum())
            
            # Find aldehyde carbons in products
            aldehyde_carbons = set()
            for mol in products:
                matches = mol.GetSubstructMatches(aldehyde_pattern)
                for match in matches:
                    aldehyde_carbon_idx = match[0]
                    atom = mol.GetAtomWithIdx(aldehyde_carbon_idx)
                    if atom.GetAtomMapNum() > 0:
                        # Verify it's actually an aldehyde (has exactly one hydrogen)
                        if atom.GetTotalNumHs() == 1:
                            aldehyde_carbons.add(atom.GetAtomMapNum())
            
            # Check if any ester carbon became an aldehyde carbon
            return bool(ester_carbons & aldehyde_carbons)
            
        except Exception:
            return False
