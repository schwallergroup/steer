"""Generated evaluation code for: Late stage aldehyde to carboxylic acid oxidation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAldehydeOxidation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage aldehyde to carboxylic acid oxidation.
    Checks if an aldehyde functional group is oxidized to carboxylic acid in the later
    stages of the synthesis, which is preferred for sensitive molecular frameworks.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Oxidation doesn't happen
        else:
            # Late-stage oxidation is better (higher depth fraction preferred)
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Reward oxidations that occur at or after target depth
                if x >= self.target_depth:
                    return 10  # Perfect score for late-stage
                else:
                    # Linear penalty for early oxidation
                    return 10 * (x / self.target_depth)
    
    def hit_condition(self, d):
        """Check if this reaction converts aldehyde to carboxylic acid via oxidation"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            if len(rxn) != 2:
                return False
                
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Define aldehyde and carboxylic acid patterns
            aldehyde_pattern = Chem.MolFromSmarts("[CH1](=O)")  # Aldehyde carbon
            carboxylic_acid_pattern = Chem.MolFromSmarts("[CH0](=O)[OH]")  # Carboxylic acid
            
            # Check if reactants contain aldehyde
            has_aldehyde_reactant = any(mol.HasSubstructMatch(aldehyde_pattern) for mol in reactants)
            
            # Check if products contain carboxylic acid
            has_carboxylic_product = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in products)
            
            # Additional check: verify the transformation using atom mapping if available
            if has_aldehyde_reactant and has_carboxylic_product:
                return self._verify_aldehyde_to_acid_transformation(reactants, products)
                
            return False
            
        except Exception:
            return False
    
    def _verify_aldehyde_to_acid_transformation(self, reactants, products):
        """Verify that the same carbon is involved in aldehyde->carboxylic acid conversion"""
        try:
            # Look for atom mapping to confirm the transformation
            aldehyde_carbons = []
            carboxylic_carbons = []
            
            # Find mapped aldehyde carbons in reactants
            for mol in reactants:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        # Check if this mapped atom is part of aldehyde
                        if atom.GetSymbol() == 'C':
                            neighbors = [n for n in atom.GetNeighbors()]
                            if len(neighbors) >= 1:
                                # Check for C=O pattern with at least one H
                                oxygen_neighbors = [n for n in neighbors if n.GetSymbol() == 'O' and any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds() if bond.GetOtherAtom(atom) == n)]
                                hydrogen_count = atom.GetTotalNumHs()
                                if oxygen_neighbors and hydrogen_count >= 1:
                                    aldehyde_carbons.append(atom.GetAtomMapNum())
            
            # Find mapped carboxylic acid carbons in products
            for mol in products:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0 and atom.GetSymbol() == 'C':
                        neighbors = [n for n in atom.GetNeighbors()]
                        # Check for COOH pattern
                        oxygen_neighbors = [n for n in neighbors if n.GetSymbol() == 'O']
                        if len(oxygen_neighbors) >= 2:  # Should have C=O and C-OH
                            carboxylic_carbons.append(atom.GetAtomMapNum())
            
            # Check if any aldehyde carbon maps to carboxylic acid carbon
            return bool(set(aldehyde_carbons) & set(carboxylic_carbons))
            
        except Exception:
            # If atom mapping verification fails, return True if basic pattern match succeeded
            return True
