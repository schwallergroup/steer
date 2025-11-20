"""Generated evaluation code for: Pyrazole ring formation via condensation chemistry"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazoleFormationCondensation(BaseScoring):
    """
    Evaluates synthesis routes for pyrazole ring formation via condensation chemistry.
    Detects the formation of pyrazole rings through condensation reactions involving
    hydrazines and carbonyl compounds or enaminone systems.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config.get("step_number", -1)
        self.condition_type = config.get("scoring_type", "depth")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.target_step > 0:
            # Penalize deviation from target step
            step_penalty = abs(x * 10 - self.target_step) * 0.5
            return max(0, 10 - step_penalty)
        else:
            # Earlier formation is better (late-stage complexity)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves pyrazole formation via condensation"""
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
            
            # Parse molecules
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            # Check for pyrazole formation
            return self._is_pyrazole_condensation(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _is_pyrazole_condensation(self, reactants, products) -> bool:
        """
        Check if reaction forms pyrazole ring via condensation
        """
        # Pyrazole core pattern
        pyrazole_pattern = Chem.MolFromSmarts("[nH]1ncc[cH]1")
        n_methyl_pyrazole_pattern = Chem.MolFromSmarts("n1(C)ncc[cH]1")
        
        # Check if products contain pyrazole rings
        pyrazole_in_products = False
        for prod in products:
            if (prod.HasSubstructMatch(pyrazole_pattern) or 
                prod.HasSubstructMatch(n_methyl_pyrazole_pattern)):
                pyrazole_in_products = True
                break
        
        if not pyrazole_in_products:
            return False
        
        # Check if reactants contain pyrazole (if so, it's not formation)
        for react in reactants:
            if (react.HasSubstructMatch(pyrazole_pattern) or 
                react.HasSubstructMatch(n_methyl_pyrazole_pattern)):
                return False
        
        # Check for condensation chemistry patterns
        return self._has_condensation_patterns(reactants, products)
    
    def _has_condensation_patterns(self, reactants, products) -> bool:
        """
        Check for typical condensation patterns in pyrazole formation
        """
        # Hydrazine patterns (including N-methylhydrazine)
        hydrazine_patterns = [
            Chem.MolFromSmarts("NN"),  # Basic hydrazine
            Chem.MolFromSmarts("N(C)N"),  # N-methylhydrazine
            Chem.MolFromSmarts("N(N)[CH3]"),  # Alternative N-methylhydrazine
        ]
        
        # Carbonyl/enaminone patterns
        carbonyl_patterns = [
            Chem.MolFromSmarts("C=O"),  # Carbonyl
            Chem.MolFromSmarts("C(=O)CC(=O)"),  # 1,3-dicarbonyl
            Chem.MolFromSmarts("C=CN"),  # Enamine
            Chem.MolFromSmarts("N=CC=O"),  # Enaminone partial
        ]
        
        # Check for hydrazine in reactants
        has_hydrazine = False
        for react in reactants:
            for pattern in hydrazine_patterns:
                if pattern and react.HasSubstructMatch(pattern):
                    has_hydrazine = True
                    break
            if has_hydrazine:
                break
        
        # Check for carbonyl/enaminone in reactants
        has_carbonyl = False
        for react in reactants:
            for pattern in carbonyl_patterns:
                if pattern and react.HasSubstructMatch(pattern):
                    has_carbonyl = True
                    break
            if has_carbonyl:
                break
        
        # Look for water elimination (condensation signature)
        reactant_count = sum(mol.GetNumAtoms() for mol in reactants)
        product_count = sum(mol.GetNumAtoms() for mol in products)
        
        # Typical condensation loses H2O (3 atoms)
        water_loss = (reactant_count - product_count) >= 3
        
        return has_hydrazine and (has_carbonyl or water_loss)
