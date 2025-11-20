"""Generated evaluation code for: Late Fischer indole synthesis with cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FischerIndoleSynthesis(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Fischer indole synthesis with cyclization.
    
    Fischer indole synthesis involves the reaction of a phenylhydrazine derivative 
    with a ketone or aldehyde under acidic conditions to form an indole ring.
    This class specifically looks for patterns indicating Fischer indole formation
    combined with additional cyclization reactions.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")  # "early", "late", or "any"
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage preference, higher depth fractions get better scores.
        """
        if x < 0:
            return 0  # Fischer indole synthesis doesn't occur
        
        if self.timing_preference == "late":
            # Late-stage reactions get higher scores
            return 10 * x  # x ranges from 0 to 1, later reactions get higher scores
        elif self.timing_preference == "early":
            # Early-stage reactions get higher scores  
            return 10 * (1 - x)
        else:  # "any"
            return 10  # Full score regardless of timing
            
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents Fischer indole synthesis with cyclization.
        """
        metadata = d.get("metadata", {})
        
        # Check for mapped reaction SMILES
        mapped_rxn = metadata.get("mapped_reaction_smiles")
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
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return False
                
            return self._detect_fischer_indole_with_cyclization(reactant_mols, product_mols)
            
        except Exception:
            return False
            
    def _detect_fischer_indole_with_cyclization(self, reactants, products) -> bool:
        """
        Detect Fischer indole synthesis patterns with additional cyclization.
        
        Fischer indole synthesis key features:
        1. Reactants should contain phenylhydrazine derivative pattern
        2. Reactants should contain carbonyl compound (ketone/aldehyde)
        3. Products should contain indole core
        4. Additional cyclization: product should have more rings than expected from simple Fischer indole
        """
        # Phenylhydrazine derivative pattern: aromatic ring connected to N-N
        phenylhydrazine_pattern = Chem.MolFromSmarts("c1ccccc1NN")
        
        # Carbonyl patterns (ketone or aldehyde)
        ketone_pattern = Chem.MolFromSmarts("CC(=O)C")
        aldehyde_pattern = Chem.MolFromSmarts("C(=O)")
        
        # Indole core pattern
        indole_pattern = Chem.MolFromSmarts("c1ccc2[nH]ccc2c1")
        
        # Check for phenylhydrazine derivative in reactants
        has_phenylhydrazine = any(mol.HasSubstructMatch(phenylhydrazine_pattern) for mol in reactants)
        
        # Check for carbonyl in reactants
        has_carbonyl = any(mol.HasSubstructMatch(ketone_pattern) or mol.HasSubstructMatch(aldehyde_pattern) 
                          for mol in reactants)
        
        # Check for indole formation in products
        has_indole_product = any(mol.HasSubstructMatch(indole_pattern) for mol in products)
        
        if not (has_phenylhydrazine and has_carbonyl and has_indole_product):
            return False
            
        # Check for additional cyclization by comparing ring counts
        reactant_ring_count = sum(mol.GetRingInfo().NumRings() for mol in reactants)
        product_ring_count = sum(mol.GetRingInfo().NumRings() for mol in products)
        
        # Fischer indole synthesis typically forms one new ring (the pyrrole part of indole)
        # Additional cyclization should result in more rings than just the indole formation
        expected_rings_simple_fischer = reactant_ring_count + 1
        has_additional_cyclization = product_ring_count > expected_rings_simple_fischer
        
        return has_additional_cyclization
