"""Generated evaluation code for: Early stage tricyclic core assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TricyclicCoreAssembly(BaseScoring):
    """
    Evaluates early stage tricyclic core assembly in synthesis routes.
    Detects when a tricyclic structure is formed and penalizes late formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Tricyclic core formation doesn't happen
        else:
            # Early formation is better, so invert the depth fraction
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if tricyclic core formation occurs in this reaction step.
        """
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
        products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
        
        # Filter out None molecules
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        if not reactants or not products:
            return False
        
        # Create SMARTS pattern for tricyclic detection
        tricyclic_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if tricyclic_pattern is None:
            return False
        
        # Count tricyclic cores in reactants and products
        reactant_tricycles = sum(len(mol.GetSubstructMatches(tricyclic_pattern)) 
                               for mol in reactants)
        product_tricycles = sum(len(mol.GetSubstructMatches(tricyclic_pattern)) 
                              for mol in products)
        
        # Check for tricyclic core formation (increase in tricyclic count)
        if self.direction == "formation":
            return product_tricycles > reactant_tricycles
        elif self.direction == "breaking":
            return reactant_tricycles > product_tricycles
        
        return False
