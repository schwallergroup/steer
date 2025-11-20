"""Generated evaluation code for: Late pyrazole ring formation via double cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazoleRingFormation(BaseScoring):
    """
    Evaluates late-stage pyrazole ring formation via double cyclization.
    
    Detects formation of pyrazoledione rings through double condensation
    reactions where hydrazine reacts with adjacent ester carbonyls to form
    the heterocyclic ring system.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_mechanism = config["parameters"]["formation_mechanism"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Later formation is better for late-stage strategy
            return 1 - x
        else:
            # Early formation preferred
            return x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms the target pyrazole ring via double cyclization.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol is not None:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check if pyrazole ring is formed (present in products but not reactants)
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            
            if not ring_in_products or ring_in_reactants:
                return False
            
            # Check for double condensation mechanism
            if self.formation_mechanism == "double_condensation":
                return self._detect_double_condensation(reactants, products)
            
            return True
            
        except Exception:
            return False
    
    def _detect_double_condensation(self, reactants, products) -> bool:
        """
        Detect if the reaction involves double condensation mechanism.
        Look for hydrazine + dicarbonyl -> pyrazole + water pattern.
        """
        # Hydrazine pattern
        hydrazine_pattern = Chem.MolFromSmarts("[NH2][NH2]")
        
        # Adjacent carbonyl pattern (ester or similar)
        dicarbonyl_pattern = Chem.MolFromSmarts("[C](=O)[C](=O)")
        
        # Check if reactants contain hydrazine and dicarbonyl components
        has_hydrazine = any(mol.HasSubstructMatch(hydrazine_pattern) for mol in reactants)
        has_dicarbonyl = any(mol.HasSubstructMatch(dicarbonyl_pattern) for mol in reactants)
        
        # Count total atoms in reactants vs products to detect condensation
        reactant_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactants)
        product_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in products)
        
        # In double condensation, we typically lose small molecules (like 2 H2O)
        atom_loss = reactant_heavy_atoms - product_heavy_atoms
        
        return has_hydrazine and has_dicarbonyl and atom_loss >= 2
