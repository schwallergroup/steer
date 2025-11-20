"""Generated evaluation code for: Friedländer annulation for quinoline core construction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FriedlanderAnnulation(BaseScoring):
    """
    Detects Friedländer annulation reactions for quinoline core construction.
    
    Friedländer annulation involves the condensation of a 2-aminoaryl carbonyl compound
    with a carbonyl compound containing an active methylene group to form quinolines.
    The reaction creates a new C-C bond and forms the pyridine ring fused to benzene.
    """
    
    def __init__(self, config: Dict):
        self.reaction_name = config["parameters"]["reaction_name"]
        self.ring_formed = config["parameters"]["ring_formed"]
        self.convergent = config["parameters"]["convergent"]
    
    def route_scoring(self, x) -> float:
        """Score based on depth where Friedländer annulation occurs."""
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Earlier application is better for convergent synthesis
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Friedländer annulation."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains quinoline core
            quinoline_pattern = Chem.MolFromSmarts("c1ccc2ncccc2c1")
            if not product.HasSubstructMatch(quinoline_pattern):
                return False
            
            # Check for characteristic Friedländer annulation pattern
            # Need 2-aminoaryl carbonyl (or equivalent) + carbonyl with active methylene
            aminoaryl_carbonyl = Chem.MolFromSmarts("c1ccc(N)c(C=O)c1")  # 2-aminobenzaldehyde pattern
            aminoaryl_ketone = Chem.MolFromSmarts("c1ccc(N)c(C(=O)C)c1")   # 2-aminoacetophenone pattern
            
            # Active methylene carbonyl patterns
            active_methylene_1 = Chem.MolFromSmarts("CC(=O)CC(=O)")  # β-diketone
            active_methylene_2 = Chem.MolFromSmarts("CC(=O)C")       # simple ketone
            active_methylene_3 = Chem.MolFromSmarts("C(=O)CC(=O)")   # malonic derivatives
            
            has_aminoaryl = False
            has_active_methylene = False
            
            # Check reactants for required components
            for reactant in reactants:
                if (reactant.HasSubstructMatch(aminoaryl_carbonyl) or 
                    reactant.HasSubstructMatch(aminoaryl_ketone)):
                    has_aminoaryl = True
                
                if (reactant.HasSubstructMatch(active_methylene_1) or
                    reactant.HasSubstructMatch(active_methylene_2) or
                    reactant.HasSubstructMatch(active_methylene_3)):
                    has_active_methylene = True
            
            # Must have both components and be convergent (2+ reactants)
            return (has_aminoaryl and has_active_methylene and 
                    len(reactants) >= 2 and self.convergent)
            
        except Exception:
            return False
