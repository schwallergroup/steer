"""Generated evaluation code for: Late stage pyrrolidine ring formation via intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates late-stage pyrrolidine ring formation via intramolecular cyclization.
    Checks if a pyrrolidine ring (C1CCNC1) is formed through intramolecular cyclization
    and rewards later-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_type = config["parameters"]["formation_type"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is better (closer to 1.0 depth fraction)
            # Score increases as depth fraction approaches 1
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a pyrrolidine ring via intramolecular cyclization"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            product = Chem.MolFromSmiles(rxn[0])
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains the pyrrolidine ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if this is intramolecular cyclization (single reactant -> ring in product)
            if len(reactants) != 1:
                return False
            
            reactant = reactants[0]
            
            # Reactant should not have the complete ring
            if reactant.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if reactant has the open-chain precursor
            # Look for nitrogen and carbon atoms that could cyclize
            reactant_atoms = [atom.GetAtomMapNum() for atom in reactant.GetAtoms() 
                            if atom.GetAtomMapNum() > 0]
            product_atoms = [atom.GetAtomMapNum() for atom in product.GetAtoms() 
                           if atom.GetAtomMapNum() > 0]
            
            # Atoms should be conserved (intramolecular reaction)
            if set(reactant_atoms) != set(product_atoms):
                return False
            
            # Additional check: look for N-C bond formation pattern
            # Find pyrrolidine rings in product and check if the N-C bond was formed
            matches = product.GetSubstructMatches(self.ring_pattern)
            for match in matches:
                ring_atoms = [product.GetAtomWithIdx(idx) for idx in match]
                nitrogen_idx = None
                for atom in ring_atoms:
                    if atom.GetSymbol() == 'N':
                        nitrogen_idx = atom.GetIdx()
                        break
                
                if nitrogen_idx is not None:
                    nitrogen = product.GetAtomWithIdx(nitrogen_idx)
                    # Check if this nitrogen forms new bonds compared to reactant
                    product_map = nitrogen.GetAtomMapNum()
                    if product_map > 0:
                        # Find corresponding atom in reactant
                        reactant_n = None
                        for atom in reactant.GetAtoms():
                            if atom.GetAtomMapNum() == product_map:
                                reactant_n = atom
                                break
                        
                        if reactant_n:
                            # Compare bond counts - should increase due to cyclization
                            if nitrogen.GetDegree() > reactant_n.GetDegree():
                                return True
            
            return False
            
        except Exception:
            return False
