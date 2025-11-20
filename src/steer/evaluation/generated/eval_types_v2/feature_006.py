"""Generated evaluation code for: Late stage cascade cyclization approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CascadeCyclization(BaseScoring):
    """
    Evaluates synthesis routes for late-stage cascade cyclization reactions.
    A cascade cyclization is defined as a reaction that forms multiple rings
    in a single step, creating complex polycyclic systems.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No cascade cyclization found
        else:
            # Late-stage cyclization is better, return higher score for later occurrence
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Detects cascade cyclization by checking if multiple rings are formed
        in a single reaction step.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Count rings in reactants and products
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
            product_rings = sum(mol.GetRingInfo().NumRings() for mol in products)
            
            # Check if multiple rings (≥2) are formed in this step
            rings_formed = product_rings - reactant_rings
            
            # Also check for specific cascade cyclization patterns
            cascade_patterns = self._detect_cascade_patterns(reactants, products)
            
            return rings_formed >= 2 or cascade_patterns
            
        except Exception:
            return False
    
    def _detect_cascade_patterns(self, reactants, products) -> bool:
        """
        Detect common cascade cyclization patterns like:
        - Diels-Alder cascades
        - Robinson annulation cascades  
        - Radical cascade cyclizations
        """
        # Pattern for detecting tricyclic systems that could result from cascades
        tricyclic_patterns = [
            "[R]1~[R]~[R]~[R]2~[R]~[R]~[R]3~[R]~[R]~[R]~[R]~[R]~1~[R]~2~3",  # General tricyclic
            "C1CC2CCC3CCCC(C1)C23",  # Steroid-like cascade product
            "C1CCC2C(C1)CCC3CCCCC23"   # Decalin-type cascade product
        ]
        
        # Check if products contain complex polycyclic patterns not in reactants
        for product in products:
            product_ring_count = product.GetRingInfo().NumRings()
            if product_ring_count >= 3:  # Focus on complex polycyclic products
                # Check if any reactant already contains this complexity
                reactant_max_rings = max((mol.GetRingInfo().NumRings() for mol in reactants), default=0)
                if product_ring_count > reactant_max_rings + 1:  # Multiple rings formed
                    return True
                    
                # Check for specific tricyclic patterns
                for pattern in tricyclic_patterns:
                    try:
                        patt_mol = Chem.MolFromSmarts(pattern)
                        if patt_mol and product.HasSubstructMatch(patt_mol):
                            # Check if this pattern exists in reactants
                            pattern_in_reactants = any(
                                reactant.HasSubstructMatch(patt_mol) 
                                for reactant in reactants
                            )
                            if not pattern_in_reactants:
                                return True
                    except:
                        continue
        
        return False
