"""Generated evaluation code for: Protecting group swap strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates whether a route involves a protecting group swap strategy for a specific functional group.
    Checks if the specified protection sequence occurs: deprotection followed by re-protection.
    """
    
    def __init__(self, config):
        self.functional_group = config["functional_group"]
        self.protection_sequence = config["protection_sequence"]
        self.involves_swap = config.get("involves_swap", True)
        
        # Define SMARTS patterns for functional groups and protecting groups
        self.fg_patterns = {
            "carboxylic_acid": "[CX3](=O)[OX2H1]",
            "amine": "[NX3;H2,H1;!$(NC=O)]",
            "alcohol": "[OX2H1]"
        }
        
        self.protection_patterns = {
            "ethyl_ester": "[CX3](=O)[OX2][CH2][CH3]",
            "tert_butyl_ester": "[CX3](=O)[OX2]C(C)(C)C",
            "free_acid": "[CX3](=O)[OX2H1]",
            "boc": "[NX3][CX3](=O)[OX2]C(C)(C)C",
            "cbz": "[NX3][CX3](=O)[OX2][CH2]c1ccccc1",
            "benzyl": "[OX2][CH2]c1ccccc1",
            "silyl": "[OX2][SiX4]"
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track the sequence of protecting group transformations
        pg_transformations = []
        
        for rxn in reactions:
            transformation = self.detect_protection_change(rxn)
            if transformation:
                pg_transformations.append(transformation)
        
        # Check if the specified protection sequence occurs
        swap_detected = self.detect_swap_sequence(pg_transformations)
        
        condition_met = swap_detected == self.involves_swap
        return condition_met, len(reactions)

    def detect_protection_change(self, rxn):
        """Detect if a reaction involves protection/deprotection of the target functional group."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return None
            
            # Check what protecting groups are present before and after
            reactant_pgs = set()
            product_pgs = set()
            
            for mol in reactants:
                reactant_pgs.update(self.identify_protecting_groups(mol))
            
            for mol in products:
                product_pgs.update(self.identify_protecting_groups(mol))
            
            # Determine the type of transformation
            if reactant_pgs != product_pgs:
                lost_pgs = reactant_pgs - product_pgs
                gained_pgs = product_pgs - reactant_pgs
                
                if lost_pgs and gained_pgs:
                    return ("swap", list(lost_pgs)[0], list(gained_pgs)[0])
                elif lost_pgs:
                    return ("deprotection", list(lost_pgs)[0], None)
                elif gained_pgs:
                    return ("protection", None, list(gained_pgs)[0])
            
            return None
            
        except Exception:
            return None

    def identify_protecting_groups(self, mol):
        """Identify which protecting groups are present in a molecule."""
        protecting_groups = set()
        
        for pg_name, pattern in self.protection_patterns.items():
            try:
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    protecting_groups.add(pg_name)
            except Exception:
                continue
                
        return protecting_groups

    def detect_swap_sequence(self, transformations):
        """Check if the transformations match the specified protection sequence."""
        if len(self.protection_sequence) < 2:
            return False
            
        # Look for the specific sequence pattern
        sequence_matches = 0
        expected_sequence = list(self.protection_sequence)
        
        for i, transformation in enumerate(transformations):
            if transformation is None:
                continue
                
            trans_type, from_pg, to_pg = transformation
            
            if trans_type == "swap":
                # Direct swap from one protecting group to another
                if (from_pg in expected_sequence and 
                    to_pg in expected_sequence and
                    expected_sequence.index(to_pg) > expected_sequence.index(from_pg)):
                    sequence_matches += 1
            
            elif trans_type == "deprotection" and i + 1 < len(transformations):
                # Deprotection followed by protection
                next_trans = transformations[i + 1]
                if (next_trans and next_trans[0] == "protection" and
                    from_pg in expected_sequence and
                    next_trans[2] in expected_sequence):
                    if (expected_sequence.index(next_trans[2]) > 
                        expected_sequence.index(from_pg)):
                        sequence_matches += 1
        
        # Consider it a swap strategy if we found the expected transformation
        return sequence_matches > 0

    def route_scoring(self, x):
        """Score based on whether the protecting group swap strategy is detected."""
        if x < 0:
            return 5  # Neutral score if condition not evaluated
        else:
            return 10 - x * 10  # Earlier occurrence scores higher
