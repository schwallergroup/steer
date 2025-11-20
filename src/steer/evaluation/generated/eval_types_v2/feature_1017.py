"""Generated evaluation code for: Triple protecting group strategy: Boc-PNB-TBDMS"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TripleProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the simultaneous use of three specific protecting groups:
    Boc (tert-butyloxycarbonyl), PNB (para-nitrobenzyl), and TBDMS (tert-butyldimethylsilyl)
    to protect amine, carboxylic acid, and alcohol functionalities respectively.
    """
    
    def __init__(self, config):
        self.required_groups = config["parameters"]["protecting_groups"]  # ["Boc", "PNB", "TBDMS"]
        self.simultaneous_count = config["parameters"]["simultaneous_count"]  # 3
        self.functional_groups = config["parameters"]["functional_groups"]  # ["amine", "carboxylic_acid", "alcohol"]
        
        # SMARTS patterns for detecting protecting group formation/removal
        self.protecting_patterns = {
            "Boc": "[NX3:1]C(=O)OC(C)(C)C",  # Boc-protected amine
            "PNB": "[OX2:1]Cc1ccc(cc1)[N+](=O)[O-]",  # PNB-protected carboxylic acid/alcohol
            "TBDMS": "[OX2:1][Si](C)(C)C(C)(C)C"  # TBDMS-protected alcohol
        }
        
        # Patterns for detecting protecting group installation reactions
        self.protection_rxn_patterns = {
            "Boc": "[NX3:1][H]>>[NX3:1]C(=O)OC(C)(C)C",  # Boc installation
            "PNB": "[OH:1]>>[O:1]Cc1ccc(cc1)[N+](=O)[O-]",  # PNB installation  
            "TBDMS": "[OH:1]>>[O:1][Si](C)(C)C(C)(C)C"  # TBDMS installation
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check if we have simultaneous presence of all three protecting groups
        has_simultaneous_protection = self.check_simultaneous_protection(reactions)
        
        return has_simultaneous_protection, len(reactions)

    def check_simultaneous_protection(self, reactions) -> bool:
        """Check if there's a point in the synthesis where all three protecting groups are present"""
        
        # Track protecting group status through the reaction sequence
        protection_status = {"Boc": False, "PNB": False, "TBDMS": False}
        max_simultaneous = 0
        
        for rxn in reactions:
            # Update protection status based on current reaction
            self.update_protection_status(rxn, protection_status)
            
            # Count current simultaneous protecting groups
            current_count = sum(protection_status.values())
            max_simultaneous = max(max_simultaneous, current_count)
            
            # Early return if we achieve the target
            if current_count >= self.simultaneous_count:
                return True
        
        return False

    def update_protection_status(self, rxn, status):
        """Update the protection status based on the current reaction"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for protection installation
            for group, pattern in self.protecting_patterns.items():
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol is None:
                    continue
                    
                # Check if protecting group appears in products but not reactants
                reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
                product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
                
                reactant_matches = any(mol and mol.HasSubstructMatch(pattern_mol) 
                                     for mol in reactant_mols if mol)
                product_matches = any(mol and mol.HasSubstructMatch(pattern_mol) 
                                    for mol in product_mols if mol)
                
                # Protection installation: appears in products but not reactants
                if product_matches and not reactant_matches:
                    status[group] = True
                    
                # Protection removal: appears in reactants but not products  
                elif reactant_matches and not product_matches:
                    status[group] = False
                    
        except Exception:
            # Skip malformed reactions
            pass

    def detect_protection_reaction(self, rxn, group_name):
        """Detect if a specific protecting group installation occurs in this reaction"""
        try:
            pattern = self.protecting_patterns.get(group_name)
            if not pattern:
                return False
                
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return False
                
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants, products = rxn_parts
            
            # Parse molecules
            reactant_mols = []
            for r in reactants.split("."):
                mol = Chem.MolFromSmiles(r.strip())
                if mol:
                    reactant_mols.append(mol)
                    
            product_mols = []
            for p in products.split("."):
                mol = Chem.MolFromSmiles(p.strip())
                if mol:
                    product_mols.append(mol)
            
            # Check if protecting group is formed (present in products but not reactants)
            reactant_has_pattern = any(mol.HasSubstructMatch(pattern_mol) for mol in reactant_mols)
            product_has_pattern = any(mol.HasSubstructMatch(pattern_mol) for mol in product_mols)
            
            return product_has_pattern and not reactant_has_pattern
            
        except Exception:
            return False

    def route_scoring(self, x) -> float:
        """Convert condition result to 0-10 score"""
        if x < 0:
            return 0  # Condition not met
        else:
            return 10  # Perfect score when triple protection strategy is achieved
