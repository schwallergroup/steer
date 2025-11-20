"""Generated evaluation code for: Phthalimide amine protecting group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PhthalimideAmineCycling(MultiRxnCondBase):
    """
    Evaluates protecting group cycling strategy where Boc group is removed and 
    replaced with phthalimide for one reaction, then deprotected again.
    Checks for the sequence: Boc deprotection -> phthalimide protection -> reaction -> phthalimide deprotection
    """
    
    def __init__(self, config):
        self.steps_protected = config.get("steps_protected", 1)
        # SMARTS patterns for different protection states
        self.boc_pattern = "[NX3;H0,H1;!$(NC=O)]C(=O)OC(C)(C)C"  # Boc-protected amine
        self.phthalimide_pattern = "[NX3]C(=O)c1ccccc1C(=O)[NX3]"  # Phthalimide-protected amine
        self.free_amine_pattern = "[NX3;H1,H2;!$(NC=O);!$(N-C=O)]"  # Free amine
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if the protecting group cycling strategy is present in the route.
        Returns (condition_met, total_reactions)
        """
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        if total_reactions < 4:  # Need at least 4 steps for cycling
            return False, total_reactions
        
        # Look for the cycling pattern in the reaction sequence
        cycling_found = self.detect_protection_cycling(reactions)
        
        return cycling_found, total_reactions
    
    def detect_protection_cycling(self, reactions) -> bool:
        """
        Detect the specific cycling pattern:
        1. Boc deprotection (Boc-amine -> free amine)
        2. Phthalimide protection (free amine -> phthalimide-amine)  
        3. Protected reaction (phthalimide present for specified steps)
        4. Phthalimide deprotection (phthalimide-amine -> free amine)
        """
        for i in range(len(reactions) - 3):
            # Check for Boc deprotection
            if self.is_boc_deprotection(reactions[i]):
                # Check for phthalimide protection in next step
                if self.is_phthalimide_protection(reactions[i + 1]):
                    # Check that phthalimide stays for specified number of steps
                    protected_steps = 0
                    for j in range(i + 2, min(i + 2 + self.steps_protected, len(reactions))):
                        if self.has_phthalimide_protection(reactions[j]):
                            protected_steps += 1
                        else:
                            break
                    
                    # Check for phthalimide deprotection after protected steps
                    deprotection_idx = i + 2 + protected_steps
                    if (protected_steps == self.steps_protected and 
                        deprotection_idx < len(reactions) and
                        self.is_phthalimide_deprotection(reactions[deprotection_idx])):
                        return True
        
        return False
    
    def is_boc_deprotection(self, rxn) -> bool:
        """Check if reaction removes Boc protecting group"""
        reactants, products = self.parse_reaction(rxn)
        
        # Look for Boc group in reactants but not in products
        boc_in_reactants = any(self.has_substructure(r, self.boc_pattern) for r in reactants)
        boc_in_products = any(self.has_substructure(p, self.boc_pattern) for p in products)
        free_amine_in_products = any(self.has_substructure(p, self.free_amine_pattern) for p in products)
        
        return boc_in_reactants and not boc_in_products and free_amine_in_products
    
    def is_phthalimide_protection(self, rxn) -> bool:
        """Check if reaction adds phthalimide protecting group"""
        reactants, products = self.parse_reaction(rxn)
        
        # Look for free amine in reactants and phthalimide in products
        free_amine_in_reactants = any(self.has_substructure(r, self.free_amine_pattern) for r in reactants)
        phthalimide_in_products = any(self.has_substructure(p, self.phthalimide_pattern) for p in products)
        
        return free_amine_in_reactants and phthalimide_in_products
    
    def has_phthalimide_protection(self, rxn) -> bool:
        """Check if phthalimide protection is present in the reaction"""
        reactants, products = self.parse_reaction(rxn)
        all_molecules = reactants + products
        
        return any(self.has_substructure(mol, self.phthalimide_pattern) for mol in all_molecules)
    
    def is_phthalimide_deprotection(self, rxn) -> bool:
        """Check if reaction removes phthalimide protecting group"""
        reactants, products = self.parse_reaction(rxn)
        
        # Look for phthalimide in reactants but not in products
        phthalimide_in_reactants = any(self.has_substructure(r, self.phthalimide_pattern) for r in reactants)
        phthalimide_in_products = any(self.has_substructure(p, self.phthalimide_pattern) for p in products)
        free_amine_in_products = any(self.has_substructure(p, self.free_amine_pattern) for p in products)
        
        return phthalimide_in_reactants and not phthalimide_in_products and free_amine_in_products
    
    def parse_reaction(self, rxn):
        """Parse reaction SMILES into reactants and products"""
        rxn_smiles = rxn["metadata"]["mapped_reaction_smiles"]
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
        products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
        
        # Filter out None molecules (invalid SMILES)
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        return reactants, products
    
    def has_substructure(self, mol, pattern):
        """Check if molecule contains the given SMARTS pattern"""
        if mol is None:
            return False
        
        pattern_mol = Chem.MolFromSmarts(pattern)
        if pattern_mol is None:
            return False
            
        return mol.HasSubstructMatch(pattern_mol)
