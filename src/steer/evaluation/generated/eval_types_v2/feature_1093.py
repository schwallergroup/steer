"""Generated evaluation code for: Convergent synthesis via two distinct fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(MultiRxnCondBase):
    """
    Evaluates convergent synthesis via two distinct fragments.
    Checks for the presence of specified fragment types and coupling reactions.
    """
    
    def __init__(self, config):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "negishi_coupling")
        self.fragment_types = config.get("fragment_types", ["organozinc_chain", "bromo_indole_oxadiazole"])
        
        # Define SMARTS patterns for fragment detection
        self.fragment_patterns = {
            "organozinc_chain": "[Zn]C",  # Organozinc compound
            "bromo_indole_oxadiazole": "[Br]c1ccc2[nH]c(cc2c1)c1nnc(*)o1",  # Bromo-indole with oxadiazole
            "indole_core": "c1ccc2[nH]ccc2c1",  # General indole pattern
            "oxadiazole": "c1nnco1"  # Oxadiazole ring
        }
        
        # Define coupling reaction patterns
        self.coupling_patterns = {
            "negishi_coupling": "[C:1][Zn].[Br:2][c]>>[C:1][c:2]",  # Negishi coupling pattern
            "suzuki_coupling": "[C:1][B].[Br:2][c]>>[C:1][c:2]",   # Alternative coupling
            "general_coupling": "[C,c:1].[Br,I:2][c,C]>>[C,c:1][c,C:2]"  # General C-C coupling
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for coupling reaction
        has_coupling = any(self.detect_coupling_reaction(r) for r in reactions)
        
        # Check for required fragments
        fragment_matches = self.count_fragment_types(reactions)
        has_required_fragments = len(fragment_matches) >= self.fragment_count
        
        # Check for convergent assembly (fragments come together in single step)
        has_convergent_step = any(self.detect_convergent_assembly(r) for r in reactions)
        
        condition = has_coupling and has_required_fragments and has_convergent_step
        return condition, len(reactions)

    def detect_coupling_reaction(self, rxn):
        """Detect if reaction is the specified coupling type"""
        rxn_smiles = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        reactants = parts[0]
        products = parts[1]
        
        # Check for Negishi coupling pattern (organozinc + bromide)
        if self.coupling_reaction == "negishi_coupling":
            has_organozinc = "[Zn]" in reactants
            has_bromide = "Br" in reactants
            return has_organozinc and has_bromide
            
        return False

    def count_fragment_types(self, reactions):
        """Count how many different fragment types are present"""
        found_fragments = set()
        
        for rxn in reactions:
            rxn_smiles = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                continue
                
            reactants = rxn_smiles.split(">>")[0].split(".")
            
            for reactant_smiles in reactants:
                try:
                    mol = Chem.MolFromSmiles(reactant_smiles)
                    if mol is None:
                        continue
                        
                    # Check each required fragment type
                    for fragment_type in self.fragment_types:
                        if fragment_type in self.fragment_patterns:
                            pattern = Chem.MolFromSmarts(self.fragment_patterns[fragment_type])
                            if pattern and mol.HasSubstructMatch(pattern):
                                found_fragments.add(fragment_type)
                        elif fragment_type == "bromo_indole_oxadiazole":
                            # Special case: check for combined pattern
                            if self.has_bromo_indole_oxadiazole(mol):
                                found_fragments.add(fragment_type)
                                
                except:
                    continue
                    
        return found_fragments

    def has_bromo_indole_oxadiazole(self, mol):
        """Check for bromo-indole-oxadiazole fragment"""
        if mol is None:
            return False
            
        # Check for bromine
        has_br = any(atom.GetSymbol() == "Br" for atom in mol.GetAtoms())
        
        # Check for indole substructure
        indole_pattern = Chem.MolFromSmarts(self.fragment_patterns["indole_core"])
        has_indole = indole_pattern and mol.HasSubstructMatch(indole_pattern)
        
        # Check for oxadiazole substructure  
        oxadiazole_pattern = Chem.MolFromSmarts(self.fragment_patterns["oxadiazole"])
        has_oxadiazole = oxadiazole_pattern and mol.HasSubstructMatch(oxadiazole_pattern)
        
        return has_br and has_indole and has_oxadiazole

    def detect_convergent_assembly(self, rxn):
        """Detect if reaction represents convergent assembly of fragments"""
        rxn_smiles = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        reactants = parts[0].split(".")
        
        # Convergent step should have multiple reactants (fragments coming together)
        if len(reactants) < 2:
            return False
            
        # Check that reactants contain different fragment types
        fragment_types_in_reaction = set()
        
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                for fragment_type in self.fragment_types:
                    if fragment_type == "organozinc_chain":
                        zn_pattern = Chem.MolFromSmarts("[Zn]")
                        if zn_pattern and mol.HasSubstructMatch(zn_pattern):
                            fragment_types_in_reaction.add(fragment_type)
                    elif fragment_type == "bromo_indole_oxadiazole":
                        if self.has_bromo_indole_oxadiazole(mol):
                            fragment_types_in_reaction.add(fragment_type)
                            
            except:
                continue
                
        return len(fragment_types_in_reaction) >= 2
