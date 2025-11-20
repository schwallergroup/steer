"""Generated evaluation code for: Multiple protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on protecting group cycling strategy.
    Checks for repeated protection/deprotection cycles using specified protecting groups
    on the same or different functional groups.
    """
    
    def __init__(self, config):
        self.protection_types = config["protection_types"]
        self.cycle_count = config["cycle_count"]
        self.same_functional_group = config["same_functional_group"]
        
        # Define SMARTS patterns for protecting group reactions
        self.protection_patterns = {
            "tert-butyl_ester": {
                "protection": "[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[O:3]C(C)(C)C",
                "deprotection": "[C:1](=[O:2])[O:3]C(C)(C)C>>[C:1](=[O:2])[OH:3]",
                "protected_pattern": "[C](=O)OC(C)(C)C"
            },
            "ethyl_ester": {
                "protection": "[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[O:3]CC",
                "deprotection": "[C:1](=[O:2])[O:3]CC>>[C:1](=[O:2])[OH:3]",
                "protected_pattern": "[C](=O)OCC"
            }
        }
    
    def condition_depth(self, d):
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events for each protecting group type
        protection_events = {pg_type: [] for pg_type in self.protection_types}
        
        for i, rxn in enumerate(reactions):
            for pg_type in self.protection_types:
                if self.is_protection_reaction(rxn, pg_type):
                    protection_events[pg_type].append(('protect', i, self.get_reaction_sites(rxn, pg_type)))
                elif self.is_deprotection_reaction(rxn, pg_type):
                    protection_events[pg_type].append(('deprotect', i, self.get_reaction_sites(rxn, pg_type)))
        
        # Count complete protection/deprotection cycles
        total_cycles = 0
        for pg_type in self.protection_types:
            cycles = self.count_cycles(protection_events[pg_type])
            total_cycles += cycles
        
        # Check if we have cross-protection (using different protecting groups on same sites)
        if not self.same_functional_group:
            cross_cycles = self.count_cross_protection_cycles(protection_events)
            total_cycles += cross_cycles
        
        condition_met = total_cycles >= self.cycle_count
        return condition_met, len(reactions)
    
    def is_protection_reaction(self, rxn, pg_type):
        """Check if reaction is a protection reaction for given protecting group type."""
        rxn_smiles = rxn.split(">>")
        if len(rxn_smiles) != 2:
            return False
        
        reactants = Chem.MolFromSmiles(rxn_smiles[0])
        products = Chem.MolFromSmiles(rxn_smiles[1])
        
        if not reactants or not products:
            return False
        
        # Check if free carboxylic acid is consumed and protected ester is formed
        free_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
        protected_pattern = Chem.MolFromSmarts(self.protection_patterns[pg_type]["protected_pattern"])
        
        has_free_acid_reactant = reactants.HasSubstructMatch(free_acid_pattern)
        has_protected_product = products.HasSubstructMatch(protected_pattern)
        
        return has_free_acid_reactant and has_protected_product
    
    def is_deprotection_reaction(self, rxn, pg_type):
        """Check if reaction is a deprotection reaction for given protecting group type."""
        rxn_smiles = rxn.split(">>")
        if len(rxn_smiles) != 2:
            return False
        
        reactants = Chem.MolFromSmiles(rxn_smiles[0])
        products = Chem.MolFromSmiles(rxn_smiles[1])
        
        if not reactants or not products:
            return False
        
        # Check if protected ester is consumed and free carboxylic acid is formed
        protected_pattern = Chem.MolFromSmarts(self.protection_patterns[pg_type]["protected_pattern"])
        free_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
        
        has_protected_reactant = reactants.HasSubstructMatch(protected_pattern)
        has_free_acid_product = products.HasSubstructMatch(free_acid_pattern)
        
        return has_protected_reactant and has_free_acid_product
    
    def get_reaction_sites(self, rxn, pg_type):
        """Get the reaction sites (atom map numbers) involved in protection/deprotection."""
        rxn_smiles = rxn.split(">>")
        if len(rxn_smiles) != 2:
            return set()
        
        mol = Chem.MolFromSmiles(rxn_smiles[0])
        if not mol:
            return set()
        
        sites = set()
        pattern = Chem.MolFromSmarts("[C](=O)[OH]") if self.is_protection_reaction(rxn, pg_type) else \
                 Chem.MolFromSmarts(self.protection_patterns[pg_type]["protected_pattern"])
        
        if mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetAtomMapNum() > 0:
                        sites.add(atom.GetAtomMapNum())
        
        return sites
    
    def count_cycles(self, events):
        """Count complete protection/deprotection cycles for a single protecting group."""
        if len(events) < 2:
            return 0
        
        cycles = 0
        protected_sites = set()
        
        for event_type, step, sites in events:
            if event_type == 'protect':
                protected_sites.update(sites)
            elif event_type == 'deprotect':
                # Check if any of the deprotected sites were previously protected
                if sites.intersection(protected_sites):
                    cycles += 1
                    protected_sites -= sites
        
        return cycles
    
    def count_cross_protection_cycles(self, all_events):
        """Count cycles involving different protecting groups on the same functional groups."""
        # Flatten all events and sort by step
        all_flat_events = []
        for pg_type, events in all_events.items():
            for event_type, step, sites in events:
                all_flat_events.append((event_type, step, sites, pg_type))
        
        all_flat_events.sort(key=lambda x: x[1])  # Sort by step
        
        cycles = 0
        site_protection_history = {}  # site -> [(pg_type, protected)]
        
        for event_type, step, sites, pg_type in all_flat_events:
            for site in sites:
                if site not in site_protection_history:
                    site_protection_history[site] = []
                
                if event_type == 'protect':
                    site_protection_history[site].append((pg_type, True))
                elif event_type == 'deprotect':
                    # Check if this site was protected with a different protecting grou
