"""Generated evaluation code for: Ester functional group cycling approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for ester functional group cycling patterns.
    
    Detects routes that cycle the same ester functional group through multiple
    formation and cleavage reactions (esterification, hydrolysis, transesterification).
    """
    
    def __init__(self, config):
        self.required_cycle_count = config.get("cycle_count", 2)
        self.same_functional_group = config.get("same_functional_group", True)
        self.ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track ester-related reactions and their positions
        ester_reactions = []
        for i, rxn in enumerate(reactions):
            if self.is_ester_reaction(rxn):
                reaction_type = self.classify_ester_reaction(rxn)
                ester_reactions.append((i, reaction_type, rxn))
        
        # Check for cycling pattern
        cycle_count = self.count_ester_cycles(ester_reactions)
        condition_met = cycle_count >= self.required_cycle_count
        
        return condition_met, len(reactions)
    
    def is_ester_reaction(self, rxn):
        """Check if reaction involves ester functional group formation/cleavage"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
            
        reactant_esters = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in reactants)
        product_esters = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in products)
        
        # Reaction involves ester if ester count changes
        return reactant_esters != product_esters
    
    def classify_ester_reaction(self, rxn):
        """Classify the type of ester reaction"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        reactant_esters = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in reactants)
        product_esters = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in products)
        
        # Check for alcohol/carboxylic acid patterns
        alcohol_pattern = Chem.MolFromSmarts("[OH1][C]")
        carboxylic_pattern = Chem.MolFromSmarts("[C](=[O])[OH1]")
        
        reactant_alcohols = sum(len(mol.GetSubstructMatches(alcohol_pattern)) for mol in reactants)
        reactant_carboxylic = sum(len(mol.GetSubstructMatches(carboxylic_pattern)) for mol in reactants)
        product_alcohols = sum(len(mol.GetSubstructMatches(alcohol_pattern)) for mol in products)
        product_carboxylic = sum(len(mol.GetSubstructMatches(carboxylic_pattern)) for mol in products)
        
        if product_esters > reactant_esters:
            # Ester formation
            if reactant_alcohols > 0 and reactant_carboxylic > 0:
                return "esterification"
            elif reactant_esters > 0:
                return "transesterification"
        elif reactant_esters > product_esters:
            # Ester cleavage
            if product_alcohols > 0 and product_carboxylic > 0:
                return "hydrolysis"
            elif product_esters > 0:
                return "transesterification"
                
        return "unknown"
    
    def count_ester_cycles(self, ester_reactions):
        """Count the number of complete ester cycles in the reaction sequence"""
        if len(ester_reactions) < 2:
            return 0
            
        cycles = 0
        formation_types = ["esterification", "transesterification"]
        cleavage_types = ["hydrolysis", "transesterification"]
        
        # Look for alternating formation/cleavage patterns
        i = 0
        while i < len(ester_reactions) - 1:
            current_type = ester_reactions[i][1]
            
            # Look for formation followed by cleavage (or vice versa)
            for j in range(i + 1, len(ester_reactions)):
                next_type = ester_reactions[j][1]
                
                formation_to_cleavage = (current_type in formation_types and 
                                       next_type in cleavage_types)
                cleavage_to_formation = (current_type in cleavage_types and 
                                       next_type in formation_types)
                
                if formation_to_cleavage or cleavage_to_formation:
                    if self.same_functional_group:
                        # Check if same ester group is involved
                        if self.involves_same_ester_group(ester_reactions[i][2], 
                                                        ester_reactions[j][2]):
                            cycles += 1
                    else:
                        cycles += 1
                    break
            i += 1
            
        return cycles
    
    def involves_same_ester_group(self, rxn1, rxn2):
        """Check if two reactions involve the same ester functional group"""
        # Simplified check - could be enhanced with atom mapping analysis
        # For now, check if similar ester patterns are present
        
        def get_ester_environments(rxn):
            """Get the local environment around ester groups"""
            environments = set()
            rxn_parts = rxn.split(">>")
            
            for part in rxn_parts:
                mols = [Chem.MolFromSmiles(m.strip()) for m in part.split(".") if m.strip()]
                for mol in mols:
                    if mol:
                        matches = mol.GetSubstructMatches(self.ester_pattern)
                        for match in matches:
                            # Get atoms around the ester group
                            env_atoms = set()
                            for atom_idx in match:
                                atom = mol.GetAtomWithIdx(atom_idx)
                                for neighbor in atom.GetNeighbors():
                                    if neighbor.GetIdx() not in match:
                                        env_atoms.add(neighbor.GetSymbol())
                            environments.add(frozenset(env_atoms))
            
            return environments
        
        env1 = get_ester_environments(rxn1)
        env2 = get_ester_environments(rxn2)
        
        # Check for overlap in ester environments
        return len(env1.intersection(env2)) > 0
