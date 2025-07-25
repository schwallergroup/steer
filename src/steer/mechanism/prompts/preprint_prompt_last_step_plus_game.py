"""
Prompt used to evaluate the last step of a proposed mechanism (without natural language guidance) in the mechanism game.
"""

prefix = """You are an expert chemist participating in a step-by-step search algorithm to evaluate proposed reaction mechanisms. Your task is to analyze partial mechanisms and determine their potential to explain a target reaction, even if they are incomplete. This evaluation will help identify promising directions for further exploration.

Also, keep in mind that you are playing a game. This is a mechanism game that forces all chemical processes to be broken down into their most fundamental steps. Even processes that occur in a concerted fashion in reality must be shown as separate elementary steps. The allowed moves are only of two types:
1. Ionization of a bond (decrease the bond order by one, with one end receiving a negative charge and the other a positive charge).
2. Attack (Any atom with at least one lone pair can attack any atom with at least one empty orbital).

Because of this stepwise requirement, you will see intermediates that might look unstable or unusual (like free protons or high-energy species). These are acceptable and necessary within the game's framework, even if they wouldn't exist as discrete species in real chemistry, but only if they are part of a concerted mechanism in which they are not isolated.

Important Note on Reaction Progress:
- Any species formed in earlier steps (including byproducts and intermediates) are considered available reactants for subsequent steps
- The formation of byproducts (like water from deprotonation) should be tracked but not penalized if they don't immediately contribute to the target product
- Steps that generate key reactive species (like bases, nucleophiles, or electrophiles) should be evaluated based on their role in setting up future transformations
- The game allows for the temporary "storage" of reactive species that will be used in later steps

Example of ionization moves:
[C]=[O] -> [C+]-[O-]
[C]-[O][H] -> [C+].[O-][H]

Example of an attack move:
[O-][H].[H+] -> [H][O][H]

Here are the key pieces of information you need to consider:

1. The target reaction:
<target_reaction>"""


intermed = """</target_reaction>

2. The proposed partial mechanism until now:
<proposed_mechanism>"""

suffix = """</proposed_mechanism>

3. Finally, a potential next step to evaluate to continue the mechanism:
<potential_next_step>
{step}
</potential_next_step>

Instructions:
1. Carefully read the target reaction, proposed mechanism, and proposed next step.
2. Analyze the proposed mechanism based on your chemistry knowledge, considering the following aspects:
   a. Alignment with established chemical principles
   b. Reasonableness of intermediates and transition states
   c. Potential to account for all reactants and products in the target reaction
   d. Consistency with known reaction conditions (if specified)
   e. Presence of necessary steps and absence of unnecessary ones
   f. Role of intermediates and byproducts in subsequent steps
3. Wrap your detailed analysis in <mechanism_evaluation> tags. Be thorough and specific in your evaluation, focusing on the mechanism's potential rather than its completeness. Include the following:
   - A step-by-step breakdown of the proposed mechanism, analyzing each step individually
   - Identification of key reactive species formed (including intermediates and byproducts)
   - A list of key chemical principles and concepts relevant to the reaction
   It's OK for this section to be quite long.
4. After your analysis, assign a score between 0 and 10 to the potential next step:
   - 0 indicates a completely incorrect or implausible direction
   - 5 indicates a step that maintains necessary byproducts/intermediates but doesn't advance toward the product
   - 10 indicates a perfect alignment with the target reaction
   Note: Next steps can receive high scores (6-10) if they show strong potential for advancing toward the target, even if they don't reach the goal.
5. Provide a brief justification for your score, focusing on both:
   a. The step's immediate contribution to the target reaction
   b. Its role in generating or maintaining important reactive species
6. Remember you are playing a step-wise mechanism game with simplified rules, so some intermediates may look unusual but might be correct within the game's constraints.

Output Format:
Please structure your response as follows:

<mechanism_evaluation>
[Your detailed analysis here, considering all aspects mentioned in the instructions]

Key Reactive Species Formed/Maintained:
- [Species 1]: [Role in mechanism]
- [Species 2]: [Role in mechanism]
...

Key chemical principles and concepts:
1. [Principle 1]
2. [Principle 2]
...
</mechanism_evaluation>

<score_justification>
[Brief justification addressing both immediate progress and role in maintaining reactive species]
</score_justification>

<score>
[Your score between 0 and 10]
</score>

Remember to base your evaluation solely on the information provided in the target reaction and proposed mechanism. Consider both immediate progress toward the target and the importance of generating/maintaining reactive species for future steps."""
