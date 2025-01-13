prefix = """You are an expert AI chemist tasked with evaluating proposed reaction mechanisms as part of a step-by-step mechanism construction algorithm. Your goal is to analyze each elementary step or partial mechanism and determine its potential to contribute to the target reaction.

An expert chemist has proposed a mechanism and described it as follows:

"The mechanism starts with the deprotonation of the N adjacent to the carbonyl in the pyrazolidinone. The mechanism then proceeds through a nucleophylic attack of the nitrogen on the allyl bromide. In the final step, the Br- is eliminated."

First, carefully review the target reaction:

<target_reaction>
"""

intermed = """</target_reaction>

Now, examine the proposed mechanism or partial mechanism:

<proposed_mechanism>
"""

suffix = """</proposed_mechanism>

Your task is to evaluate the potential of this proposed mechanism or partial mechanism to explain the target reaction, even if it is incomplete. Consider each step's contribution to the overall process and its alignment with established chemical principles.

In your analysis, address the following aspects:
1. Chemical soundness: Does each step follow established chemical principles?
2. Intermediate plausibility: Are the proposed intermediates and transition states reasonable?
3. Alignment with target reaction: How well does the mechanism or partial mechanism align with the reactants and products in the target reaction?
4. Alignment with the description from the expert chemist: How well does the mechanism or partial mechanism align with the description from the expert chemist?
5. Consistency with known conditions: If reaction conditions are specified, is the proposed mechanism consistent with them?

Provide your detailed analysis inside <mechanism_evaluation> tags. Be thorough and specific in your evaluation, keeping in mind that you are assessing the potential of each step or partial mechanism. Include the following sections in your evaluation:

1. Step-by-step breakdown: Identify and number each elementary step in the proposed mechanism.
2. Missing steps or intermediates: List any crucial steps or intermediates that are not present in the proposed mechanism.
3. Comparison to target reaction: Explicitly compare the proposed mechanism to the target reaction, noting similarities and differences.

After your analysis, assign a potential score between 0 and 10 to the proposed mechanism or partial mechanism:
- 0 indicates a completely incorrect or implausible step/mechanism with no potential to contribute to the target reaction.
- 10 indicates a perfect step/mechanism that fully explains or has high potential to lead to the target reaction.

Provide a brief justification for your score, focusing on the potential of the proposed steps to contribute to the final mechanism.

Format your response as follows:

<mechanism_evaluation>
[Your detailed analysis here, including the step-by-step breakdown, missing steps/intermediates, comparison to target reaction, and evaluation of each aspect mentioned above]
</mechanism_evaluation>

<score_justification>
[Brief justification for your potential score, emphasizing promising aspects and areas for improvement]
</score_justification>

<score>
[Your integer score between 0 and 10]
</score>

Remember:
- Base your evaluation solely on the information provided in the target reaction and proposed mechanism.
- Focus on identifying promising steps and directions, even if the mechanism is incomplete.
- Consider how each step or partial mechanism could potentially contribute to the final, complete mechanism.
- Avoid penalizing incomplete mechanisms; instead, evaluate their potential to lead to a correct final mechanism.

Your role is crucial in guiding the mechanism construction algorithm towards promising pathways. Provide constructive feedback that can help in refining and completing the mechanism in subsequent steps."""
