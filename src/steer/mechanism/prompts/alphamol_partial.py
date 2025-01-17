prefix = """You are an expert chemist participating in a step-by-step search algorithm to evaluate proposed reaction mechanisms. Your task is to analyze partial mechanisms and determine their potential to explain a target reaction, even if they are incomplete. This evaluation will help identify promising directions for further exploration.

Here are the key pieces of information you need to consider:

1. The target reaction:
<target_reaction>"""

intermed = """</target_reaction>

2. The proposed partial mechanism to evaluate:
<proposed_mechanism>
"""

suffix = """</proposed_mechanism>

3. Finally, a potential next step to continue the mechanism:
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
   f. Alignment with the expert's description
3. Wrap your detailed analysis in <mechanism_evaluation> tags. Be thorough and specific in your evaluation, focusing on the mechanism's potential rather than its completeness. Include the following:
   - A step-by-step breakdown of the proposed mechanism, analyzing each step individually.
   - A list of key chemical principles and concepts relevant to the reaction.
   - An explicit comparison of the proposed mechanism to the expert description, noting similarities and differences.
   It's OK for this section to be quite long.
4. After your analysis, assign a score between 0 and 10 to the proposed mechanism:
   - 0 indicates a completely incorrect or implausible direction
   - 10 indicates a perfect alignment with the target reaction and expert description
   Note: Partial mechanisms can receive high scores if they show strong potential.
5. Provide a brief justification for your score, focusing on the mechanism's promise.

Output Format:
Please structure your response as follows:

<mechanism_evaluation>
[Your detailed analysis here, considering all aspects mentioned in the instructions]
</mechanism_evaluation>

<score_justification>
[Brief justification for your score, emphasizing the mechanism's potential]
</score_justification>

<score>
[Your score between 0 and 10]
</score>

Example output structure (do not copy the content, only the structure):

<mechanism_evaluation>
Step-by-step breakdown:
1. [Analysis of first step]
2. [Analysis of second step]
...

Key chemical principles and concepts:
1. [Principle 1]
2. [Principle 2]
...

Comparison to expert description:
- Similarities: [List of similarities]
- Differences: [List of differences]

Overall analysis:
The proposed mechanism shows promise in explaining the target reaction of [brief description]. It aligns with the expert's description by [key point]. The first step involves [chemical principle], which is a reasonable approach. The intermediate [chemical name] formed in step 2 is plausible and could lead to the desired product. However, the mechanism is currently missing [important step] which would be necessary to complete the reaction. Despite this, the proposed steps are chemically sound and show potential for further development.
</mechanism_evaluation>

<score_justification>
While incomplete, this partial mechanism demonstrates a strong foundation for explaining the target reaction. It aligns well with expert insights and uses established chemical principles. The proposed steps are promising and could likely be extended to fully account for the reaction.
</score_justification>

<score>
7
</score>

Remember to base your evaluation solely on the information provided in the target reaction, expert description, and proposed mechanism. Focus on the potential and promise of the partial mechanism rather than penalizing it for incompleteness.
"""
