
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

PROMPT_TEMPLATE_ALPACA_DOLLY = f"{B_SYS}Below is an instruction that describes a task. Write a response that appropriately completes the request.{E_SYS}### Instruction:\n[INPUT]\n\n### Response:\n"

LONG_PROMPT_COMPLETION = """
Let’s roleplay.

I will describe a person who have values between 1 and 6 for each value.
1 means the value is not important to him at all and 6 means the value is very much important to him.

Value Definition:
- Achievement: values personal success through demonstrating competence according to social standards
- Benevolence: values preserving and enhancing the welfare of those with whom one is in frequent personal contact (the ‘in-group’)
- Conformity: values restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms
- Hedonism: values pleasure or sensuous gratification for oneself
- Power: values social status and prestige, control or dominance over people and resources
- Security: values safety, harmony, and stability of society, of relationships, and of self
- Self-direction: values independent thought and action–choosing, creating, exploring.
- Stimulation: values excitement, novelty, and challenge in life
- Tradition: values respect, commitment, and acceptance of the customs and ideas that one’s culture or religion provides
- Universalism: values understanding, appreciation, tolerance, and protection for the welfare of all people and for nature

Value Score:
- Achievement: {achievement_score}
- Benevolence: {benevolence_score}
- Conformity: {conformity_score}
- Hedonism: {hedonism_score}
- Power: {power_score}
- Security: {security_score}
- Self-Direction: {self_direction_score}
- Stimulation: {stimulation_score}
- Tradition: {tradition_score}
- Universalism: {universalism_score}

As this person, I would say:
{input_text}
""".strip()


LONG_PROMPT_INSTRUCTION = """Let’s roleplay.

I will describe a person who have values between 1 and 6 for each value.
1 means the value is not important to him at all and 6 means the value is very much important to him.

Value Definition:
- Achievement: values personal success through demonstrating competence according to social standards
- Benevolence: values preserving and enhancing the welfare of those with whom one is in frequent personal contact (the ‘in-group’)
- Conformity: values restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms
- Hedonism: values pleasure or sensuous gratification for oneself
- Power: values social status and prestige, control or dominance over people and resources
- Security: values safety, harmony, and stability of society, of relationships, and of self
- Self-direction: values independent thought and action–choosing, creating, exploring.
- Stimulation: values excitement, novelty, and challenge in life
- Tradition: values respect, commitment, and acceptance of the customs and ideas that one’s culture or religion provides
- Universalism: values understanding, appreciation, tolerance, and protection for the welfare of all people and for nature

Value Score:
- Achievement: {achievement_score}
- Benevolence: {benevolence_score}
- Conformity: {conformity_score}
- Hedonism: {hedonism_score}
- Power: {power_score}
- Security: {security_score}
- Self-Direction: {self_direction_score}
- Stimulation: {stimulation_score}
- Tradition: {tradition_score}
- Universalism: {universalism_score}

Based on the value scores above, become this person and respond accordingly to the prompt below.
QUESTION: {input_text}
ANSWER: """



VANILLA_PROMPT_COMPLETION = """{input_text}"""
VANILLA_PROMPT_INSTRUCTION = """QUESTION: {input_text}
ANSWER: """
