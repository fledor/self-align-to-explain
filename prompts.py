"""
Prompt templates for the Counterfactual DPO Training Pipeline.

Contains templates for counterfactual generation and evaluation.
"""

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT = "You are an excellent assistant for text editing."

SYSTEM_PROMPT_EVALUATION = "You are an expert at natural language inference tasks."


# =============================================================================
# NLI Counterfactual Generation Prompts
# =============================================================================

def get_nli_premise_edit_prompt(
    premise: str,
    hypothesis: str,
    original_label: str,
    target_label: str,
) -> str:
    """
    Get prompt for editing the premise to change the NLI relationship.
    
    Args:
        premise: The premise text to edit
        hypothesis: The hypothesis text (fixed)
        original_label: Current NLI label
        target_label: Desired NLI label after editing
        
    Returns:
        The formatted prompt string
    """
    return f"""Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the premise with minimal edits to achieve the target relation from the original one and output the edited premise surrounding by <edit>[premise]</edit>. 
Do not make any unnecessary changes.

#####Begin Example####

Original relation: **entailment**
premise: A woman is talking to a man. 
hypothesis: Brown-haired woman talking to man with backpack.
Target relation: **neutral**

Step 1: Identify phrases, words in the premise leading to the entailment relation:
'man',
Step 2: Change these phrases, words to get neutral relation with minimal changes:
'man' to 'student'.
Step 3: Replace the phrases, words from step 1 in the original text by the phrases, words, sentences in step 2:

Edited premise: <edit>A woman is talking to a student.</edit>

#####End Example####

Request: Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the premise with minimal edits to achieve the target relation from the original one and output the edited premise surrounding by <edit>[premise]</edit>. Do not make any unnecessary changes. Do not add anything else.

Original relation: **{original_label}**
premise: {premise}
hypothesis: {hypothesis}
Target relation: **{target_label}**
Edited premise:"""


def get_nli_hypothesis_edit_prompt(
    premise: str,
    hypothesis: str,
    original_label: str,
    target_label: str,
) -> str:
    """
    Get prompt for editing the hypothesis to change the NLI relationship.
    
    Args:
        premise: The premise text (fixed)
        hypothesis: The hypothesis text to edit
        original_label: Current NLI label
        target_label: Desired NLI label after editing
        
    Returns:
        The formatted prompt string
    """
    return f"""Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the hypothesis with minimal edits to achieve the target relation from the original one and output the edited hypothesis surrounding by <edit>[hypothesis]</edit>. 
Do not make any unnecessary changes.

#####Begin Example####

Original relation: **entailment**
premise: A woman is talking to a man.
hypothesis: A woman is having a conversation.
Target relation: **contradiction**

Step 1: Identify phrases, words in the hypothesis leading to the entailment relation:
'having a conversation',
Step 2: Change these phrases, words to get contradiction relation with minimal changes:
'having a conversation' to 'sitting alone in silence'.
Step 3: Replace the phrases, words from step 1 in the original text by the phrases, words, sentences in step 2:

Edited hypothesis: <edit>A woman is sitting alone in silence.</edit>

#####End Example####

Request: Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the hypothesis with minimal edits to achieve the target relation from the original one and output the edited hypothesis surrounding by <edit>[hypothesis]</edit>. Do not make any unnecessary changes. Do not add anything else.

Original relation: **{original_label}**
premise: {premise}
hypothesis: {hypothesis}
Target relation: **{target_label}**
Edited hypothesis:"""


# =============================================================================
# BoolQ Counterfactual Generation Prompts
# =============================================================================

def get_boolq_passage_edit_prompt(
    passage: str,
    question: str,
    original_label: str,
    target_label: str,
) -> str:
    """
    Get prompt for editing the passage to change the yes/no answer.
    
    Args:
        passage: The passage text to edit
        question: The yes/no question (fixed)
        original_label: Current answer (true/false)
        target_label: Desired answer after editing (true/false)
        
    Returns:
        The formatted prompt string
    """
    original_answer = "Yes" if original_label == "true" else "No"
    target_answer = "Yes" if target_label == "true" else "No"
    
    return f"""Given a passage and a yes/no question, the current answer is "{original_answer}". Edit the passage with minimal changes so that the answer becomes "{target_answer}". Output the edited passage surrounded by <edit>[passage]</edit>.
Do not make any unnecessary changes.

#####Begin Example####

Question: Is the movie Rudy based on a true story?
Passage: Daniel Eugene "Rudy" Ruettiger (born August 23, 1948) is a motivational speaker who played college football at the University of Notre Dame. His early life and career at Notre Dame was the inspiration for the 1993 film Rudy.
Current answer: **Yes**
Target answer: **No**

Step 1: Identify phrases in the passage that support the current answer "Yes":
'His early life and career at Notre Dame was the inspiration for the 1993 film Rudy.'
Step 2: Change these phrases to support the target answer "No" with minimal changes:
'The 1993 film Rudy is a fictional story not based on any real person.'
Step 3: Replace the phrases from step 1 with the phrases from step 2:

Edited passage: <edit>Daniel Eugene "Rudy" Ruettiger (born August 23, 1948) is a motivational speaker who played college football at the University of Notre Dame. The 1993 film Rudy is a fictional story not based on any real person.</edit>

#####End Example####

Request: Given a passage and a yes/no question, the current answer is "{original_answer}". Edit the passage with minimal changes so that the answer becomes "{target_answer}". Output the edited passage surrounded by <edit>[passage]</edit>. Do not make any unnecessary changes. Do not add anything else.

Question: {question}
Passage: {passage}
Current answer: **{original_answer}**
Target answer: **{target_answer}**
Edited passage:"""


# =============================================================================
# Evaluation Prompts
# =============================================================================

def get_nli_verification_prompt(premise: str, hypothesis: str) -> str:
    """
    Get prompt for verifying the NLI relationship between premise and hypothesis.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        
    Returns:
        The formatted prompt string
    """
    return f"""Classify the relationship between the following premise and hypothesis.

The relationship must be one of:
- entailment: The hypothesis is definitely true given the premise
- contradiction: The hypothesis is definitely false given the premise  
- neutral: The hypothesis may or may not be true given the premise

Premise: {premise}
Hypothesis: {hypothesis}

Respond with ONLY the label (entailment, neutral, or contradiction) followed by your confidence score from 1-5 (where 5 is most confident).

Format: [label] [confidence]
Example: entailment 5"""


def get_nli_verification_prompt_detailed(premise: str, hypothesis: str) -> str:
    """
    Get detailed prompt for verifying NLI relationship with explanation.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        
    Returns:
        The formatted prompt string
    """
    return f"""Analyze the logical relationship between the following premise and hypothesis.

Premise: {premise}
Hypothesis: {hypothesis}

Step 1: Identify key information in the premise.
Step 2: Determine if the hypothesis is supported by, contradicted by, or independent of the premise.
Step 3: Classify as one of: entailment, contradiction, or neutral.
Step 4: Rate your confidence from 1 (very uncertain) to 5 (very certain).

Provide your answer in this exact format:
Label: [entailment/contradiction/neutral]
Confidence: [1-5]"""


def get_boolq_verification_prompt(passage: str, question: str) -> str:
    """
    Get prompt for verifying the yes/no answer given passage and question.
    
    Args:
        passage: The passage text
        question: The yes/no question
        
    Returns:
        The formatted prompt string
    """
    return f"""Based on the following passage, answer the yes/no question.

Passage: {passage}

Question: {question}

Respond with ONLY "Yes" or "No" followed by your confidence score from 1-5 (where 5 is most confident).

Format: [Yes/No] [confidence]
Example: Yes 5"""


SYSTEM_PROMPT_BOOLQ = "You are an expert at reading comprehension and answering questions."


# =============================================================================
# Prompt Registry for Different Dataset Types
# =============================================================================

PROMPT_REGISTRY = {
    "snli_premise": {
        "generation": get_nli_premise_edit_prompt,
        "verification": get_nli_verification_prompt,
        "system": SYSTEM_PROMPT,
        "system_eval": SYSTEM_PROMPT_EVALUATION,
    },
    "snli_hypothesis": {
        "generation": get_nli_hypothesis_edit_prompt,
        "verification": get_nli_verification_prompt,
        "system": SYSTEM_PROMPT,
        "system_eval": SYSTEM_PROMPT_EVALUATION,
    },
    "boolq": {
        "generation": get_boolq_passage_edit_prompt,
        "verification": get_boolq_verification_prompt,
        "system": SYSTEM_PROMPT,
        "system_eval": SYSTEM_PROMPT_BOOLQ,
    },
}


def get_generation_prompt(
    dataset_name: str,
    entry: dict,
    target_label: str,
) -> tuple[str, str]:
    """
    Get the generation prompt for a dataset entry.
    
    Args:
        dataset_name: Name of the dataset
        entry: Formatted entry from the dataset
        target_label: Target label for the counterfactual
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if dataset_name not in PROMPT_REGISTRY:
        raise ValueError(f"No prompts registered for dataset: {dataset_name}")
    
    prompts = PROMPT_REGISTRY[dataset_name]
    
    # Handle different dataset types
    if dataset_name.startswith("snli"):
        user_prompt = prompts["generation"](
            premise=entry["premise"],
            hypothesis=entry["hypothesis"],
            original_label=entry["label"],
            target_label=target_label,
        )
    elif dataset_name == "boolq":
        user_prompt = prompts["generation"](
            passage=entry["passage"],
            question=entry["question"],
            original_label=entry["label"],
            target_label=target_label,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")
    
    return prompts["system"], user_prompt


def get_verification_prompt(
    dataset_name: str,
    **kwargs,
) -> tuple[str, str]:
    """
    Get the verification prompt for a counterfactual.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Dataset-specific arguments (premise/hypothesis for NLI, passage/question for BoolQ)
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if dataset_name not in PROMPT_REGISTRY:
        raise ValueError(f"No prompts registered for dataset: {dataset_name}")
    
    prompts = PROMPT_REGISTRY[dataset_name]
    
    # Handle different dataset types
    if dataset_name.startswith("snli"):
        user_prompt = prompts["verification"](
            premise=kwargs["premise"],
            hypothesis=kwargs["hypothesis"],
        )
    elif dataset_name == "boolq":
        user_prompt = prompts["verification"](
            passage=kwargs["passage"],
            question=kwargs["question"],
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")
    
    return prompts["system_eval"], user_prompt


def format_chat_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """
    Format prompts into chat message format.
    
    Args:
        system_prompt: The system prompt
        user_prompt: The user prompt
        
    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

