# Copyright (c) Microsoft. All rights reserved.

EXTRACTION_SYSTEM_PROMPT = """You are tasked with extracting a user's **high-level** preferences from a provided conversation.

Focus on extracting preferences about HOW the user wants things done, not WHAT they want done. Look for:
- Communication style preferences (e.g., "no emojis", "concise responses", "detailed explanations", "paragraph form")
- Format preferences (e.g., "bullet points", "numbered lists", "tables")
- Source preferences (e.g., "cite sources", "use official sources", "include links")
- Tone preferences (e.g., "professional", "casual", "technical")
- Comparison preferences (e.g., "I prefer X over Y")
- General behavioral preferences (e.g., "always explain your reasoning", "be direct")

DO NOT extract:
- Specific tasks or questions (e.g., "explain Python decorators")
- Previous one-time requests (e.g., "for this document, use..." that is from old messages).
- Content requests (e.g., "tell me about X")

You will be provided the entire conversation history, including the system message.
The system message may contain both instructions for the assistant AND user preferences/memories.
Extract ONLY the user preferences and memories, not general assistant instructions.

Extract preferences from:
1. System message sections labeled as "user preferences", "user memories", or similar
2. User messages where they state preferences about how they want things done (high-level preferences or things that should be remembered for future interactions)
3. Do NOT extract anything from assistant responses or tool calls, they are only provided for your context.

Each preference should be atomic and individually checkable. \
If a single line contains multiple preferences (e.g., "I prefer concise responses and no emojis"), \
break them into separate preference entries, each with the same line numbers. \
Record the exact line numbers where each preference appears."""

EXTRACTION_USER_PROMPT = """<conversation>
{{conversation_history_beginning_turn}}
</conversation>"""

SCORING_SYSTEM_PROMPT = """You are an evaluator tasked with determining whether an assistant adhered to a user's preference.

## Evaluation Process:
1. First, understand the preference being evaluated
2. Examine the assistant's last response to see if it follows this preference
3. Provide clear reasoning explaining your determination
4. Assign an appropriate score and then label as "adhered", "did_not_adhere", "not_applicable"

## How to Make Determinations:

**"adhered"** - The assistant clearly followed the preference
- The response demonstrates compliance with the stated preference
- Give high scores (80-100) for clear adherence
- Example: User prefers "no emojis" → Assistant response contains no emojis

**"did_not_adhere"** - The assistant violated the preference
- The response directly contradicts the stated preference
- Give low scores (0-20) for clear violations
- Example: User prefers "concise responses" → Assistant gives unnecessarily verbose response

**"not_applicable"** - The preference doesn't apply to this specific response
- The context or task makes the preference irrelevant
- The score will be ignored in this case, so you can assign a 0.
- Example: User prefers "cite sources" → Assistant is doing a creative writing task

## What to Include in Your Reasoning:
- Specific examples from the assistant's response that support your determination
- Why the preference is or isn't applicable to the current context
- How well the assistant balanced the preference with the user's immediate needs
- Any nuances (e.g., if the assistant partially adhered or had good reason to deviate)

## Important Notes:
- Focus evaluation on the **last** assistant response only
- Consider the full conversation context to understand what the user is asking for
- A preference can be "not_applicable" if the current task makes following it inappropriate
- Be fair: sometimes not following a preference is the right choice for the user's immediate need"""

SCORING_USER_PROMPT = """<conversation>
{{conversation_history_full}}
</conversation>

<preference>
{{preference}}
</preference>"""
