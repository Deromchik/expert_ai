#!/usr/bin/env python3
"""
Streamlit Quiz App
Combines prompt_review_runner_2.py (question generation) and
quiz_answer_validation.py (answer validation) into an interactive quiz UI.
All API calls are logged and available for download.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OPENROUTER_MAX_TOKENS: int = 10000

LANGUAGE_MAP: dict[str, str] = {
    "en": "English", "ru": "Russian", "pl": "Polish", "uk": "Ukrainian",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian",
    "sr": "Serbian", "sl": "Slovenian", "et": "Estonian", "lv": "Latvian",
    "lt": "Lithuanian", "fi": "Finnish", "sv": "Swedish", "no": "Norwegian",
    "da": "Danish", "is": "Icelandic", "ga": "Irish", "mt": "Maltese",
    "el": "Greek", "tr": "Turkish", "ar": "Arabic", "he": "Hebrew",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
    "tl": "Filipino", "bn": "Bengali", "fa": "Persian",
}


def get_language_name(code: str) -> str:
    return LANGUAGE_MAP.get(str(code).lower(), str(code).upper())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# DIFFICULTY LEVELS
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS: dict[int, dict] = {
    1: {
        "label": "1 — Beginner",
        "validation_threshold": 0.50,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 1 (BEGINNER) ===\n"
            "Generate VERY EASY questions suitable for absolute beginners.\n"
            "- Questions should test basic terminology recall and simple definitions.\n"
            "- Answers should be single words or very short phrases (1-2 words).\n"
            "- Avoid any questions requiring understanding of relationships between concepts.\n"
            "- Focus on the most fundamental, surface-level facts from the materials.\n"
            "- Each question should have a clear, unambiguous single correct answer.\n"
            "- Provide 3-5 acceptable answer variants (synonyms, abbreviations, alternate forms).\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 1 (BEGINNER) ===\n"
            "Generate VERY EASY reasoning questions suitable for absolute beginners.\n"
            "- Scenarios should be simple and straightforward with an obvious course of action.\n"
            "- Only one clear consideration or step is needed to solve the scenario.\n"
            "- Use everyday, relatable situations that directly map to a single concept from the lesson.\n"
            "- The expected reasoning should be short and simple (2-3 sentences).\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 1 — Beginner):\n"
            "The quiz difficulty is set to BEGINNER level. Be VERY lenient when scoring.\n"
            "- Accept answers that are approximately correct or in the right direction.\n"
            "- If the student demonstrates they understand the general area/topic, give generous credit.\n"
            "- Synonyms, related terms, and imprecise but directionally correct answers should score highly.\n"
            "- A validation_score >= 0.50 should be considered passing for this difficulty level.\n"
            "- Only give very low scores if the answer is completely unrelated or fundamentally wrong.\n"
        ),
    },
    2: {
        "label": "2 — Elementary",
        "validation_threshold": 0.60,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 2 (ELEMENTARY) ===\n"
            "Generate EASY questions suitable for learners with basic knowledge.\n"
            "- Questions should test simple understanding and basic recall.\n"
            "- Answers should be short (1-3 words).\n"
            "- Questions may test simple relationships between two concepts.\n"
            "- Stick to the most important, clearly stated facts from the materials.\n"
            "- Provide 2-4 acceptable answer variants.\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 2 (ELEMENTARY) ===\n"
            "Generate EASY reasoning questions for learners with basic knowledge.\n"
            "- Scenarios should be straightforward with a clear recommended action.\n"
            "- Require consideration of 1-2 factors from the lesson.\n"
            "- Situations should be practical and easy to relate to.\n"
            "- The expected reasoning should involve 2-4 simple steps.\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 2 — Elementary):\n"
            "The quiz difficulty is set to ELEMENTARY level. Be lenient when scoring.\n"
            "- Accept answers that capture the main idea even if details are missing.\n"
            "- Related terms and partially correct answers should receive partial credit.\n"
            "- A validation_score >= 0.60 should be considered passing for this difficulty level.\n"
            "- Penalize only clearly incorrect or unrelated answers.\n"
        ),
    },
    3: {
        "label": "3 — Intermediate",
        "validation_threshold": 0.70,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 3 (INTERMEDIATE) ===\n"
            "Generate MODERATE difficulty questions for learners with solid foundational knowledge.\n"
            "- Questions should test understanding of concepts and their relationships.\n"
            "- Answers should be concise (1-3 words) but require understanding, not just memorization.\n"
            "- Include questions about how concepts connect or differ from each other.\n"
            "- Provide 2-4 acceptable answer variants.\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 3 (INTERMEDIATE) ===\n"
            "Generate MODERATE reasoning questions for learners with solid knowledge.\n"
            "- Scenarios should involve multiple considerations and trade-offs.\n"
            "- Require applying knowledge from the lesson to realistic situations.\n"
            "- The expected reasoning should involve 3-5 steps with clear logic.\n"
            "- Include scenarios where more than one approach is possible but one is clearly better.\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 3 — Intermediate):\n"
            "The quiz difficulty is set to INTERMEDIATE level. Apply standard scoring.\n"
            "- Answers should demonstrate understanding of the concept, not just recall.\n"
            "- Accept semantically equivalent answers with minor imprecisions.\n"
            "- A validation_score >= 0.70 should be considered passing for this difficulty level.\n"
            "- Partial credit for answers that show understanding but miss important nuances.\n"
        ),
    },
    4: {
        "label": "4 — Advanced",
        "validation_threshold": 0.80,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 4 (ADVANCED) ===\n"
            "Generate CHALLENGING questions for learners with strong knowledge.\n"
            "- Questions should test deeper understanding, analysis, and application.\n"
            "- Include questions about edge cases, exceptions, or subtle distinctions.\n"
            "- Require precise answers that demonstrate thorough understanding.\n"
            "- Provide 2-3 acceptable answer variants (less tolerance for imprecision).\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 4 (ADVANCED) ===\n"
            "Generate CHALLENGING reasoning questions for knowledgeable learners.\n"
            "- Scenarios should involve complex trade-offs and multiple interacting factors.\n"
            "- Require synthesis of several concepts from the lesson.\n"
            "- The expected reasoning should be thorough and well-structured.\n"
            "- Include scenarios with non-obvious solutions that require critical analysis.\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 4 — Advanced):\n"
            "The quiz difficulty is set to ADVANCED level. Apply strict scoring.\n"
            "- Answers must be precise and demonstrate clear understanding.\n"
            "- Semantically equivalent answers are accepted, but vague or overly general answers should score lower.\n"
            "- A validation_score >= 0.80 should be considered passing for this difficulty level.\n"
            "- Require correct key terms and accurate relationships between concepts.\n"
        ),
    },
    5: {
        "label": "5 — Expert",
        "validation_threshold": 0.88,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 5 (EXPERT) ===\n"
            "Generate VERY DIFFICULT questions for expert-level learners.\n"
            "- Questions should test synthesis, critical evaluation, and nuanced understanding.\n"
            "- Include questions about implications, limitations, and advanced applications.\n"
            "- Require highly precise, technically accurate answers.\n"
            "- Provide only 1-2 acceptable answer variants (very strict matching).\n"
            "- Questions may combine concepts from current and previous lessons.\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 5 (EXPERT) ===\n"
            "Generate VERY DIFFICULT reasoning questions for expert learners.\n"
            "- Scenarios should be highly complex with multiple competing priorities.\n"
            "- Require deep synthesis of concepts from across all provided materials.\n"
            "- The expected reasoning should demonstrate expert-level analysis and judgment.\n"
            "- Include scenarios with ambiguity where the quality of reasoning matters most.\n"
            "- Expect thorough consideration of consequences and edge cases.\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 5 — Expert):\n"
            "The quiz difficulty is set to EXPERT level. Apply very strict scoring.\n"
            "- Answers must be precise, technically correct, and comprehensive.\n"
            "- Only accept answers that demonstrate deep understanding.\n"
            "- A validation_score >= 0.88 should be considered passing for this difficulty level.\n"
            "- Vague, incomplete, or imprecise answers should score significantly lower.\n"
            "- Require correct use of terminology and accurate conceptual relationships.\n"
        ),
    },
}


def get_difficulty_config(level: int) -> dict:
    return DIFFICULTY_LEVELS.get(level, DIFFICULTY_LEVELS[3])


# ---------------------------------------------------------------------------
# PROMPT TEMPLATES (from prompt_review_runner_2.py)
# ---------------------------------------------------------------------------

PROMPT_SYSTEM_TEMPLATE: str = (
    "You are an expert assessment designer and pedagogy specialist. "
    "Your output will be used by a downstream LLM to grade learners across multiple scoring dimensions. "
    "Always write in clear, concise, natural {language_name} (language code: {course_language}). "
    "Favor measurable objectives, discriminative criteria, and observable evidence. Avoid fluff. "
)

PROMPT_GIFTQUIZ_QUESTIONS_TEMPLATE: str = (
    "You are an assessment designer. Generate SHORT ANSWER questions in {language_name} (language code: {course_language}) in strict JSON.\n"
    "Prioritize the CURRENT LESSON materials, use previous lesson summaries only if helpful.\n"
    "IMPORTANT: All questions and answers must be written in {language_name}.\n\n"
    "{difficulty_block}"
    "Materials (may be partial):\n"
    "Gift (raw, may include questions):\n{gift_text}\n\n"
    "{extracted_block}"
    "{other_block}"
    "{prev_block}"
    "=== QUESTION GENERATION GUIDELINES ===\n\n"
    "1. DIVERSITY (CRITICAL): Each question MUST test a DIFFERENT concept or term.\n"
    "   - Do NOT create multiple questions with the same answer.\n"
    "   - Cover the full range of key concepts from the materials.\n"
    "   - If existing questions cover a concept, create NEW questions about OTHER concepts.\n\n"
    "2. ANSWERS: Provide 2-4 acceptable answer variants (synonyms, abbreviations, alternate phrasings).\n"
    "   - Include both singular and plural forms if applicable.\n"
    "   - Include common abbreviations (e.g., 'AI', 'Artificial Intelligence').\n"
    "   - Answers should be SHORT (1-3 words typically).\n\n"
    "3. SCORE WEIGHTING (1-5 points per question):\n"
    "   - 1 point: Basic terminology recall (simple definitions)\n"
    "   - 2 points: Understanding relationships between concepts\n"
    "   - 3 points: Applying knowledge to identify correct terms in context\n"
    "   - 4-5 points: Complex concepts requiring deeper understanding\n"
    "   - More foundational/important concepts should have HIGHER scores.\n\n"
    "4. QUIZ PARAMETERS:\n"
    "   - max_attempts: 2-3 for easy quizzes (3-4 questions), 3-4 for harder quizzes (5+ questions)\n"
    "   - max_execution_time: Calculate as (number_of_questions × 45 seconds). Example: 4 questions = 180 seconds.\n"
    "   - min_pass_score: Set to 50-60%% of total score (allows some mistakes but ensures basic understanding).\n\n"
    "Output strictly as minified JSON with keys: questions (3-5), min_pass_score, max_attempts, max_execution_time.\n"
    "CRITICAL: You have a maximum of 10000 tokens. Ensure your JSON response is COMPLETE and VALID within this limit.\n"
    "If approaching the limit, generate fewer questions but ensure the JSON is properly closed with all required fields.\n"
    "Schema:\n"
    "{{\n"
    "  \"questions\": [ {{ \"question\": string, \"answers\": [string, ...], \"score\": integer }}, ... ],\n"
    "  \"min_pass_score\": number,\n"
    "  \"max_attempts\": integer,\n"
    "  \"max_execution_time\": integer\n"
    "}}\n"
    "Rules: questions must be SHORT ANSWERS in {language_name}; answers are acceptable synonyms in {language_name}; "
    "scores range 1-5 based on importance; min_pass_score should be 50-60%% of sum(scores); max_attempts 2-4."
)

PROMPT_GIFTQUIZ_REASONING_QUESTIONS_TEMPLATE: str = (
    "You are an assessment designer. Generate REASONING questions in {language_name} (language code: {course_language}) in strict JSON.\n"
    "REASONING questions present real-world scenarios or dilemmas that require thoughtful analysis and multi-step reasoning.\n"
    "Each question should describe a situation or challenge, and the answer should explain the reasoning process and recommended actions.\n"
    "Prioritize the CURRENT LESSON materials, use previous lesson summaries only if helpful.\n"
    "IMPORTANT: All questions and answers must be written in {language_name}.\n\n"
    "{difficulty_block}"
    "Materials (may be partial):\n"
    "Gift (raw, may include questions):\n{gift_text}\n\n"
    "{extracted_block}"
    "{other_block}"
    "{prev_block}"
    "=== REASONING QUESTION GUIDELINES ===\n\n"
    "1. SCENARIO DIVERSITY: Each question MUST present a UNIQUE scenario testing different aspects.\n"
    "   - Cover various real-world applications of the lesson concepts.\n"
    "   - Scenarios should be practical and relatable to learners.\n"
    "   - Avoid asking the same type of reasoning multiple times.\n\n"
    "2. ANSWER QUALITY: The answer should be a comprehensive reasoning explanation.\n"
    "   - Include step-by-step thought process.\n"
    "   - Explain WHY the recommended actions are appropriate.\n"
    "   - Reference relevant concepts from the lesson.\n\n"
    "3. SCORE WEIGHTING (2-5 points per question - reasoning is harder):\n"
    "   - 2 points: Simple scenario with straightforward reasoning\n"
    "   - 3 points: Moderate complexity requiring multiple considerations\n"
    "   - 4 points: Complex scenario with trade-offs to analyze\n"
    "   - 5 points: Advanced scenario requiring synthesis of multiple concepts\n\n"
    "4. QUIZ PARAMETERS:\n"
    "   - max_attempts: 3-4 (reasoning questions are harder, allow more attempts)\n"
    "   - max_execution_time: Calculate as (number_of_questions × 90 seconds). Reasoning takes longer.\n"
    "   - min_pass_score: Set to 50-60%% of total score.\n\n"
    "Output strictly as minified JSON with keys: questions (2-4), min_pass_score, max_attempts, max_execution_time.\n"
    "CRITICAL: You have a maximum of 10000 tokens. Ensure your JSON response is COMPLETE and VALID within this limit.\n"
    "If approaching the limit, generate fewer questions but ensure the JSON is properly closed with all required fields.\n"
    "Schema:\n"
    "{{\n"
    "  \"questions\": [ {{ \"question\": string, \"answers\": [string], \"score\": integer }}, ... ],\n"
    "  \"min_pass_score\": number,\n"
    "  \"max_attempts\": integer,\n"
    "  \"max_execution_time\": integer\n"
    "}}\n"
    "Rules: questions must be REASONING QUESTIONS in {language_name} that present scenarios requiring analysis;\n"
    "each question should have ONE answer in the answers array containing the reasoning explanation and recommended actions;\n"
    "scores range 2-5 based on complexity; min_pass_score should be 50-60%% of sum(scores); max_attempts 3-4.\n"
    "Example question format: \"During [scenario], [situation occurs]. How do you react?\"\n"
    "Example answer format: \"[Step-by-step reasoning and recommended actions based on the scenario].\""
)

PROMPT_GIFTQUIZ_EXTRACTED_BLOCK_TEMPLATE: str = "Extracted file text:\n{extracted}\n\n"
PROMPT_GIFTQUIZ_OTHER_BLOCK_TEMPLATE: str = "Other lesson topics (title: excerpt):\n{other_texts}\n\n"
PROMPT_GIFTQUIZ_PREV_BLOCK_TEMPLATE: str = "Previous lessons summaries:\n{prev_text}\n\n"

# ---------------------------------------------------------------------------
# ANSWER VALIDATION PROMPT (from quiz_answer_validation.py)
# ---------------------------------------------------------------------------

VALIDATION_SYSTEM_PROMPT_TEMPLATE: str = """
You are an AI quiz validation agent specialized in assessing student answers to quiz questions.

------------------------------------------------
ROLE & OVERALL GOAL
------------------------------------------------
- You validate how well a student has answered a SINGLE quiz question.
- You must consider:
    - the CURRENT message (user_answer),
    - and all PREVIOUS relevant messages from conversation_history.
- Your job is to:
    1) Judge how well the student's combined answers match the correct answer(s).
    2) Detect the student's current intent (answering, asking for hints, asking for clarification, or off-topic).
    3) Produce a numeric score and a clear explanation when the answer is not yet correct.

You do NOT reveal the correct answer. You only indicate what is missing or incorrect.

------------------------------------------------
INPUT DATA (logical view)
------------------------------------------------
You receive the following data via the user prompt:

1) quiz_question (string)
- The quiz question that the student is trying to answer.

2) correct_answers (JSON array of strings)
- One or more acceptable reference answers.
- Use these to infer the key ideas and required elements of a correct response.
- You must accept semantically equivalent answers, not only exact wording.

3) user_answer (string)
- The student's current message for this quiz turn.
- It may be:
    - a genuine answer attempt,
    - a request for a hint,
    - a request for clarification,
    - or off-topic/small talk.

4) conversation_history (JSON array)
- All previous messages for this quiz question, in chronological order.
- May include messages from both "user" and "agent".
- You MUST:
    - Use all previous user messages that contain answer attempts as part of a COMBINED answer.
    - Use previous agent messages as context (e.g., hints, clarifications) when interpreting the student's current intent.

5) max_possible_score (float)
- Maximum numeric score for this question.
- You must ensure answer_score <= max_possible_score.

6) language (string)
- Language code: "english", "german", or "russian".
- You MUST produce both "reasoning" and "validation_error" entirely in this language only.

------------------------------------------------
COMBINED ANSWER EVALUATION
------------------------------------------------
You must always evaluate the student's understanding based on ALL their answers together:

1) Build a COMBINED ANSWER:
- Collect all previous user messages in conversation_history that contain answer attempts.
- Combine them conceptually with the current user_answer into a single answer (do NOT output this; it is only for evaluation).

2) Compare COMBINED ANSWER with correct_answers:
- Focus on semantic equivalence, not exact wording.
- Identify the key ideas and required elements implied by correct_answers.
- For each key idea, decide if the COMBINED ANSWER:
    - fully covers it,
    - partially covers it,
    - or misses/misunderstands it.

3) Be fair to multi-step attempts:
- If the student initially gives a partial answer and later adds missing parts, treat the combination as one answer.
- Do NOT penalize the student for multiple attempts as long as the final combined content is correct and complete enough.

------------------------------------------------
SCORING LOGIC
------------------------------------------------
You must compute a normalized validation_score in [0.0, 1.0] and a scaled answer_score:

- validation_score:
- 0.0 = completely incorrect / unrelated / fundamentally wrong.
- 0.5-0.79 = partial understanding (some important parts missing or incorrect).
- >= 0.8 = answer is acceptable overall (semantically correct and sufficiently complete).

- answer_score:
- answer_score = validation_score * max_possible_score
- MUST NOT exceed max_possible_score.

Interpretation guide:
- validation_score >= 0.8:
- The COMBINED ANSWER is correct enough to be considered a valid solution.
- Small omissions or minor phrasing issues are acceptable.

- 0.5 <= validation_score < 0.8:
- There is meaningful partial understanding, but one or more important elements are missing or wrong.
- The answer should be considered not yet correct.

- validation_score < 0.5:
- Major misunderstanding or almost no overlap with the required ideas.

------------------------------------------------
USER INTENT CLASSIFICATION (CURRENT MESSAGE)
------------------------------------------------
You MUST classify the student's current message (user_answer) into one of:

- "answer_attempt"
- The student is primarily trying to answer the quiz question (even if the answer is wrong, short, or incomplete).

- "hint_request"
- The student is asking for help, hints, clues, or the answer itself.
- Examples (in any language): "give me a hint", "help me", "what's the answer", "tell me", "how to solve this".

- "clarification_request"
- The student is asking what the question means or asking to clarify the concept or task.
- Examples: "what does this question mean?", "I don't understand the question", "can you explain this term?".

- "off_topic"
- The message is clearly unrelated, extremely short/small talk, or does not contribute to solving the question.

INTENT RULES:
- Base user_intent primarily on the CURRENT message, but use conversation_history for context.
- If the current message is mostly a question about the task or concept, prefer "clarification_request".
- If the current message explicitly asks for a hint or the answer, choose "hint_request".
- Only use "answer_attempt" if the current message is primarily content that could be evaluated as part of an answer.
- If the current message is irrelevant or non-substantive, set "off_topic".

Intent does NOT change how you compute validation_score:
- validation_score is always based on the best COMBINED ANSWER so far in the history.
- user_intent only describes what the student is doing right now.

------------------------------------------------
VALIDATION_ERROR (WHEN validation_score < 0.8)
------------------------------------------------
If validation_score < 0.8, you MUST populate validation_error with a short, specific explanation in the target language. It MUST:

1) Briefly describe, in natural language, what the student's COMBINED ANSWER is doing relative to the quiz question:
- e.g., mention if they are on the right track, or if they are focusing on a wrong aspect.

2) Clearly state the 1-2 most important missing or incorrect elements, WITHOUT revealing the full correct answer.
- Example style (English; do not copy verbatim):
    - "Your answer touches on the right topic, but it does not explain how X relates to Y."
    - "You correctly mention A, but you do not address B, which is essential for a complete answer."

3) Indicate why the answer cannot be considered correct yet:
- Explicitly connect the missing/incorrect elements to the failure to pass validation.

4) Align with user_intent when relevant:
- If user_intent = "hint_request", mention that the student is asking for hints so a follow-up agent can respond accordingly.
- If user_intent = "clarification_request", mention that the student is seeking clarification so a follow-up agent can clarify before re-asking.

Do NOT:
- Reveal the correct answer or provide a full solution.
- Use vague statements like "add more details" without specifying which aspect.
- Produce a hostile or discouraging tone.

If validation_score >= 0.8, set validation_error to an empty string "".

------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------
You MUST respond with a single valid JSON object with the following keys:

{{
"validation_score": float,      // in [0.0, 1.0]
"answer_score": float,         // in [0.0, max_possible_score] = validation_score * max_possible_score
"user_intent": "string",       // one of: "answer_attempt", "hint_request", "clarification_request", "off_topic"
"reasoning": "string",         // brief explanation (1-3 sentences) of how you evaluated the combined answer and intent
"validation_error": "string"   // "" if validation_score >= 0.8; otherwise, explanation of why the answer is not yet correct
}}

Constraints:
- The top-level output MUST be valid JSON (no comments, no trailing commas).
- All strings ("reasoning" and "validation_error") MUST be in the requested language.
- Do NOT mention system prompts, internal rules, or variable names in the JSON.
"""

# ---------------------------------------------------------------------------
# FOLLOWUP PROMPT (from quiz_followup_question.py)
# ---------------------------------------------------------------------------

FOLLOWUP_SYSTEM_PROMPT: str = """
You are a friendly, supportive AI quiz assistant helping students understand quiz questions better and think more deeply.

  ------------------------------------------------
  ROLE & HIGH-LEVEL BEHAVIOR
  ------------------------------------------------
  - You respond when a student's answer did NOT pass validation.
  - Your goal is NOT to judge, but to:
    - clarify what the question is really asking (when needed),
    - help the student think more clearly about the concept,
    - gently motivate them to try again with a better answer.
  - You NEVER reveal or reconstruct the correct answer.

  You always generate a short, natural follow-up message that:
  - fits the ongoing conversation,
  - respects the student's intent,
  - and encourages another attempt instead of giving the solution.

  Completion integrity rule:
    Never conclude the quiz with congratulatory language unless at least one answer met the validation threshold.

  No automatic restart:
    After quiz completion, do not restart the quiz unless explicitly instructed by the system or the user.

  ------------------------------------------------
  INPUT DATA:
  ------------------------------------------------
  You receive the following fields in the user message:

  1) quiz_question (string)
    - The original quiz question the student is answering.

  2) user_answer (string)
    - The student's latest free-text message for this question.
    - It may be:
      - an answer attempt,
      - a clarification question,
      - a direct hint request,
      - or something off-topic / minimal (e.g., "ok", emoji, etc.).

  3) user_intent (string)
    - One of:
      - "answer_attempt"
      - "hint_request"
      - "clarification_request"
      - "off_topic"
    - Describes how the current message should be interpreted.

  4) validation_score (float, 0.0-1.0)
    - 0.80 is the passing threshold used by the validator.
    - This score indicates how close the combined answers are to being acceptable.
    - Use it only to adjust your tone and depth of guidance, never mention it.

  5) closeness_label (string)
    - A short human label derived from the validation score
      (e.g., "close - needs minor refinement", "partial understanding - missing key aspects").
    - Use it internally as a hint about how far the student is, never mention it.

  6) validation_error (string)
    - Human-readable feedback from the validation agent describing why the answer was not accepted
      and which key aspects are missing or unclear.
    - You use it ONLY to understand:
      - what kind of guidance is needed,
      - which concepts are likely missing or misunderstood.
    - You MUST NOT:
      - quote it,
      - closely paraphrase its wording,
      - list all missing pieces explicitly,
      - or reconstruct the correct answer from it.

  7) current_question_messages (JSON array)
    - History of messages for THIS quiz question only (student + assistant), in chronological order.
    - Use it to:
      - see what you already asked,
      - avoid repeating the same guiding questions and openings,
      - avoid asking the student to repeat what they clearly explained before.

  8) conversation_history_all_questions (JSON array)
    - Broader history across ALL quiz questions (student + assistant), in chronological order.
    - Use it to:
      - detect which opening patterns, refusal phrases, and guiding styles were already used,
      - maintain variety in tone and phrasing across the entire quiz.

  9) language (string)
    - Target language indicator (code or full name), e.g.:
      - "en" or "english" -> English
      - "de" or "german"  -> German
      - "ru" or "russian" -> Russian
    - You MUST:
      - normalize this internally,
      - produce the entire reply in exactly that language,
      - never mention language selection logic.

  ------------------------------------------------
  STRICT ANSWER PRIVACY
  ------------------------------------------------
  You must assume you are NOT allowed to expose the correct answer in any way.

  Therefore, you MUST NOT:
  - give the correct answer directly,
  - list all key elements that would obviously reconstruct the solution,
  - step-by-step "walk" the student through the full solution,
  - copy or closely paraphrase anything that looks like the correct solution from any context, including:
    - previous assistant messages,
    - validation_error,
    - or any other input.

  Use validation_error only as a high-level signal about what type of guidance is missing (e.g., "there is a missing condition / key concept / distinction"), NOT as content to echo.

  ------------------------------------------------
  INTENT-BASED BEHAVIOR
  ------------------------------------------------
  Your behavior depends on user_intent and the content of user_answer.

  Always:
  - respect user_intent,
  - keep messages short (2-4 sentences),
  - speak naturally, like a human tutor.

  1) user_intent = "clarification_request"
    - The student is mainly asking what the question means or what they are supposed to do.
    - Behavior:
      - If user_answer contains a direct question, address it FIRST in 1 short, high-level sentence (no hints, no solution).
      - Briefly clarify the task requirement (what kind of answer is expected), NOT the subject matter or underlying concept of the question.
      - Optionally rephrase the quiz_question in simpler language, without giving the answer.
      - Don't make complex question, make it simple and understandable for user.
      - End with ONE gentle prompt inviting them to answer in their own words.
      - Do NOT use analogies, metaphors, comparisons, or explanatory constructions such as "you can imagine this as...", "it is like...", or similar teaching-style framing.
    - Clarification_request constraint: Clarification must focus on the expected answer format, not on interpreting or explaining the question itself.
    - Hard prohibition:
      - Do NOT explain, describe, or restate the subject matter of the question.
      - Do NOT include defining properties, processes, or conditions that narrow the answer space.
    - Clarification_request principle: Clarification is allowed if it helps the student move forward. Clarification becomes blocking if it explains rules, evaluates behavior, or shifts focus away from producing an answer.

  2) user_intent = "hint_request"
    - The student explicitly asks for help, hints, or the answer ("I don't know", "give me a hint", etc.).
    - Behavior:
      - Do NOT provide the answer or hints. You MAY start with a brief refusal sentence, and then prompt the student to produce their own version.
        - Do NOT provide partial solutions or directional hints that reveal the structure of the answer.
        - Do NOT rephrase the question in a way that explains or narrows the subject matter. Rephrasing that keeps the question abstract and open-ended is allowed.
        - Do NOT name or imply the domain, timeframe, category, or conceptual area the answer belongs to.
        - Encourage the student only to make a guess or rephrase the question in their own words.
      - Keep it brief (1-2 sentences) and vary your refusal style across turns (apologetic / encouraging / matter-of-fact), but always supportive.
    - No topic switching on uncertainty:
        When the student says "I don't know", do NOT move to the next question.
    - Allowed support types (choose only ONE, and ONLY if it does NOT narrow the answer space):
        - simplify the wording without adding new semantic content, OR
        - ask a neutral meta-level question about how to approach the answer (e.g., "How would you start thinking about this?").
        Concrete examples are NOT allowed in hint_request.
    - Hint_request response rule:
        - Do NOT explain why an answer is not given.
        - Do NOT mention rules, conditions, formats, or restrictions.
        - Immediately prompt the student to produce their own version or attempt.
    - Allowed phrasing in hint_request: Direct invitations to give a personal version are allowed (e.g., "your version", "in your own words", "as you would describe"), as long as no subject-matter hints are added.

    Refusal phrasing: Do NOT use contrast connectors ("but/aber/however") in refusals. Use a clean refusal + one request for a student attempt.


  3) user_intent = "off_topic"
    - The message is short, non-substantive, or unrelated (e.g., "ok", emoji, small talk).
    - Behavior:
      - Gently redirect the student back to the quiz question.
      - Briefly remind them in general terms what kind of answer is expected (e.g., "explain in a few words...", "name the main idea..."), without hinting at the solution.
      - Invite them to share their own answer now.
      - Do NOT restate or rephrase the quiz question when redirecting from off_topic.
    - Constraint:
        - Choose exactly ONE action per reply.
        - Do NOT combine redirection, format explanation, and invitation in separate sentences.


  4) user_intent = "answer_attempt"
    - The student is genuinely trying to answer, even if partially or incorrectly.
    - First, detect if there is a direct question inside user_answer:
      - If yes, briefly answer that question at a high level (without hints or solution) in your first sentence,
        unless it is effectively another hint_request, in which case handle as hint_request logic but still respond naturally.
    - Do NOT use analogies, metaphors, comparisons, or explanatory constructions such as "you can imagine this as...", "it is like...", or similar teaching-style framing.
    - Then use validation_score and closeness_label to adapt your guidance:

    a) validation_score in [0.6, 0.79]  (answer is close)
        - Briefly acknowledge that they are thinking in a relevant direction (without generic filler like "good question").
        - Encourage them to refine or extend their answer by focusing on 1-2 important aspects that are still missing or unclear (as inferred from validation_error).
        - Ask 1 targeted guiding question that nudges them to those aspects, without revealing them explicitly.
        Close-case hard constraint:
        When the answer is close but too general, the reply must consist of exactly ONE short question requesting more precision.
        Hard prohibitions:
        - Do NOT mention the student, the student's answer, or what it contains.
        - Do NOT use phrases like "you mention", "your answer", "what you said".
        - Do NOT describe correctness, relevance, scope, coverage, or quality.
        - Do NOT include acknowledgments, evaluations, or commentary.
        Only allowed action: Ask a single abstract precision question (e.g., "Can this be stated more precisely?").

    b) validation_score in [0.4, 0.59]  (partial understanding)
        - Recognize that they have some connection to the idea, but there are still notable gaps.
        - Guide them to reconsider the core concept or structure behind the question (e.g., definition, main components, key distinction), using validation_error as a signal.
        - Ask 1-2 guiding questions to help them explore those missing areas.

    Partial-understanding hard constraint: In validation_score [0.4-0.59], the assistant must NOT explain, contextualize, or summarize the question or the student's thinking.
    Do NOT comment on the quality, generality, or completeness of the student's answer.
    Allowed actions: - Ask up to TWO short guiding questions.
    Hard prohibitions:
    - Do NOT describe the student's reasoning or focus ("you think about...", "it sounds like...").
    - Do NOT introduce structural terms such as 'Bereich', 'Rahmen', 'Zusammenhang', 'uebergeordnet', or similar domain-framing labels.
    - Do NOT combine explanation + question.


    c) validation_score < 0.4  (significant gap)
        - Help them step back and think from a more fundamental level (e.g., "what is the goal / definition / main outcome here?").
        - Do NOT tell them they are "wrong"; instead, gently steer them to reconsider basics.
        - Ask at most ONE very general, concept-level question that does NOT mention properties, behavior, comparisons, or consequences that uniquely identify the correct answer.

    - Meaning check before acknowledgment:
        - If the user_answer is unclear, fragmented, or semantically unrelated, do NOT continue with conceptual guidance.
        - You MUST first ask a short clarification question ("Do you mean...?" / "Could you rephrase?") and wait for confirmation.
        - Meaning-check may ask for clarification or prioritization, as long as it does NOT introduce new domain terms or any content that narrows the answer space.
    - Clarify intent first:
        - Use a neutral clarification question (e.g., "Could you rephrase that?").
        - Do NOT add any content that narrows, specifies, or hints at the expected answer while asking for clarification.

        Complexity fallback:
        If the student signals confusion or says "I don't know", reduce conceptual level instead of rephrasing the same complexity.

  ------------------------------------------------
    CONVERSATION RECOVERY & SIMPLIFICATION
    ------------------------------------------------
    1. Core Recovery Principles
        Progress Over Blocking: If a student is incorrect, unclear, or off-target, prioritize keeping the conversation moving.
        Strict No-Refusal Policy: Outside of user_intent = "hint_request", refusals are strictly forbidden. This includes stating inability ("I can't say"), denial phrasing ("I won't provide"), or safety-style disclaimers.
        Single-Action Enforcement: A reply MUST contain exactly ONE recovery action. Multiple sentences are allowed if they all serve the same recovery purpose (e.g. specifying ONE format attribute and inviting the student to attempt an answer).
        Zero Evaluative Framing: Do NOT comment on, assess, or contrast the student's thinking (avoid: "but", "however", "step back", "interesting approach").
        Recovery hierarchy clarification: Single-Action Enforcement applies to off_topic responses as well.

    2. Recovery Strategies (Choose exactly ONE per turn)
        A. Clarification (If intent/meaning is uncertain)
            Action: Ask a short, neutral, interpretive question about what the student meant.
            No Domain Leakage: Do NOT introduce terms, categories, or labels not used by the student. Do NOT suggest what they should have meant.
        B. Simplification (If the task is too complex)
            Action: Strip the task to its core requirement.
            Constraint: Remove abstract wording and secondary conditions. Do NOT introduce new concepts, examples, or hints.
            Simplification may restate the task in more basic terms, but must NOT suggest how to think, analyze, or conceptualize the answer.
            Simplification may specify only ONE format attribute (e.g., length OR structure), not multiple at once.
        C. "I don't know" Handling (Treat as confusion, not a hint request)
            Action: Choose exactly ONE:
            Ask the student to restate the question in their own words.
            Ask a neutral meta-level question about how they might begin.
            Constraint: Do NOT explain, summarize, or interpret the question content.

    3. Negative Constraints & Formatting
        Anti-Stall: Rotate strategies (Clarify -> Simplify -> Re-attempt) if the student is stuck. Avoid repeating the same pattern more than twice.
        No Meta-Discussion: Do NOT mention "approaches," "paths," "origins," "principles," or "sources."
        Neutral Meta Constraint: Questions may refer ONLY to the form/format of the answer, never to the reasoning logic or underlying ideas.


  ------------------------------------------------
  USE OF HISTORY & VARIABILITY
  ------------------------------------------------
  You must use both histories to keep your responses natural and non-repetitive:

  - current_question_messages:
    - See which explanations, clarifications, and guiding questions were already given for THIS question.
    - Avoid:
      - repeating the same opening sentence or very similar phrasings,
      - asking the exact same guiding question again,
      - requesting content the student has already clearly provided.
    - Build an internal "do-not-repeat" set from the last few assistant messages for this question (their first 2-3 words and typical guiding patterns) and avoid reusing them.

  - conversation_history_all_questions:
    - Look at recent assistant replies across the quiz.
    - Avoid reusing:
      - identical or near-identical openings,
      - repeated refusal sentences for hints,
      - the same "template" of encouragement multiple times in a row.
    - Vary:
      - opening style (statement, gentle question, soft suggestion),
      - sentence structure (declarative vs interrogative),
      - vocabulary and connectors.

    - Context anchoring rule:
        Before responding, internally verify that your message refers strictly to the current quiz_question and not to a different concept from earlier turns.

    - No topic drift:
        Never introduce a new data type, concept, or example unless it is explicitly required to understand the current question.

    - Transition clarity:
        When moving to a new question, include a short, natural transition sentence indicating the shift, without evaluation.

  GLOBAL VARIABILITY REQUIREMENTS:
  - Do NOT start multiple replies with the same phrase (e.g., "Let's", "Try to", "Think about").
  - Rotate between:
    - starting with a direct gentle suggestion,
    - starting with a short conceptual reminder,
    - starting with a quick acknowledgment of their effort (without generic phrases like "Interesting question").
  - Avoid meta-comments like:
    - "I see that you asked..."
    - "I notice that you..."
    - "I understand that you..."
  - Avoid generic empty praise like "Good question", "Interesting question".
  - Pattern suppression: Do not reuse the same syntactic structure for guidance (e.g. "You seem to think..., but..."). If a structure was used recently, avoid it.

    Global no-contrast wording:
    Avoid contrast connectors that reframe the student (e.g., "but/aber", "however/jedoch", "instead/stattdessen"). Prefer neutral transitions without opposition (e.g., "Und...", "Okay...", "Dann...").

  ------------------------------------------------
  TONE & LENGTH
  ------------------------------------------------
  - Tone:
    - Warm, respectful, and supportive.
    - Calm and professional, not overly enthusiastic.
    - Never judgmental or dismissive.
  - Length:
    - 2-4 sentences total.
    - Focused on this specific exchange, not on re-explaining the whole topic.
    - One-turn focus:
        Each reply must focus on exactly ONE clarification or ONE guiding angle. Never restate the same idea in multiple sentences.

    - No paraphrase loops:
        Do not rephrase the same clarification in different words within the same message.

    - Conversational language only:
        Avoid teaching-style scaffolding. You are a quiz follow-up, not a lesson.
        Do NOT explain concepts, mechanisms, timelines, comparisons, or typical use cases - only prompt the student to think or answer.

    - Ban abstract meta-terms:
        Avoid phrases like "grundsaetzlich einordnen", "typischerweise", "in welche Kategorie man dies einordnet". Prefer simple everyday wording.

    - Language consistency:
        Stick to one form of address (du or Sie) for the entire quiz. Never mix them.

  ------------------------------------------------
  FINAL RESPONSE CHECK
  ------------------------------------------------
  Before sending the reply, verify that the message contains only ONE communicative action or ONE recovery purpose. If the reply includes more than one directive, invitation, clarification, or question, reduce it to the single most essential one and remove the rest.

  ------------------------------------------------
  OUTPUT FORMAT
  ------------------------------------------------
  - Output ONLY the final follow-up message text.
  - No JSON, no bullet points, no labels, no explanation about rules.
  - Entirely in the target language.
  - The message must:
    - respect user_intent,
    - not reveal or reconstruct the correct answer,
    - feel context-aware and non-template,
    - encourage the student to think and respond again.
"""


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def _init_logs():
    if "api_logs" not in st.session_state:
        st.session_state.api_logs = []


def _log_api_call(
    phase: str,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
    parsed_response: object,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
):
    _init_logs()
    entry = {
        "timestamp": _now_iso(),
        "phase": phase,
        "request": {
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        "response": {
            "raw": raw_response,
            "parsed": parsed_response,
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "duration_ms": round(duration_ms, 1),
    }
    st.session_state.api_logs.append(entry)


# ---------------------------------------------------------------------------
# OPENROUTER CALL
# ---------------------------------------------------------------------------

def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = OPENROUTER_MAX_TOKENS,
    temperature: float = 0.3,
    retries: int = 3,
) -> tuple[str, str, int, int, float]:
    """Returns (content, model_used, input_tokens, output_tokens, duration_ms)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
            )
            duration_ms = (time.perf_counter() - t0) * 1000
            content = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            in_t = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_t = int(getattr(usage, "completion_tokens", 0) or 0)
            resp_model = str(getattr(resp, "model", None) or model)
            return content, resp_model, in_t, out_t, duration_ms
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise last_err if last_err else RuntimeError("OpenRouter call failed")


# ---------------------------------------------------------------------------
# HELPERS (from prompt_review_runner_2.py)
# ---------------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


def safe_json_loads(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(_strip_code_fences(raw))
    except Exception:
        pass
    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return json.loads(match.group(0))
    except Exception:
        return None
    return None


def normalize_short_answers(obj: Optional[dict]) -> tuple[list[dict], dict]:
    questions: list[dict] = []
    quiz: dict = {}
    if isinstance(obj, dict):
        raw_q = obj.get("questions") or []
        if isinstance(raw_q, list):
            for it in raw_q:
                try:
                    q = str((it or {}).get("question") or "").strip()
                    ans = (it or {}).get("answers") or []
                    sc = (it or {}).get("score")
                    if not q or not isinstance(ans, list):
                        continue
                    clean_ans = []
                    for a in ans:
                        s = str(a or "").strip()
                        if s and s not in clean_ans:
                            clean_ans.append(s)
                    if not clean_ans:
                        continue
                    score = int(sc) if isinstance(sc, (int, float)) else 1
                    if score <= 0:
                        score = 1
                    questions.append({"question": q, "answers": clean_ans, "score": score})
                except Exception:
                    continue
        quiz = {
            "min_pass_score": obj.get("min_pass_score"),
            "max_attempts": obj.get("max_attempts"),
            "max_execution_time": obj.get("max_execution_time"),
        }

    total_score = sum(int(q.get("score", 1)) for q in questions)
    num_questions = len(questions)

    mps = quiz.get("min_pass_score")
    try:
        mps_val = float(mps) if mps is not None else None
    except Exception:
        mps_val = None
    if mps_val is None or mps_val >= total_score or mps_val < 0:
        mps_val = max(1.0, round(total_score * 0.55, 2)) if total_score > 0 else 1.0
        if mps_val >= total_score and total_score > 0:
            mps_val = max(1.0, total_score - 1.0)

    att = quiz.get("max_attempts")
    att_val = int(att) if isinstance(att, (int, float)) and 2 <= int(att) <= 5 else (3 if num_questions <= 4 else 4)

    met = quiz.get("max_execution_time")
    if isinstance(met, (int, float)) and int(met) >= 30:
        met_val = int(met)
        met_val = max(num_questions * 30, min(met_val, num_questions * 120))
    else:
        met_val = max(60, num_questions * 45)

    return questions, {"min_pass_score": float(mps_val), "max_attempts": int(att_val), "max_execution_time": int(met_val)}


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------------

def build_system_prompt(course_language: str) -> str:
    return PROMPT_SYSTEM_TEMPLATE.format(
        language_name=get_language_name(course_language),
        course_language=course_language,
    )


def build_giftquiz_questions_prompt(payload: dict, course_language: str = "en", difficulty_level: int = 3) -> str:
    language_name = get_language_name(course_language)
    difficulty_cfg = get_difficulty_config(difficulty_level)
    data = payload.get("data") or {}
    gift_text = (data.get("gift") or "")[:4000]
    extracted = (data.get("extracted_text") or "")[:4000]
    other = data.get("lesson_other_topics") or []
    other_texts = []
    for item in other[:50]:
        title = str(item.get("title") or "")
        text = str(item.get("text") or "")
        if text:
            other_texts.append(f"- {title}: {text[:1000]}")
    prev_summaries = data.get("previous_lesson_summaries") or []
    prev_text = "\n".join(str(s)[:1000] for s in prev_summaries[:20])

    extracted_block = PROMPT_GIFTQUIZ_EXTRACTED_BLOCK_TEMPLATE.format(extracted=extracted) if extracted else ""
    other_block = PROMPT_GIFTQUIZ_OTHER_BLOCK_TEMPLATE.format(other_texts="\n".join(other_texts)) if other_texts else ""
    prev_block = PROMPT_GIFTQUIZ_PREV_BLOCK_TEMPLATE.format(prev_text=prev_text) if prev_text else ""

    return PROMPT_GIFTQUIZ_QUESTIONS_TEMPLATE.format(
        language_name=language_name, course_language=course_language,
        gift_text=gift_text, extracted_block=extracted_block,
        other_block=other_block, prev_block=prev_block,
        difficulty_block=difficulty_cfg["question_generation_instruction"],
    )


def build_giftquiz_reasoning_questions_prompt(payload: dict, course_language: str = "en", difficulty_level: int = 3) -> str:
    language_name = get_language_name(course_language)
    difficulty_cfg = get_difficulty_config(difficulty_level)
    data = payload.get("data") or {}
    gift_text = (data.get("gift") or "")[:4000]
    extracted = (data.get("extracted_text") or "")[:4000]
    other = data.get("lesson_other_topics") or []
    other_texts = []
    for item in other[:50]:
        title = str(item.get("title") or "")
        text = str(item.get("text") or "")
        if text:
            other_texts.append(f"- {title}: {text[:1000]}")
    prev_summaries = data.get("previous_lesson_summaries") or []
    prev_text = "\n".join(str(s)[:1000] for s in prev_summaries[:20])

    extracted_block = PROMPT_GIFTQUIZ_EXTRACTED_BLOCK_TEMPLATE.format(extracted=extracted) if extracted else ""
    other_block = PROMPT_GIFTQUIZ_OTHER_BLOCK_TEMPLATE.format(other_texts="\n".join(other_texts)) if other_texts else ""
    prev_block = PROMPT_GIFTQUIZ_PREV_BLOCK_TEMPLATE.format(prev_text=prev_text) if prev_text else ""

    return PROMPT_GIFTQUIZ_REASONING_QUESTIONS_TEMPLATE.format(
        language_name=language_name, course_language=course_language,
        gift_text=gift_text, extracted_block=extracted_block,
        other_block=other_block, prev_block=prev_block,
        difficulty_block=difficulty_cfg["reasoning_generation_instruction"],
    )


def build_validation_system_prompt(difficulty_level: int = 3) -> str:
    diff_cfg = get_difficulty_config(difficulty_level)
    return VALIDATION_SYSTEM_PROMPT_TEMPLATE + "\n\n" + diff_cfg["validation_instruction"]


def build_validation_user_prompt(
    quiz_question: str,
    user_answer: str,
    correct_answers: list[str],
    conversation_history: list[dict],
    max_possible_score: float,
    language: str,
) -> str:
    return (
        f"INPUT DATA:\n"
        f"quiz_question: ```{quiz_question}```\n\n"
        f"correct_answers (JSON array): ```{json.dumps(correct_answers, ensure_ascii=False)}```\n\n"
        f"user_answer: ```{user_answer}```\n\n"
        f"conversation_history (JSON array): ```{json.dumps(conversation_history, ensure_ascii=False)}```\n\n"
        f"max_possible_score: ```{max_possible_score}```\n\n"
        f"language: ```{language}```"
    )


def build_followup_user_prompt(
    quiz_question: str,
    user_answer: str,
    user_intent: str,
    validation_score: float,
    validation_error: str,
    conversation_history: list[dict],
    language: str,
) -> str:
    if validation_score >= 0.6:
        closeness = "close - needs minor refinement"
    elif validation_score >= 0.4:
        closeness = "partial understanding - missing key aspects"
    else:
        closeness = "significant gap - needs to reconsider approach"

    return (
        f"INPUT DATA:\n"
        f"quiz_question: ```{quiz_question}```\n\n"
        f"user_answer: ```{user_answer}```\n\n"
        f"user_intent: ```{user_intent}```\n\n"
        f"validation_score: ```{validation_score:.2f}```\n\n"
        f"closeness_label: ```{closeness}```\n\n"
        f"validation_error: ```{validation_error}```\n\n"
        f"conversation_history: ```{json.dumps(conversation_history, ensure_ascii=False)}```\n\n"
        f"language: ```{language}```"
    )


# ---------------------------------------------------------------------------
# DEFAULT PAYLOADS (from prompt_review_runner_2.py — only GiftQuiz topics)
# ---------------------------------------------------------------------------

DEFAULT_PAYLOADS: list[dict] = [
    {
        "label": "Quiz #1283 — Quantum Computer (reasoning)",
        "type": "topic",
        "course_id": 27,
        "entity_id": 1283,
        "topic_type": "EscolaLms\\TopicTypeGift\\Models\\GiftQuiz",
        "course_language": "en",
        "data": {
            "title": "Quiz",
            "introduction": "Ok12",
            "topicable_class": "EscolaLms\\TopicTypeGift\\Models\\GiftQuiz",
            "gift": "theProject",
            "topic_json": {"reasoningQuiz": True},
            "questions_count": 4,
            "questions": [
                {"question": "What type of computer performs calculations using the principles of quantum mechanics? {=Quantum computer=A quantum computer}", "type": "short_answers", "max_score": 25},
                {"question": "What is the basic unit of information in a quantum computer, which can represent more than just 0 or 1? {=Qubit=Qubits}", "type": "short_answers", "max_score": 25},
                {"question": "In contrast to quantum computers, what is the basic unit of information in classical computers? {=Bit=Bits}", "type": "short_answers", "max_score": 25},
                {"question": "What field of physics provides the principles that quantum computers use for calculations? {=Quantum mechanics=The principles of quantum mechanics}", "type": "short_answers", "max_score": 25},
            ],
            "lesson_other_topics": [
                {
                    "title": "A quantum computer",
                    "topicable_type": "EscolaLms\\TopicTypes\\Models\\TopicContent\\RichText",
                    "text": (
                        "A quantum computer is a type of computer that uses the principles of "
                        "quantum mechanics to perform calculations. Unlike classical computers "
                        "that use bits to represent either a 0 or a 1, quantum computers use "
                        "qubits which can represent 0, 1, or both simultaneously through a property "
                        "called superposition."
                    ),
                }
            ],
            "previous_lesson_summaries": [],
        },
    },
    {
        "label": "Quiz #1369 — Quantum State (short answers)",
        "type": "topic",
        "course_id": 27,
        "entity_id": 1369,
        "topic_type": "EscolaLms\\TopicTypeGift\\Models\\GiftQuiz",
        "course_language": "en",
        "data": {
            "title": "Quiz",
            "topicable_class": "EscolaLms\\TopicTypeGift\\Models\\GiftQuiz",
            "gift": "theProject",
            "questions_count": 5,
            "questions": [
                {"question": "What term describes the mathematical representation of a quantum system that holds all possible information about it? {=Quantum state}", "type": "short_answers", "max_score": 10},
                {"question": "What is the basic unit of quantum information, analogous to a bit in classical computing? {=Qubit=Quantum bit}", "type": "short_answers", "max_score": 10},
                {"question": "What quantum principle allows a qubit to be in a combination of both 0 and 1 states at the same time? {=Superposition}", "type": "short_answers", "max_score": 10},
                {"question": "What is the name for the quantum phenomenon where multiple qubits are linked and their states are correlated, no matter how far apart they are? {=Entanglement=Quantum entanglement}", "type": "short_answers", "max_score": 10},
                {"question": "According to the lesson material, what is one mathematical object used to represent a quantum state? {=State vector=Wavefunction}", "type": "short_answers", "max_score": 10},
            ],
            "lesson_other_topics": [
                {
                    "title": "A quantum state",
                    "topicable_type": "EscolaLms\\TopicTypes\\Models\\TopicContent\\RichText",
                    "text": (
                        "A quantum state is a mathematical description of a quantum system, like "
                        "a particle, that contains all possible information about it. It is "
                        "represented by a state vector or wavefunction and determines the "
                        "probabilities of the system's properties when measured."
                    ),
                }
            ],
            "previous_lesson_summaries": [
                "This lesson introduces quantum computing, which uses quantum-mechanical phenomena for calculations."
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------------

def _init_state():
    _init_logs()
    defaults = {
        "stage": "setup",          # setup | generating | quiz | results
        "questions": [],           # generated questions
        "quiz_cfg": {},            # min_pass_score, max_attempts, max_execution_time
        "current_q_idx": 0,
        "answers": {},             # {q_idx: {"user_answer": str, "result": dict, "attempts": int, "conversation": []}}
        "total_score": 0.0,
        "payload_json_text": "",
        "is_reasoning": False,
        "difficulty_level": 3,     # 1-5, default intermediate
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to os.getenv."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


def main():
    st.set_page_config(page_title="Quiz Runner", page_icon="📝", layout="wide")
    _init_state()

    api_key = _get_secret("OPENROUTER_API_KEY")
    model = _get_secret("OPENROUTER_MODEL", "google/gemini-2.5-pro")
    validation_model = model

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        model = st.text_input("Model", value=model)
        validation_model = st.text_input("Validation Model", value=model,
                                         help="Model used for answer validation & follow-up")

        st.divider()

        # Logs section
        st.subheader("API Logs")
        st.caption(f"{len(st.session_state.api_logs)} call(s) recorded")

        if st.session_state.api_logs:
            logs_json = json.dumps(st.session_state.api_logs, ensure_ascii=False, indent=2)
            st.download_button(
                "Download full logs (JSON)",
                data=logs_json,
                file_name=f"quiz_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
            if st.button("Clear logs"):
                st.session_state.api_logs = []
                st.rerun()

        st.divider()
        if st.button("Reset quiz", type="secondary"):
            for k in list(st.session_state.keys()):
                if k not in ("api_logs",):
                    del st.session_state[k]
            _init_state()
            st.rerun()

    # ── Main area ─────────────────────────────────────────────────────────
    st.title("Quiz Runner")

    if st.session_state.stage == "setup":
        _render_setup()
    elif st.session_state.stage == "generating":
        _render_generating(api_key, model)
    elif st.session_state.stage == "quiz":
        _render_quiz(api_key, validation_model)
    elif st.session_state.stage == "results":
        _render_results()


# ---------------------------------------------------------------------------
# SETUP STAGE
# ---------------------------------------------------------------------------

def _render_setup():
    st.subheader("1. Choose difficulty level")

    difficulty_options = {v["label"]: k for k, v in DIFFICULTY_LEVELS.items()}
    current_level = st.session_state.get("difficulty_level", 3)
    current_label = DIFFICULTY_LEVELS[current_level]["label"]

    selected_label = st.select_slider(
        "Difficulty level",
        options=list(difficulty_options.keys()),
        value=current_label,
        help=(
            "Level 1 (Beginner): very easy questions, maximum tolerance for imprecise answers. "
            "Level 5 (Expert): very hard questions, answers must be precise."
        ),
    )
    st.session_state.difficulty_level = difficulty_options[selected_label]

    diff_cfg = get_difficulty_config(st.session_state.difficulty_level)
    threshold_pct = int(diff_cfg["validation_threshold"] * 100)
    st.caption(f"Passing threshold for validation: **{threshold_pct}%** match required")

    st.divider()
    st.subheader("2. Select or paste a payload")

    tab_preset, tab_custom = st.tabs(["Preset payloads", "Custom JSON"])

    with tab_preset:
        labels = [p["label"] for p in DEFAULT_PAYLOADS]
        choice = st.selectbox("Choose a preset quiz payload", labels)
        idx = labels.index(choice)
        selected = DEFAULT_PAYLOADS[idx]
        st.json(selected, expanded=False)
        if st.button("Use this payload", key="btn_preset"):
            st.session_state.payload_json_text = json.dumps(selected, ensure_ascii=False, indent=2)
            st.session_state.stage = "generating"
            st.rerun()

    with tab_custom:
        custom = st.text_area("Paste payload JSON", height=300,
                              placeholder='{"type": "topic", "course_language": "en", "data": {...}}')
        if st.button("Use custom payload", key="btn_custom"):
            try:
                parsed = json.loads(custom)
                st.session_state.payload_json_text = json.dumps(parsed, ensure_ascii=False, indent=2)
                st.session_state.stage = "generating"
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")


# ---------------------------------------------------------------------------
# GENERATING STAGE
# ---------------------------------------------------------------------------

def _render_generating(api_key: str, model: str):
    payload = json.loads(st.session_state.payload_json_text)
    course_language = str(payload.get("course_language") or "en").strip() or "en"

    data = payload.get("data") or {}
    topic_json = data.get("topic_json") or {}
    is_reasoning = bool(topic_json.get("reasoningQuiz"))
    st.session_state.is_reasoning = is_reasoning

    difficulty_level = st.session_state.get("difficulty_level", 3)
    diff_cfg = get_difficulty_config(difficulty_level)

    quiz_type = "REASONING" if is_reasoning else "SHORT ANSWER"
    st.info(
        f"Generating **{quiz_type}** questions for entity_id={payload.get('entity_id')} "
        f"| Difficulty: **{diff_cfg['label']}** …"
    )

    if not api_key:
        st.error("Please provide an OpenRouter API key in the sidebar.")
        if st.button("Back"):
            st.session_state.stage = "setup"
            st.rerun()
        return

    system_prompt = build_system_prompt(course_language)
    if is_reasoning:
        user_prompt = build_giftquiz_reasoning_questions_prompt(payload, course_language, difficulty_level)
    else:
        user_prompt = build_giftquiz_questions_prompt(payload, course_language, difficulty_level)

    with st.spinner("Calling LLM to generate questions..."):
        try:
            raw, model_used, in_t, out_t, dur = call_openrouter(
                system_prompt, user_prompt, api_key, model,
            )
        except Exception as e:
            st.error(f"API call failed: {e}")
            if st.button("Back"):
                st.session_state.stage = "setup"
                st.rerun()
            return

    _log_api_call(
        phase="generate_questions",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw,
        parsed_response=None,
        model=model_used,
        input_tokens=in_t,
        output_tokens=out_t,
        duration_ms=dur,
    )

    parsed = safe_json_loads(raw)
    if not parsed:
        st.error("Failed to parse LLM response as JSON.")
        st.code(raw, language="json")
        if st.button("Back"):
            st.session_state.stage = "setup"
            st.rerun()
        return

    questions, quiz_cfg = normalize_short_answers(parsed)
    if not questions:
        st.error("No valid questions generated.")
        st.code(raw, language="json")
        if st.button("Back"):
            st.session_state.stage = "setup"
            st.rerun()
        return

    # Update the log with parsed data
    st.session_state.api_logs[-1]["response"]["parsed"] = {
        "questions": questions,
        "quiz_cfg": quiz_cfg,
    }

    st.session_state.questions = questions
    st.session_state.quiz_cfg = quiz_cfg
    st.session_state.current_q_idx = 0
    st.session_state.answers = {}
    st.session_state.stage = "quiz"
    st.rerun()


# ---------------------------------------------------------------------------
# QUIZ STAGE
# ---------------------------------------------------------------------------

def _render_quiz(api_key: str, validation_model: str):
    questions = st.session_state.questions
    quiz_cfg = st.session_state.quiz_cfg
    q_idx = st.session_state.current_q_idx

    if q_idx >= len(questions):
        st.session_state.stage = "results"
        st.rerun()
        return

    q = questions[q_idx]
    total_q = len(questions)
    max_attempts = quiz_cfg.get("max_attempts", 3)
    course_language = "en"
    try:
        payload = json.loads(st.session_state.payload_json_text)
        course_language = str(payload.get("course_language") or "en").strip() or "en"
    except Exception:
        pass

    difficulty_level = st.session_state.get("difficulty_level", 3)
    diff_cfg = get_difficulty_config(difficulty_level)
    pass_threshold = diff_cfg["validation_threshold"]

    if q_idx not in st.session_state.answers:
        st.session_state.answers[q_idx] = {
            "attempts": 0,
            "best_score": 0.0,
            "best_validation_score": 0.0,
            "conversation": [],
            "passed": False,
            "skipped": False,
        }

    ans_state = st.session_state.answers[q_idx]

    # Progress bar
    progress = q_idx / total_q
    st.progress(progress, text=f"Question {q_idx + 1} of {total_q}")

    # Quiz config info
    cols = st.columns(5)
    cols[0].metric("Min pass score", quiz_cfg.get("min_pass_score", "?"))
    cols[1].metric("Max attempts", max_attempts)
    cols[2].metric("Question score", q["score"])
    cols[3].metric("Attempts used", ans_state["attempts"])
    cols[4].metric("Difficulty", diff_cfg["label"])

    st.divider()

    # Question
    quiz_type_label = "Reasoning" if st.session_state.is_reasoning else "Short Answer"
    st.subheader(f"[{quiz_type_label}] Question {q_idx + 1}")
    st.markdown(f"**{q['question']}**")

    # Conversation history
    if ans_state["conversation"]:
        st.caption("Conversation:")
        for msg in ans_state["conversation"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

    # Check limits
    if ans_state["passed"]:
        st.success(f"Correct! Score: {ans_state['best_score']:.1f}/{q['score']}")
        if st.button("Next question", key=f"next_{q_idx}"):
            st.session_state.current_q_idx = q_idx + 1
            st.rerun()
        return

    if ans_state["attempts"] >= max_attempts:
        st.warning(f"Max attempts reached. Best score: {ans_state['best_score']:.1f}/{q['score']}")
        if st.button("Next question", key=f"next_{q_idx}"):
            st.session_state.current_q_idx = q_idx + 1
            st.rerun()
        return

    # Answer input
    user_answer = st.text_area("Your answer:", key=f"answer_input_{q_idx}_{ans_state['attempts']}", height=100)

    col_submit, col_skip = st.columns([1, 1])

    with col_skip:
        if st.button("Skip question", key=f"skip_{q_idx}"):
            ans_state["skipped"] = True
            st.session_state.current_q_idx = q_idx + 1
            st.rerun()

    with col_submit:
        if st.button("Submit answer", key=f"submit_{q_idx}", type="primary"):
            if not user_answer.strip():
                st.warning("Please enter an answer.")
                return
            if not api_key:
                st.error("Please provide an OpenRouter API key in the sidebar.")
                return

            ans_state["attempts"] += 1
            ans_state["conversation"].append({"role": "user", "content": user_answer.strip()})

            # --- Validation call ---
            val_system = build_validation_system_prompt(difficulty_level)
            val_user = build_validation_user_prompt(
                quiz_question=q["question"],
                user_answer=user_answer.strip(),
                correct_answers=q["answers"],
                conversation_history=ans_state["conversation"],
                max_possible_score=float(q["score"]),
                language=course_language,
            )

            with st.spinner("Validating your answer..."):
                try:
                    raw_val, m_used, in_t, out_t, dur = call_openrouter(
                        val_system, val_user, api_key, validation_model, max_tokens=2000,
                    )
                except Exception as e:
                    st.error(f"Validation API call failed: {e}")
                    ans_state["attempts"] -= 1
                    ans_state["conversation"].pop()
                    return

            val_parsed = safe_json_loads(raw_val)
            _log_api_call(
                phase=f"validate_q{q_idx}_attempt{ans_state['attempts']}",
                system_prompt=val_system,
                user_prompt=val_user,
                raw_response=raw_val,
                parsed_response=val_parsed,
                model=m_used,
                input_tokens=in_t,
                output_tokens=out_t,
                duration_ms=dur,
            )

            if not val_parsed:
                st.error("Failed to parse validation response.")
                st.code(raw_val)
                return

            v_score = float(val_parsed.get("validation_score", 0.0))
            a_score = float(val_parsed.get("answer_score", 0.0))
            user_intent = val_parsed.get("user_intent", "answer_attempt")
            validation_error = val_parsed.get("validation_error", "")

            if a_score > ans_state["best_score"]:
                ans_state["best_score"] = a_score
                ans_state["best_validation_score"] = v_score

            if v_score >= pass_threshold:
                ans_state["passed"] = True
                st.rerun()
                return

            # --- Follow-up call ---
            fu_system = FOLLOWUP_SYSTEM_PROMPT
            fu_user = build_followup_user_prompt(
                quiz_question=q["question"],
                user_answer=user_answer.strip(),
                user_intent=user_intent,
                validation_score=v_score,
                validation_error=validation_error,
                conversation_history=ans_state["conversation"],
                language=course_language,
            )

            with st.spinner("Generating feedback..."):
                try:
                    raw_fu, m_fu, in_fu, out_fu, dur_fu = call_openrouter(
                        fu_system, fu_user, api_key, validation_model,
                        max_tokens=1000, temperature=0.7,
                    )
                except Exception as e:
                    raw_fu = validation_error or "Try again!"
                    m_fu, in_fu, out_fu, dur_fu = validation_model, 0, 0, 0.0

            _log_api_call(
                phase=f"followup_q{q_idx}_attempt{ans_state['attempts']}",
                system_prompt=fu_system,
                user_prompt=fu_user,
                raw_response=raw_fu,
                parsed_response=None,
                model=m_fu,
                input_tokens=in_fu,
                output_tokens=out_fu,
                duration_ms=dur_fu,
            )

            ans_state["conversation"].append({"role": "assistant", "content": raw_fu})
            st.rerun()


# ---------------------------------------------------------------------------
# RESULTS STAGE
# ---------------------------------------------------------------------------

def _render_results():
    questions = st.session_state.questions
    quiz_cfg = st.session_state.quiz_cfg
    answers = st.session_state.answers

    st.subheader("Quiz Results")

    total_possible = sum(q["score"] for q in questions)
    total_earned = sum(answers.get(i, {}).get("best_score", 0.0) for i in range(len(questions)))
    min_pass = quiz_cfg.get("min_pass_score", 0)
    passed = total_earned >= min_pass

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Score", f"{total_earned:.1f} / {total_possible}")
    col2.metric("Min Pass Score", f"{min_pass}")
    col3.metric("Result", "PASSED" if passed else "NOT PASSED")

    if passed:
        st.success("Congratulations! You passed the quiz!")
    else:
        st.error(f"You did not reach the minimum pass score of {min_pass}.")

    st.divider()

    for i, q in enumerate(questions):
        ans = answers.get(i, {})
        status = "Passed" if ans.get("passed") else ("Skipped" if ans.get("skipped") else "Not passed")
        icon = {"Passed": "✅", "Skipped": "⏭️", "Not passed": "❌"}.get(status, "")

        with st.expander(f"{icon} Q{i+1}: {q['question'][:80]}... — {status} ({ans.get('best_score', 0):.1f}/{q['score']})"):
            st.markdown(f"**Correct answers:** {', '.join(q['answers'])}")
            st.markdown(f"**Attempts:** {ans.get('attempts', 0)}")
            if ans.get("conversation"):
                st.caption("Conversation:")
                for msg in ans["conversation"]:
                    if msg["role"] == "user":
                        st.chat_message("user").write(msg["content"])
                    else:
                        st.chat_message("assistant").write(msg["content"])

    st.divider()

    # Logs download (also in sidebar, but convenient here too)
    if st.session_state.api_logs:
        st.subheader("API Logs")
        st.caption(f"{len(st.session_state.api_logs)} API call(s) logged during this session.")
        logs_json = json.dumps(st.session_state.api_logs, ensure_ascii=False, indent=2)
        st.download_button(
            "Download full pipeline logs (JSON)",
            data=logs_json,
            file_name=f"quiz_pipeline_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="results_download_logs",
        )

        with st.expander("Preview logs"):
            for i, log in enumerate(st.session_state.api_logs):
                st.markdown(f"**{i+1}. [{log['phase']}]** — {log['timestamp']} — {log['usage']['input_tokens']}in/{log['usage']['output_tokens']}out — {log['duration_ms']}ms")

    if st.button("Start new quiz"):
        for k in list(st.session_state.keys()):
            if k not in ("api_logs",):
                del st.session_state[k]
        _init_state()
        st.rerun()


if __name__ == "__main__":
    main()
