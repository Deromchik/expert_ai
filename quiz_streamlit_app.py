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

1) quiz_question (string) - The quiz question that the student is trying to answer.
2) correct_answers (JSON array of strings) - One or more acceptable reference answers. Use these to infer the key ideas and required elements of a correct response. You must accept semantically equivalent answers, not only exact wording.
3) user_answer (string) - The student's current message for this quiz turn.
4) conversation_history (JSON array) - All previous messages for this quiz question, in chronological order.
5) max_possible_score (float) - Maximum numeric score for this question. You must ensure answer_score <= max_possible_score.
6) language (string) - Language code. You MUST produce both "reasoning" and "validation_error" entirely in this language only.

------------------------------------------------
COMBINED ANSWER EVALUATION
------------------------------------------------
You must always evaluate the student's understanding based on ALL their answers together:

1) Build a COMBINED ANSWER: Collect all previous user messages in conversation_history that contain answer attempts. Combine them conceptually with the current user_answer into a single answer.
2) Compare COMBINED ANSWER with correct_answers: Focus on semantic equivalence, not exact wording.
3) Be fair to multi-step attempts.

------------------------------------------------
SCORING LOGIC
------------------------------------------------
- validation_score: 0.0 to 1.0. >= 0.8 means answer is acceptable overall.
- answer_score = validation_score * max_possible_score. MUST NOT exceed max_possible_score.

------------------------------------------------
USER INTENT CLASSIFICATION
------------------------------------------------
Classify the student's current message into one of: "answer_attempt", "hint_request", "clarification_request", "off_topic".

------------------------------------------------
VALIDATION_ERROR (WHEN validation_score < 0.8)
------------------------------------------------
If validation_score < 0.8, populate validation_error with a short, specific explanation in the target language, WITHOUT revealing the correct answer.
If validation_score >= 0.8, set validation_error to "".

------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------
You MUST respond with a single valid JSON object:
{{
  "validation_score": float,
  "answer_score": float,
  "user_intent": "string",
  "reasoning": "string",
  "validation_error": "string"
}}
"""

# ---------------------------------------------------------------------------
# FOLLOWUP PROMPT (from quiz_followup_question.py, simplified)
# ---------------------------------------------------------------------------

FOLLOWUP_SYSTEM_PROMPT: str = """
You are a friendly, supportive AI quiz assistant helping students understand quiz questions better and think more deeply.

- You respond when a student's answer did NOT pass validation.
- Your goal is NOT to judge, but to help the student think more clearly about the concept.
- You NEVER reveal or reconstruct the correct answer.
- Keep messages short (2-4 sentences), supportive, and encouraging.
- Vary your opening style and avoid repeating the same guiding questions.

You receive:
- quiz_question: The original quiz question
- user_answer: The student's latest message
- user_intent: One of "answer_attempt", "hint_request", "clarification_request", "off_topic"
- validation_score: How close the answer is (0.0 to 1.0, 0.8+ = pass)
- validation_error: Why the answer was not accepted
- conversation_history: Previous messages for this question
- language: Target language

Output ONLY the follow-up message text. No JSON, no bullet points, no labels.
Entirely in the target language.
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


def build_giftquiz_questions_prompt(payload: dict, course_language: str = "en") -> str:
    language_name = get_language_name(course_language)
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
    )


def build_giftquiz_reasoning_questions_prompt(payload: dict, course_language: str = "en") -> str:
    language_name = get_language_name(course_language)
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
    )


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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Quiz Runner", page_icon="📝", layout="wide")
    _init_state()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input("OpenRouter API Key", type="password",
                                value=os.getenv("OPENROUTER_API_KEY", ""))
        model = st.text_input("Model", value=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro"))
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
    st.subheader("1. Select or paste a payload")

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

    quiz_type = "REASONING" if is_reasoning else "SHORT ANSWER"
    st.info(f"Generating **{quiz_type}** questions for entity_id={payload.get('entity_id')} …")

    if not api_key:
        st.error("Please provide an OpenRouter API key in the sidebar.")
        if st.button("Back"):
            st.session_state.stage = "setup"
            st.rerun()
        return

    system_prompt = build_system_prompt(course_language)
    if is_reasoning:
        user_prompt = build_giftquiz_reasoning_questions_prompt(payload, course_language)
    else:
        user_prompt = build_giftquiz_questions_prompt(payload, course_language)

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

    # Init answer state for this question
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
    cols = st.columns(4)
    cols[0].metric("Min pass score", quiz_cfg.get("min_pass_score", "?"))
    cols[1].metric("Max attempts", max_attempts)
    cols[2].metric("Question score", q["score"])
    cols[3].metric("Attempts used", ans_state["attempts"])

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
            val_system = VALIDATION_SYSTEM_PROMPT_TEMPLATE
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

            if v_score >= 0.8:
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
