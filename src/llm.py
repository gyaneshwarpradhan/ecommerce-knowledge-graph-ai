"""
llm.py - Google Gemini API integration using google-genai SDK.

Two functions:
  generate_query(question)   → structured query dict  (LLM call #1)
  generate_answer(question, retrieved_data) → final answer string (LLM call #2)

The LLM is NEVER allowed to execute code. Call #1 converts natural language to
a controlled JSON operation; call #2 synthesises a grounded answer from the
retrieved data only.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

from src.prompts import (
    ANSWER_GENERATOR_SYSTEM,
    ANSWER_GENERATOR_USER_TEMPLATE,
    QUERY_PLANNER_SYSTEM,
    QUERY_PLANNER_USER_TEMPLATE,
)

# Supported operations — used for validation after LLM call #1
SUPPORTED_OPERATIONS = {
    "products_by_brand",
    "products_by_vendor",
    "products_by_brand_vendor",
    "products_by_category",
    "customer_orders",
    "product_details",
    "search_value",
    "unsupported",
}

# Required parameters per operation
REQUIRED_PARAMS: dict[str, list[str]] = {
    "products_by_brand": ["brand"],
    "products_by_vendor": ["vendor"],
    "products_by_brand_vendor": ["brand", "vendor"],
    "products_by_category": ["category"],
    "customer_orders": ["customer"],
    "product_details": ["product"],
    "search_value": ["value"],
    "unsupported": [],
}

FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )
    return genai.Client(api_key=api_key)


def _get_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


def _call_gemini_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries_per_model: int = 2,
    initial_delay: float = 2.0,
) -> str:
    """
    Call Gemini API with model fallback and exponential backoff retry.
    Tries the configured model first, then candidate fallbacks if quota is exceeded.
    """
    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_error: Exception | None = None

    for candidate_model in models_to_try:
        delay = initial_delay
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                if "429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str:
                    if attempt < max_retries_per_model - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    # Move to next model if quota exhausted
                    break
                # Non-quota error (e.g. invalid config) — break
                break

    if last_error:
        raise last_error
    return ""


def validate_query(query: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that the LLM-generated query has a supported operation and all
    required fields. Returns (is_valid, error_message).
    """
    if not isinstance(query, dict):
        return False, "Query must be a dictionary."

    operation = query.get("operation")

    if operation not in SUPPORTED_OPERATIONS:
        return False, f"Unknown operation: '{operation}'. Supported: {SUPPORTED_OPERATIONS}"

    required = REQUIRED_PARAMS.get(operation, [])
    missing = [p for p in required if not query.get(p)]
    if missing:
        return False, f"Operation '{operation}' is missing required fields: {missing}"

    return True, ""


def generate_query(question: str) -> dict[str, Any]:
    """
    LLM Call #1 — Query Planner.

    Converts a natural-language question into a validated structured query dict.
    Falls back to {"operation": "unsupported"} on any error.
    """
    try:
        client = _get_client()
        model = _get_model()

        config = types.GenerateContentConfig(
            system_instruction=QUERY_PLANNER_SYSTEM,
            response_mime_type="application/json",
            temperature=0.0,
        )

        user_content = QUERY_PLANNER_USER_TEMPLATE.format(question=question)

        raw = _call_gemini_with_retry(
            client=client,
            model=model,
            contents=user_content,
            config=config,
        )

        query = json.loads(raw or "{}")

    except json.JSONDecodeError as exc:
        print(f"[LLM] JSON parse error in query planner: {exc}")
        return {"operation": "unsupported"}
    except Exception as exc:
        print(f"[LLM] Error calling query planner: {exc}")
        return {"operation": "unsupported"}

    # Validate before returning — never trust raw LLM output
    is_valid, error = validate_query(query)
    if not is_valid:
        print(f"[LLM] Query validation failed: {error}")
        return {"operation": "unsupported"}

    return query


def generate_answer(question: str, retrieved_data: Any) -> str:
    """
    LLM Call #2 — Answer Generator.

    Receives only the original question and the retrieved graph data.
    Must not invent any facts.
    """
    try:
        client = _get_client()
        model = _get_model()

        # Serialise retrieved data to a readable string
        if isinstance(retrieved_data, (dict, list)):
            data_str = json.dumps(retrieved_data, indent=2, ensure_ascii=False)
        else:
            data_str = str(retrieved_data) if retrieved_data else "No data retrieved."

        user_message = ANSWER_GENERATOR_USER_TEMPLATE.format(
            question=question,
            retrieved_data=data_str,
        )

        config = types.GenerateContentConfig(
            system_instruction=ANSWER_GENERATOR_SYSTEM,
            temperature=0.2,
        )

        return _call_gemini_with_retry(
            client=client,
            model=model,
            contents=user_message,
            config=config,
        ) or "No answer generated."

    except Exception as exc:
        return f"[LLM Error] Could not generate answer: {exc}"
