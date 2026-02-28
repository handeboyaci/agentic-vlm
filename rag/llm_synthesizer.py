import os
import logging
import json
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a computational chemist assisting in drug discovery.
Given the following excerpts from scientific literature for a disease, identify a single most promising protein target.
Return a minified JSON object with these keys:
- target_name: protein name (e.g. "BACE1")
- uniprot: UniProt accession (e.g. "P56817")
- pdb_id: representative PDB ID (e.g. "4B7R")
- location: "CNS" or "Systemic"
- known_inhibitors: list of names (up to 3)
- rationale: one-sentence explanation
"""

def _synthesise_fallback(disease: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
  text = " ".join([d["text"] for d in documents]).lower()
  location = "CNS" if any(k in disease.lower() or k in text for k in ["alzheimer", "brain", "cns", "neuron"]) else "Systemic"
  targets = {"alzheimer": "BACE1", "cancer": "EGFR", "covid": "MPro"}
  name = "Unknown"
  for k, v in targets.items():
    if k in disease.lower(): name = v
  return {"target_name": name, "uniprot": "", "pdb_id": "", "location": location, "known_inhibitors": [], "rationale": "Keyword-based fallback identification."}

def _synthesise_with_gemini(disease: str, context: str) -> Optional[dict[str, Any]]:
  api_key = os.environ.get("GOOGLE_API_KEY")
  if not api_key: return None
  try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-latest")
    resp = model.generate_content(f"{SYSTEM_PROMPT}\n\nDisease: {disease}\nLiterature:\n{context}")
    match = re.search(r"\{.*\}", resp.text, re.DOTALL)
    if match: return json.loads(match.group(0))
  except Exception as exc:
    logger.warning("Gemini synthesis failed: %s", exc)
  return None

def _synthesise_with_openai(disease: str, context: str) -> Optional[dict[str, Any]]:
  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key: return None
  try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Disease: {disease}\nLiterature: {context}"}]
    )
    return json.loads(resp.choices[0].message.content)
  except Exception as exc:
    logger.warning("OpenAI synthesis failed: %s", exc)
  return None

def synthesise(disease: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
  context = "\n".join([d["text"][:500] for d in documents[:5]])
  res = _synthesise_with_gemini(disease, context) or _synthesise_with_openai(disease, context)
  if res: return res
  return _synthesise_fallback(disease, documents)
