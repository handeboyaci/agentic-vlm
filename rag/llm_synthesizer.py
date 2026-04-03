import os
import logging
import json
from typing import Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a computational chemist assisting in drug discovery.
Given the following excerpts from scientific literature for a disease, identify a single most promising protein target.
Extract the target name, uniprot ID, PDB ID, the biological location (either "CNS" or "Systemic"), up to 3 known inhibitors, and a one sentence rationale.
"""


class TargetExtraction(BaseModel):
  target_name: str = Field(description='protein name (e.g. "BACE1")')
  uniprot: str = Field(description='UniProt accession (e.g. "P56817")')
  pdb_id: str = Field(description='representative PDB ID (e.g. "4B7R")')
  location: str = Field(description='"CNS" or "Systemic"')
  known_inhibitors: list[str] = Field(description="list of names (up to 3)")
  rationale: str = Field(description="one-sentence explanation")


def _synthesise_fallback(
  disease: str, documents: list[dict[str, Any]]
) -> dict[str, Any]:
  text = " ".join([d["text"] for d in documents]).lower()
  location = (
    "CNS"
    if any(
      k in disease.lower() or k in text for k in ["alzheimer", "brain", "cns", "neuron"]
    )
    else "Systemic"
  )
  targets = {
    "alzheimer": ("BACE1", "4B7R"),
    "cancer": ("EGFR", "1M17"),
    "covid": ("MPro", "6LU7"),
  }
  name = "Unknown"
  pdb = ""
  for k, (v, p) in targets.items():
    if k in disease.lower():
      name = v
      pdb = p
  return {
    "target_name": name,
    "uniprot": "",
    "pdb_id": pdb,
    "location": location,
    "known_inhibitors": [],
    "rationale": "Keyword-based fallback identification.",
  }


def _synthesise_with_gemini(disease: str, context: str) -> Optional[dict[str, Any]]:
  api_key = os.environ.get("GOOGLE_API_KEY")
  if not api_key:
    return None
  try:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(
      f"{SYSTEM_PROMPT}\n\nDisease: {disease}\nLiterature:\n{context}",
      generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=TargetExtraction,
      ),
    )
    if resp.text:
      return json.loads(resp.text)
  except Exception as exc:
    logger.warning("Gemini synthesis failed: %s", exc)
  return None


def _synthesise_with_openai(disease: str, context: str) -> Optional[dict[str, Any]]:
  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key:
    return None
  try:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.beta.chat.completions.parse(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
          "role": "user",
          "content": f"Disease: {disease}\nLiterature: {context}",
        },
      ],
      response_format=TargetExtraction,
    )
    if resp.choices[0].message.parsed:
      return resp.choices[0].message.parsed.model_dump()
  except Exception as exc:
    logger.warning("OpenAI synthesis failed: %s", exc)
  return None


def synthesise(disease: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
  context = "\n".join([d["text"][:500] for d in documents[:5]])
  res = _synthesise_with_gemini(disease, context) or _synthesise_with_openai(
    disease, context
  )
  if res:
    return res
  return _synthesise_fallback(disease, documents)
