import logging
import os
import json
import time
import requests
from typing import Any, Optional

logger = logging.getLogger(__name__)

PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

def fetch_pubmed_abstracts(disease: str, max_results: int = 10) -> list[dict[str, Any]]:
  try:
    params = {"db": "pubmed", "term": f"{disease} drug targets", "retmax": max_results, "retmode": "json"}
    resp = requests.get(PUBMED_URL, params=params).json()
    ids = resp.get("esearchresult", {}).get("idlist", [])
    if not ids: return []
    fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "text", "rettype": "abstract"}
    abstracts = requests.get(PUBMED_FETCH_URL, params=fetch_params).text
    docs = []
    for abs_text in abstracts.split("\n\n"):
      if len(abs_text.strip()) > 50:
        docs.append({"text": abs_text.strip(), "source": "PubMed", "disease": disease})
    return docs
  except Exception as exc:
    logger.warning("PubMed fetch failed: %s", exc)
    return []

def fetch_chembl_targets(disease: str) -> list[dict[str, Any]]:
  try:
    url = f"{CHEMBL_BASE_URL}/target/search"
    params = {"q": disease, "format": "json"}
    resp = requests.get(url, params=params).json()
    targets = resp.get("targets", [])
    docs = []
    for t in targets[:10]:
      text = f"Target: {t.get('pref_name')} ({t.get('target_type')}). {t.get('target_components', [{}])[0].get('description', '')}"
      docs.append({"text": text, "source": "ChEMBL", "disease": disease})
    return docs
  except Exception: return []

def build_knowledge_base(diseases: list[str], output_dir: str = "data/rag_kb", pubmed_per_disease: int = 20) -> list[dict[str, Any]]:
  os.makedirs(output_dir, exist_ok=True)
  all_docs = []
  for d in diseases:
    logger.info("Fetching data for %s", d)
    all_docs.extend(fetch_pubmed_abstracts(d, max_results=pubmed_per_disease))
    all_docs.extend(fetch_chembl_targets(d))
    time.sleep(1) # Rate limit
  with open(os.path.join(output_dir, "kb.json"), "w") as f:
    json.dump(all_docs, f, indent=2)
  return all_docs
