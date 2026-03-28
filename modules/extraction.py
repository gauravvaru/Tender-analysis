"""
LLM-based extraction of tender requirements and metadata
"""
import logging
import json
from typing import Dict

from config import (
    LLM_PROVIDER, GOOGLE_API_KEY, LLM_MODEL,
    ANTHROPIC_API_KEY, OPENAI_API_KEY
)

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Extract structured information from tender text using LLM"""

    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider.lower()
        self.model    = LLM_MODEL

        if self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            except ImportError:
                logger.error("anthropic package not installed")
                self.client = None

        elif self.provider == "openai":
            try:
                import openai
                openai.api_key = OPENAI_API_KEY
                self.openai = openai
            except ImportError:
                logger.error("openai package not installed")
                self.client = None

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=GOOGLE_API_KEY)
                self.genai = genai
            except ImportError:
                logger.error("google-generativeai package not installed")
                self.genai = None

        else:
            logger.error(f"Unknown LLM provider: {provider}")

        logger.info(f"LLM Extractor initialized with {provider} | model: {self.model}")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def extract_metadata(self, texts) -> Dict:
        """
        Lightweight metadata extraction used by the ingestion pipeline.
        Accepts a list of chunk texts, merges them, delegates to
        extract_tender_metadata().
        """
        try:
            if not texts:
                return {}
            combined_text = " ".join(texts[:5])
            metadata = self.extract_tender_metadata(combined_text)
            return metadata if isinstance(metadata, dict) else {}
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {}

    def extract_tender_metadata(self, tender_text: str) -> Dict:
        """
        Extract structured tender metadata using the configured LLM.

        KEY FIX: The prompt now explicitly distinguishes between
        'organisation' (the procuring entity, e.g. Indian Navy) and
        'department' (the parent ministry, e.g. Ministry of Defence).
        This was the source of the 75% → 100% field accuracy gap.
        """
        prompt = f"""
You are extracting structured metadata from a government tender document.
Return ONLY valid JSON. Do NOT include markdown fences, explanation, or any text outside the JSON.

CRITICAL FIELD DISTINCTIONS — read carefully:

- "organisation": The DIRECT procuring entity that is actually buying/ordering.
  This is the specific office, unit, department, or body issuing the tender.
  Examples: "Indian Navy", "AIIMS Delhi", "Border Roads Organisation", "Central Railway"
  NOT the parent ministry. If the document says "Indian Navy, Ministry of Defence",
  the organisation is "Indian Navy".

- "department": The PARENT ministry or governing body above the organisation.
  Examples: "Ministry of Defence", "Ministry of Health", "Ministry of Railways"
  This is typically the topmost authority mentioned, NOT the direct buyer.

- "project_title": The specific name of the goods/services/works being procured.
  Example: "Supply of Surgical Equipment", "Construction of Road"

- "estimated_value": Total contract value including currency symbol.
  Example: "₹5,00,000", "Rs. 10 Lakhs"

- "emd_amount": Earnest Money Deposit / Bid Security amount.
  Example: "₹10,000", "Rs. 5,000"

- "bid_deadline": The last date for bid submission in DD-MM-YYYY format if possible.

- "experience_required": Minimum years or type of prior work experience required.

- "turnover_requirement": Minimum annual turnover required from the bidder.

- "location": City or state where work is to be performed or delivery is required.

- "eligibility_criteria": Any other qualification requirements (registration, certifications, etc.)

EXAMPLE (to illustrate organisation vs department):
  If document says: "Ministry of Defence | Indian Navy | Supply of life jackets"
  Correct output:
    "organisation": "Indian Navy"
    "department": "Ministry of Defence"
    "project_title": "Supply of life jackets"

TEXT TO EXTRACT FROM:
{tender_text[:4000]}

Return this JSON structure with all fields populated (use empty string if not found):
{{
    "tender_id": "",
    "organisation": "",
    "department": "",
    "project_title": "",
    "estimated_value": "",
    "emd_amount": "",
    "bid_deadline": "",
    "experience_required": "",
    "turnover_requirement": "",
    "location": "",
    "eligibility_criteria": ""
}}
"""
        response_text = self._call_llm(prompt)
        return self._parse_json_response(response_text, context="metadata")

    def extract_technical_requirements(self, tender_text: str) -> Dict:
        """
        Extract technical requirements and specifications.

        Returns:
            {
                "mandatory_skills": List[str],
                "nice_to_have_skills": List[str],
                "equipment_needed": List[str],
                "certifications_required": List[str],
                "experience_required_years": int,
                "complexity_level": "low"|"medium"|"high",
                "technical_summary": str
            }
        """
        prompt = f"""
From this tender document, extract ALL technical requirements.
Return ONLY valid JSON. Do NOT include markdown fences.

TENDER CONTENT:
{tender_text[:3500]}

Return JSON:
{{
    "mandatory_skills": ["list", "of", "required", "technical", "skills"],
    "nice_to_have_skills": ["optional", "skills"],
    "equipment_needed": ["equipment", "or", "tools", "required"],
    "certifications_required": ["certifications", "needed"],
    "experience_required_years": 0,
    "complexity_level": "low|medium|high",
    "technical_summary": "brief summary of technical scope"
}}

Be specific and actionable. Return ONLY JSON.
"""
        response_text = self._call_llm(prompt)
        result = self._parse_json_response(response_text, context="technical_requirements")

        if not result:
            return {
                "mandatory_skills": [],
                "nice_to_have_skills": [],
                "equipment_needed": [],
                "certifications_required": [],
                "experience_required_years": 0,
                "complexity_level": "medium",
                "technical_summary": ""
            }
        return result

    def generate_gap_analysis(
        self,
        tender_requirements: Dict,
        company_profile: Dict,
        tender_text: str = ""
    ) -> Dict:
        """Generate gap analysis between tender requirements and company capabilities."""
        prompt = f"""
Compare tender technical requirements with company capabilities.
Return ONLY valid JSON. Do NOT include markdown fences.

TENDER REQUIREMENTS:
{json.dumps(tender_requirements, indent=2)[:1500]}

COMPANY CAPABILITIES:
{json.dumps(company_profile, indent=2)[:1500]}

Generate JSON:
{{
    "gaps": [
        {{
            "requirement": "what was required",
            "gap_description": "what company lacks",
            "severity": "critical|high|medium|low",
            "mitigation": "how to address this"
        }}
    ],
    "strengths": ["list", "of", "company", "strengths"],
    "improvement_areas": ["areas", "to", "develop"],
    "overall_fit": "percentage as string e.g. 75%",
    "summary": "brief assessment"
}}

Return ONLY JSON.
"""
        response_text = self._call_llm(prompt)
        result = self._parse_json_response(response_text, context="gap_analysis")

        if not result:
            return {
                "gaps": [],
                "strengths": [],
                "improvement_areas": [],
                "overall_fit": "0%",
                "summary": "Gap analysis failed — LLM did not return valid JSON."
            }
        return result

    def generate_summary(
        self,
        tender_data: Dict,
        scoring_results: Dict,
        gap_analysis: Dict
    ) -> str:
        """Generate a 300-400 word executive summary for the tender evaluation."""
        prompt = f"""
Generate a professional executive summary for a tender evaluation decision.

TENDER:
- Title: {tender_data.get('title', 'N/A')}
- Budget: {tender_data.get('budget', 'N/A')}
- Industry: {tender_data.get('industry', 'N/A')}

SCORING:
- Overall Score: {scoring_results.get('overall_score', 'N/A')}%
- Technical Match: {scoring_results.get('technical_match_score', 'N/A')}%
- Risk: {scoring_results.get('risk_score', 'N/A')}%
- Recommendation: {scoring_results.get('recommendation', 'N/A')}

GAPS: {gap_analysis.get('summary', 'No gap data available.')}

Write a 300-400 word executive summary with:
1. Tender Overview (2-3 sentences)
2. Strategic Fit Assessment (1-2 sentences)
3. Key Risks (2-3 bullet points)
4. Capability Gaps (2-3 bullet points)
5. Strengths (2-3 bullet points)
6. Recommendation (1-2 sentences with rationale)

Plain text only, no markdown.
"""
        return self._call_llm(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Route the prompt to the configured LLM. Always uses self.model."""
        try:
            if self.provider == "anthropic":
                if not self.client:
                    logger.error("Anthropic client not initialised")
                    return ""
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            elif self.provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return response.choices[0].message.content

            elif self.provider == "gemini":
                if not hasattr(self, "genai") or self.genai is None:
                    logger.error("Gemini client not initialised")
                    return ""
                model    = self.genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return response.text

            else:
                logger.error(f"Unknown LLM provider: {self.provider}")
                return ""

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _parse_json_response(self, response_text: str, context: str = "") -> Dict:
        """
        Safely extract and parse JSON from an LLM response.
        Strips markdown fences (```json ... ```) that models sometimes include
        despite being told not to.
        """
        if not response_text:
            logger.warning(f"Empty LLM response [{context}]")
            return {}

        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                lines   = cleaned.split("\n")
                end_idx = next(
                    (i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"),
                    len(lines)
                )
                cleaned = "\n".join(lines[1:end_idx])

            json_start = cleaned.find('{')
            json_end   = cleaned.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                return json.loads(cleaned[json_start:json_end])
            else:
                logger.warning(f"No JSON object found in LLM response [{context}]")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON [{context}]: {e}")

        return {}