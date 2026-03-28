"""
Tender scoring and ranking logic
"""
import logging
from typing import Dict, List

from config import SCORING_WEIGHTS

logger = logging.getLogger(__name__)


class TenderScorer:
    """Score tenders based on company capabilities"""

    def __init__(self, weights: Dict = None):
        self.weights = weights or SCORING_WEIGHTS
        self._validate_weights()
        logger.info("Tender Scorer initialized")

    def _validate_weights(self):
        """Ensure weights sum to 1.0; normalise if they don't."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total:.3f}, normalising…")
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_tender(
        self,
        tender_data: Dict,
        company_profile: Dict,
        extracted_requirements: Dict = None
    ) -> Dict:
        """
        Calculate a comprehensive tender score.

        Args:
            tender_data:            {tender_id, title, budget, deadline, industry, region, …}
            company_profile:        {skills, certifications, experience_years, team_size, …}
            extracted_requirements: {mandatory_skills, …} from LLM extraction (optional)

        Returns:
            {
                tender_id, title,
                technical_match_score   (0-100),
                risk_score              (0-100, higher = more risk),
                capability_alignment_score (0-100),
                compliance_score        (0-100),
                overall_score           (0-100),
                recommendation          ("pursue" | "review" | "pass"),
                breakdown               {detailed per-dimension dicts}
            }
        """
        reqs = extracted_requirements or {}

        tech_score        = self._calculate_technical_match(reqs, company_profile)
        risk_score        = self._calculate_risk_score(tender_data, company_profile)
        capability_score  = self._calculate_capability_alignment(company_profile)
        compliance_score  = self._calculate_compliance_score(tender_data, company_profile)

        w = self.weights

        # FIX 8: Original formula was wrong — risk was not multiplied by its weight
        # and the overall could exceed 100.
        # Correct formula: risk contribution REDUCES the score proportionally.
        # overall = w_tech*tech + w_cap*cap + w_comp*comp + w_risk*(100 - risk)
        overall_score = (
            w.get("technical_match",     0.35) * tech_score +
            w.get("capability_alignment", 0.25) * capability_score +
            w.get("compliance_score",     0.15) * compliance_score +
            w.get("risk_score",           0.25) * (100.0 - risk_score)
        )
        overall_score = max(0.0, min(overall_score, 100.0))

        recommendation = self._get_recommendation(overall_score, risk_score)

        return {
            "tender_id":                    tender_data.get("tender_id", "unknown"),
            "title":                        tender_data.get("title", ""),
            "technical_match_score":        round(tech_score,       2),
            "risk_score":                   round(risk_score,        2),
            "capability_alignment_score":   round(capability_score, 2),
            "compliance_score":             round(compliance_score, 2),
            "overall_score":                round(overall_score,    2),
            "recommendation":               recommendation,
            "breakdown": {
                "technical_match": {
                    "score":       round(tech_score, 2),
                    "weight":      w.get("technical_match", 0.35),
                    "explanation": f"Skills match: {tech_score:.0f}%"
                },
                "risk": {
                    "score":       round(risk_score, 2),
                    "weight":      w.get("risk_score", 0.25),
                    "explanation": self._get_risk_explanation(risk_score)
                },
                "capability": {
                    "score":       round(capability_score, 2),
                    "weight":      w.get("capability_alignment", 0.25),
                    "explanation": f"Team readiness: {capability_score:.0f}%"
                },
                "compliance": {
                    "score":       round(compliance_score, 2),
                    "weight":      w.get("compliance_score", 0.15),
                    "explanation": f"Regulatory fit: {compliance_score:.0f}%"
                }
            }
        }

    # ------------------------------------------------------------------
    # Individual score components
    # ------------------------------------------------------------------

    def _calculate_technical_match(
        self,
        requirements: Dict,
        company_profile: Dict
    ) -> float:
        """Score based on mandatory skills overlap (0-100)."""
        required_skills = set(requirements.get("mandatory_skills", []))
        company_skills  = set(company_profile.get("skills", []))

        if not required_skills:
            # FIX 9: Return 50 only when LLM gave no requirements (uncertain),
            # but log it clearly so the user knows it's a default.
            logger.info("No mandatory skills extracted — defaulting technical match to 50")
            return 50.0

        matched  = required_skills & company_skills
        coverage = (len(matched) / len(required_skills)) * 100.0

        logger.info(
            f"Technical match: {coverage:.1f}% "
            f"({len(matched)}/{len(required_skills)} skills)"
        )
        return min(coverage, 100.0)

    def _calculate_risk_score(
        self,
        tender_data: Dict,
        company_profile: Dict
    ) -> float:
        """Calculate risk (0-100, higher = riskier)."""
        risk = 0.0

        # Budget vs company's max project value
        budget             = float(tender_data.get("budget",           0) or 0)
        max_project_value  = float(company_profile.get("max_project_value", 0) or 0)

        if max_project_value > 0:
            if budget > max_project_value * 2:
                risk += 30
            elif budget > max_project_value:
                risk += 15

        # Timeline
        timeline_days = int(tender_data.get("timeline_days", 180) or 180)
        if timeline_days < 90:
            risk += 25
        elif timeline_days < 180:
            risk += 10

        # Geography
        company_regions = set(company_profile.get("served_regions", []))
        tender_region   = tender_data.get("region", "")
        if tender_region and tender_region not in company_regions:
            risk += 20

        # Industry familiarity
        company_industries = set(company_profile.get("industries", []))
        tender_industry    = tender_data.get("industry", "")
        if tender_industry and tender_industry not in company_industries:
            risk += 15

        logger.info(f"Risk score: {risk:.1f}")
        return min(risk, 100.0)

    def _calculate_capability_alignment(self, company_profile: Dict) -> float:
        """Measure overall company readiness (0-100)."""
        score = 0.0

        # Experience — 40 pts
        experience_years = float(company_profile.get("experience_years", 0) or 0)
        if experience_years >= 5:
            score += 40
        elif experience_years >= 2:
            score += 30
        elif experience_years > 0:
            score += 15

        # Team size — 30 pts
        team_size = int(company_profile.get("team_size", 0) or 0)
        if team_size >= 20:
            score += 30
        elif team_size >= 10:
            score += 20
        elif team_size >= 5:
            score += 10

        # Certifications — up to 30 pts (7.5 pts each, capped)
        certifications = company_profile.get("certifications", [])
        score += min(len(certifications) * 7.5, 30)

        logger.info(f"Capability alignment: {score:.1f}")
        return min(score, 100.0)

    def _calculate_compliance_score(
        self,
        tender_data: Dict,
        company_profile: Dict
    ) -> float:
        """Regulatory and compliance readiness (0-100)."""
        score = 0.0
        certs = {c.replace(" ", "").upper() for c in company_profile.get("certifications", [])}

        # Standard ISO certifications — 25 pts each
        if "ISO9001" in certs:
            score += 25
        if "ISO45001" in certs:
            score += 25
        if "ISO14001" in certs:
            score += 25

        # Industry-specific compliance — 25 pts
        industry       = tender_data.get("industry", "").lower()
        industry_certs = [c.lower() for c in company_profile.get("industry_certifications", [])]

        if any(k in industry for k in ("power", "electrical")):
            if any("power" in c for c in industry_certs):
                score += 25
        elif any(k in industry for k in ("construction", "civil")):
            if any("construction" in c for c in industry_certs):
                score += 25
        else:
            # Give partial credit when industry is unknown / not matched
            if industry_certs:
                score += 10

        logger.info(f"Compliance score: {score:.1f}")
        return min(score, 100.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_recommendation(self, overall_score: float, risk_score: float) -> str:
        if risk_score > 75:
            return "pass"       # Too risky regardless of score
        if overall_score >= 70:
            return "pursue"
        if overall_score >= 45:
            return "review"
        return "pass"

    def _get_risk_explanation(self, risk_score: float) -> str:
        if risk_score > 75:
            return "Very high risk — check timeline, budget, and location"
        if risk_score > 50:
            return "Moderate-to-high risk — evaluate carefully"
        if risk_score > 25:
            return "Moderate risk — manageable with planning"
        return "Low risk — good strategic fit"


# ----------------------------------------------------------------------

class TenderRanker:
    """Rank and filter a list of scored tenders."""

    @staticmethod
    def rank_tenders(
        scored_tenders: List[Dict],
        sort_by: str = "overall_score",
        min_score: float = 0.0,
        max_risk: float = 100.0
    ) -> List[Dict]:
        """
        Filter and sort tenders, then append a 1-based rank field.

        Args:
            scored_tenders: output of TenderScorer.score_tender() for each tender
            sort_by:        field to sort by (default: overall_score)
            min_score:      include only tenders with overall_score >= min_score
            max_risk:       include only tenders with risk_score <= max_risk

        Returns:
            Sorted list of tender dicts with a "rank" field added.
        """
        filtered = [
            t for t in scored_tenders
            if (t.get("overall_score", 0) >= min_score
                and t.get("risk_score", 0) <= max_risk)
        ]

        # For risk_score, lower is better → ascending sort
        reverse = (sort_by != "risk_score")
        ranked = sorted(filtered, key=lambda x: x.get(sort_by, 0), reverse=reverse)

        for rank, tender in enumerate(ranked, 1):
            tender["rank"] = rank

        logger.info(
            f"Ranked {len(ranked)} tenders (filtered from {len(scored_tenders)})"
        )
        return ranked
