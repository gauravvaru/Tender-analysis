"""
Utility functions
"""
import pandas as pd
from pathlib import Path
from config import TENDERS_CSV, COMPANY_PROFILE_JSON
import json
import logging

logger = logging.getLogger(__name__)


def load_tenders_registry() -> pd.DataFrame:
    """Load tender metadata from CSV"""
    if Path(TENDERS_CSV).exists():
        return pd.read_csv(TENDERS_CSV)
    return pd.DataFrame()


def save_tender_metadata(tender_data: dict):
    """Save tender metadata to CSV"""
    df_new = pd.DataFrame([tender_data])
    
    if Path(TENDERS_CSV).exists():
        df_existing = pd.read_csv(TENDERS_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(TENDERS_CSV, index=False)
    logger.info(f"Tender saved: {tender_data.get('tender_id')}")


def load_company_profile() -> dict:
    """Load company profile from JSON"""
    if Path(COMPANY_PROFILE_JSON).exists():
        with open(COMPANY_PROFILE_JSON) as f:
            return json.load(f)
    
    # Return default template
    return {
        "company_name": "",
        "skills": [],
        "certifications": [],
        "experience_years": 0,
        "team_size": 0,
        "industries": [],
        "served_regions": [],
        "max_project_value": 0,
        "industry_certifications": []
    }


def save_company_profile(profile: dict):
    """Save company profile to JSON"""
    with open(COMPANY_PROFILE_JSON, 'w') as f:
        json.dump(profile, f, indent=2)
    logger.info("Company profile saved")