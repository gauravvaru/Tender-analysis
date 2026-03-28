"""
Metadata evaluation: field-level exact match and partial match
"""

import sys
import os
 
# ── FIX: add project root to sys.path so 'modules' is importable ────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from modules.extraction import LLMExtractor

# Create ground-truth by manually reading your actual PDFs
# and filling in what the correct values should be
GROUND_TRUTH = [
    {
        "source": "GeM-Bidding-8778719.pdf",
        "text_sample": """GEM/2025/B/8778719 Ministry of Defence Indian Navy
                          Supply of Surgical Equipment  Bid end date: 15-04-2026
                          EMD: ₹10,000 Estimated Value: ₹5,00,000""",
        "expected": {
            "tender_id":   "GEM/2025/B/8778719",
            "organisation": "Indian Navy",
            "bid_deadline": "15-04-2026",
            "emd_amount":   "₹10,000",
        }
    },
    # Add one entry per PDF you have — this is your test set
]

def field_match(predicted: str, expected: str) -> bool:
    """Case-insensitive substring match — more forgiving than exact match"""
    if not predicted or not expected:
        return False
    return expected.lower().strip() in predicted.lower().strip()

def evaluate_metadata():
    extractor = LLMExtractor()
    total_fields, correct_fields = 0, 0
    results = []

    for item in GROUND_TRUTH:
        predicted = extractor.extract_tender_metadata(item["text_sample"])
        item_correct = 0

        for field, expected_val in item["expected"].items():
            predicted_val = predicted.get(field, "")
            match = field_match(predicted_val, expected_val)
            total_fields   += 1
            correct_fields += int(match)
            item_correct   += int(match)

            status = "✓" if match else "✗"
            print(f"  {status} {field}: expected='{expected_val}' | got='{predicted_val}'")

        accuracy = item_correct / len(item["expected"])
        results.append({"source": item["source"], "accuracy": round(accuracy, 3)})
        print(f"  → {item['source']}: {accuracy:.0%} field accuracy\n")

    overall = correct_fields / total_fields if total_fields else 0
    print("=" * 50)
    print(f"Overall field accuracy: {overall:.1%}  ({correct_fields}/{total_fields} fields correct)")
    return results

if __name__ == "__main__":
    evaluate_metadata()