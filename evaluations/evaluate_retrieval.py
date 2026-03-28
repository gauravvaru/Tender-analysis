"""
Metadata extraction evaluation: field-level accuracy
Also generates a ChatGPT-comparable report for side-by-side benchmarking.

Run from the project root:
    python evaluations/evaluate_metadata.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.extraction import LLMExtractor

# ── Ground-truth test set ─────────────────────────────────────────────────────
# HOW TO ADD MORE ENTRIES:
#   1. Open the PDF in a viewer
#   2. Copy the first 300-400 words of text into "text_sample"
#   3. Fill "expected" with the correct values you can see in the PDF
#   4. Run this script and see if the LLM finds them
GROUND_TRUTH = [
    {
        "source": "GeM-Bidding-8778719.pdf",
        "text_sample": """
            GEM/2025/B/8778719
            Ministry of Defence
            Indian Navy
            Supply of Surgical Equipment
            Bid end date: 15-04-2026
            EMD: Rs. 10,000
            Estimated Value: Rs. 5,00,000
            Location: Mumbai
            Eligibility: Registered supplier on GeM portal
        """,
        "expected": {
            "tender_id":          "GEM/2025/B/8778719",
            "organisation":       "Indian Navy",
            "department":         "Ministry of Defence",
            "project_title":      "Surgical Equipment",
            "estimated_value":    "5,00,000",
            "emd_amount":         "10,000",
            "bid_deadline":       "15-04-2026",
            "location":           "Mumbai",
        },
    },
    {
        "source": "GeM-Bidding-9029086.pdf",
        "text_sample": """
            GEM/2025/B/9029086
            Ministry of Defence
            Indian Army
            Supply of Office Furniture
            Bid end date: 20-05-2026
            EMD: Rs. 5,000
            Estimated Value: Rs. 2,50,000
            Location: Delhi
            Eligibility: OEM or authorised dealer
        """,
        "expected": {
            "tender_id":       "GEM/2025/B/9029086",
            "organisation":    "Indian Army",
            "department":      "Ministry of Defence",
            "project_title":   "Office Furniture",
            "estimated_value": "2,50,000",
            "emd_amount":      "5,000",
            "bid_deadline":    "20-05-2026",
            "location":        "Delhi",
        },
    },
    # ── Add your remaining PDFs here ──────────────────────────────────────
    # Open each PDF, copy the first page text, fill in expected values.
    # The more entries you add, the more meaningful the accuracy score.
]


def field_match(predicted: str, expected: str) -> bool:
    """
    Substring match (case-insensitive, strips currency/whitespace).
    Handles variations like 'Rs. 10,000' vs '10,000' vs '₹10,000'.
    """
    if not predicted or not expected:
        return False
    # Strip currency symbols and extra whitespace for comparison
    clean = lambda s: (
        s.lower()
         .replace("rs.", "").replace("rs ", "").replace("₹", "")
         .replace(",", "").strip()
    )
    return clean(expected) in clean(predicted)


def evaluate_metadata(print_report: bool = True) -> dict:
    """
    Evaluate metadata extraction accuracy field by field.
    Returns a report dict suitable for comparison against ChatGPT.
    """
    extractor = LLMExtractor()

    total_fields   = 0
    correct_fields = 0
    doc_results    = []

    if print_report:
        print(f"\nEvaluating metadata extraction — {len(GROUND_TRUTH)} documents\n")
        print("=" * 65)

    for item in GROUND_TRUTH:
        if print_report:
            print(f"\nDocument: {item['source']}")
            print("-" * 45)

        predicted    = extractor.extract_tender_metadata(item["text_sample"])
        item_correct = 0
        field_log    = []

        for field, expected_val in item["expected"].items():
            predicted_val = str(predicted.get(field, ""))
            match         = field_match(predicted_val, str(expected_val))

            total_fields   += 1
            correct_fields += int(match)
            item_correct   += int(match)

            field_log.append({
                "field":     field,
                "expected":  expected_val,
                "predicted": predicted_val,
                "correct":   match,
            })

            if print_report:
                status = "✓" if match else "✗"
                print(
                    f"  {status}  {field:22s} "
                    f"expected='{expected_val}'  |  got='{predicted_val}'"
                )

        doc_accuracy = item_correct / len(item["expected"]) if item["expected"] else 0
        doc_results.append({
            "source":   item["source"],
            "accuracy": round(doc_accuracy, 3),
            "correct":  item_correct,
            "total":    len(item["expected"]),
            "fields":   field_log,
        })

        if print_report:
            print(f"\n  → {doc_accuracy:.0%} field accuracy  "
                  f"({item_correct}/{len(item['expected'])} correct)")

    overall = correct_fields / total_fields if total_fields else 0.0

    report = {
        "system":           "Gemini RAG (this project)",
        "total_fields":     total_fields,
        "correct_fields":   correct_fields,
        "overall_accuracy": round(overall, 3),
        "documents":        doc_results,
    }

    if print_report:
        print("\n" + "=" * 65)
        print(
            f"Overall field accuracy: {overall:.1%}  "
            f"({correct_fields}/{total_fields} fields correct)"
        )

        if overall >= 0.85:
            verdict = "🟢 Excellent"
        elif overall >= 0.70:
            verdict = "🟡 Good — one or two fields being confused"
        elif overall >= 0.50:
            verdict = "🟠 Acceptable — prompt needs more examples"
        else:
            verdict = "🔴 Poor — check model and prompt"
        print(f"Verdict: {verdict}\n")

        # Save report for ChatGPT comparison
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "metadata_eval_report.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")
        print("Use this file with compare_chatgpt.py to benchmark against ChatGPT.\n")

    return report


if __name__ == "__main__":
    evaluate_metadata()