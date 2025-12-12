# level4_severity.py
# -------------------
# Simple severity mapping based on crime Category.

SEVERITY_MAP = {
    # Severity 1
    "NON-CRIMINAL": 1, "SUSPICIOUS OCCURRENCE": 1, "MISSING PERSON": 1, "RUNAWAY": 1, "RECOVERED VEHICLE": 1,
    # Severity 2
    "WARRANTS": 2, "OTHER OFFENSES": 2, "VANDALISM": 2, "TRESPASS": 2, "DISORDERLY CONDUCT": 2, "BAD CHECKS": 2,
    # Severity 3
    "LARCENY/THEFT": 3, "VEHICLE THEFT": 3, "FORGERY/COUNTERFEITING": 3, "DRUG/NARCOTIC": 3,
    "STOLEN PROPERTY": 3, "FRAUD": 3, "BRIBERY": 3, "EMBEZZLEMENT": 3,
    # Severity 4
    "ROBBERY": 4, "WEAPON LAWS": 4, "BURGLARY": 4, "EXTORTION": 4,
    # Severity 5
    "KIDNAPPING": 5, "ARSON": 5
}

def assign_severity(category: str) -> int:
    if not category:
        return 0
    cat = str(category).strip().upper()
    return SEVERITY_MAP.get(cat, 0)