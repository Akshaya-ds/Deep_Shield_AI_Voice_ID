import re


RISK_PATTERNS = {

    "Urgency": [
        "immediately",
        "urgent",
        "right now",
        "act now",
        "within minutes",
        "quickly"
    ],

    "Money Request": [
        "transfer",
        "send money",
        "payment",
        "pay",
        "₹",
        "rs.",
        "rupees"
    ],

    "Credential Request": [
        "otp",
        "password",
        "pin",
        "cvv",
        "verification code"
    ],

    "Threat Language": [
        "account blocked",
        "account compromised",
        "legal action",
        "police",
        "arrest",
        "security threat"
    ],

    "Suspicious Link": [
        "click this link",
        "open this link",
        "bit.ly",
        "http://",
        "https://"
    ]
}


def analyze_transcript(text: str):

    text = text.lower()

    detected = {}

    total_score = 0

    for category, keywords in RISK_PATTERNS.items():

        matches = []

        for keyword in keywords:

            if keyword in text:
                matches.append(keyword)

        if matches:

            detected[category] = matches

            if category == "Money Request":
                total_score += 30

            elif category == "Credential Request":
                total_score += 25

            elif category == "Threat Language":
                total_score += 20

            elif category == "Urgency":
                total_score += 15

            elif category == "Suspicious Link":
                total_score += 20

    total_score = min(total_score, 100)

    if total_score >= 70:
        level = "HIGH"

    elif total_score >= 40:
        level = "MEDIUM"

    else:
        level = "LOW"

    return {
        "risk_score": total_score,
        "risk_level": level,
        "detected_patterns": detected
    }