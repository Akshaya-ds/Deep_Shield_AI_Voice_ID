import re
import math


# ============================================================
# DEEPSHIELD CONVERSATION RISK ENGINE
# ============================================================
#
# This is an explainable MVP risk engine.
#
# It does NOT decide whether a voice is AI-generated.
# It analyzes the CONTENT of the conversation.
#
# Signals:
#
# 1. Urgency
# 2. Threat / consequence
# 3. Financial manipulation
# 4. Credential / OTP requests
# 5. KYC pressure
# 6. Payment / transfer requests
# 7. Account security threats
# 8. Suspicious combinations
#
# The system uses contextual phrase groups rather than
# relying on a single keyword.
# ============================================================


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str):

    if not text:
        return ""

    text = text.lower()

    # Normalize common currency formatting
    text = text.replace("₹", " rs ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# RISK PHRASE GROUPS
# ============================================================

URGENCY_PATTERNS = [

    r"\bimmediately\b",
    r"\burgent\b",
    r"\burgently\b",
    r"\bright now\b",
    r"\bdo it now\b",
    r"\bact now\b",
    r"\bas soon as possible\b",
    r"\bwithout delay\b",
    r"\btoday itself\b",
    r"\bwithin \d+ (minutes?|hours?)\b",
]


THREAT_PATTERNS = [

    r"\baccount will be blocked\b",
    r"\baccount will be closed\b",
    r"\baccount will be suspended\b",
    r"\baccount will be frozen\b",
    r"\bcard will be blocked\b",
    r"\bservice will be stopped\b",
    r"\baccess will be blocked\b",
    r"\byou will lose access\b",
    r"\botherwise\b",
    r"\bif you don't\b",
    r"\bif you do not\b",
    r"\bpenalty\b",
    r"\bfine\b",
]


FINANCIAL_THREAT_PATTERNS = [

    r"\bmoney will be deducted\b",
    r"\bamount will be deducted\b",
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:rs|rupees?)\b",
    r"\b(?:rs|rupees?)\s*\d[\d,]*(?:\.\d+)?\b",
    r"\blose (?:your )?money\b",
    r"\bmoney will be lost\b",
    r"\bfinancial loss\b",
    r"\bcharge will be applied\b",
    r"\bamount will be debited\b",
]


OTP_CREDENTIAL_PATTERNS = [

    r"\botp\b",
    r"\bone time password\b",
    r"\bverification code\b",
    r"\bsecurity code\b",
    r"\bpin\b",
    r"\bmpin\b",
    r"\bpassword\b",
    r"\bpasscode\b",
    r"\bsecret code\b",
    r"\bcvv\b",
    r"\bcard number\b",
    r"\baccount number\b",
    r"\bshare your otp\b",
    r"\btell me your otp\b",
    r"\bprovide your otp\b",
    r"\bsend your otp\b",
]


KYC_PATTERNS = [

    r"\bkyc\b",
    r"\bknow your customer\b",
    r"\bkyc update\b",
    r"\bkyc verification\b",
    r"\bkyc process\b",
    r"\bcomplete kyc\b",
    r"\bupdate kyc\b",
]


PAYMENT_PATTERNS = [

    r"\btransfer money\b",
    r"\btransfer the money\b",
    r"\bsend money\b",
    r"\bmake a payment\b",
    r"\bpay immediately\b",
    r"\bsend the amount\b",
    r"\bdeposit money\b",
    r"\bbank transfer\b",
    r"\bupi\b",
    r"\bpayment link\b",
    r"\bscan the qr\b",
    r"\bscan qr\b",
]


PERSONAL_INFO_PATTERNS = [

    r"\bshare your details\b",
    r"\bprovide your details\b",
    r"\bshare your personal information\b",
    r"\bdate of birth\b",
    r"\badhar\b",
    r"\baadhaar\b",
    r"\bpan number\b",
    r"\bidentity proof\b",
    r"\bpersonal information\b",
]


# ============================================================
# PATTERN MATCHING
# ============================================================

def _find_matches(text, patterns):

    matches = []

    for pattern in patterns:

        try:

            found = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if found:

                matches.append(
                    pattern
                )

        except re.error:

            continue

    return matches


# ============================================================
# CATEGORY SCORE
# ============================================================

def _category_score(
    matches,
    maximum
):

    if not matches:

        return 0.0

    return min(
        maximum,
        len(matches) * maximum / 2
    )


# ============================================================
# CONTEXTUAL RISK ANALYSIS
# ============================================================

def analyze_conversation_risk(
    transcript: str
):

    text = normalize_text(
        transcript
    )

    if not text:

        return {

            "risk_score": 0.0,

            "risk_level": "LOW",

            "risk_flags": [],

            "matched_signals": {},

            "explanation":
                "No conversation text was available."
        }

    # --------------------------------------------------------
    # Detect individual categories
    # --------------------------------------------------------

    urgency = _find_matches(
        text,
        URGENCY_PATTERNS
    )

    threats = _find_matches(
        text,
        THREAT_PATTERNS
    )

    financial = _find_matches(
        text,
        FINANCIAL_THREAT_PATTERNS
    )

    credentials = _find_matches(
        text,
        OTP_CREDENTIAL_PATTERNS
    )

    kyc = _find_matches(
        text,
        KYC_PATTERNS
    )

    payments = _find_matches(
        text,
        PAYMENT_PATTERNS
    )

    personal_info = _find_matches(
        text,
        PERSONAL_INFO_PATTERNS
    )

    # --------------------------------------------------------
    # Individual scores
    # --------------------------------------------------------

    urgency_score = _category_score(
        urgency,
        20
    )

    threat_score = _category_score(
        threats,
        25
    )

    financial_score = _category_score(
        financial,
        25
    )

    credential_score = _category_score(
        credentials,
        30
    )

    kyc_score = _category_score(
        kyc,
        10
    )

    payment_score = _category_score(
        payments,
        25
    )

    personal_score = _category_score(
        personal_info,
        15
    )

    # --------------------------------------------------------
    # Contextual combinations
    #
    # This is the important part.
    #
    # KYC alone is NOT automatically a scam.
    #
    # KYC + urgency
    # KYC + threat
    # KYC + financial consequence
    #
    # are much more suspicious.
    # --------------------------------------------------------

    combination_bonus = 0.0

    combination_flags = []

    # KYC + urgency
    if kyc and urgency:

        combination_bonus += 15

        combination_flags.append(
            "KYC combined with urgency"
        )

    # KYC + threat
    if kyc and threats:

        combination_bonus += 20

        combination_flags.append(
            "KYC combined with account threat"
        )

    # KYC + financial threat
    if kyc and financial:

        combination_bonus += 20

        combination_flags.append(
            "KYC combined with financial consequence"
        )

    # OTP + urgency
    if credentials and urgency:

        combination_bonus += 20

        combination_flags.append(
            "Credential request combined with urgency"
        )

    # OTP + threat
    if credentials and threats:

        combination_bonus += 20

        combination_flags.append(
            "Credential request combined with threat"
        )

    # Payment + urgency
    if payments and urgency:

        combination_bonus += 20

        combination_flags.append(
            "Payment request combined with urgency"
        )

    # Payment + threat
    if payments and threats:

        combination_bonus += 20

        combination_flags.append(
            "Payment request combined with threat"
        )

    # Financial + threat
    if financial and threats:

        combination_bonus += 15

        combination_flags.append(
            "Financial consequence combined with threat"
        )

    # --------------------------------------------------------
    # Calculate raw risk
    # --------------------------------------------------------

    raw_risk = (

        urgency_score

        + threat_score

        + financial_score

        + credential_score

        + kyc_score

        + payment_score

        + personal_score

        + combination_bonus
    )

    # --------------------------------------------------------
    # Saturating normalization
    #
    # Prevents score from simply growing linearly
    # beyond 100.
    # --------------------------------------------------------

    risk_score = (
        100
        *
        (
            1
            -
            math.exp(
                -raw_risk / 55
            )
        )
    )

    risk_score = max(
        0.0,
        min(
            100.0,
            risk_score
        )
    )

    risk_score = round(
        risk_score,
        2
    )

    # --------------------------------------------------------
    # Risk flags
    # --------------------------------------------------------

    risk_flags = []

    if urgency:

        risk_flags.append(
            "URGENT_ACTION"
        )

    if threats:

        risk_flags.append(
            "THREAT_OR_CONSEQUENCE"
        )

    if financial:

        risk_flags.append(
            "FINANCIAL_PRESSURE"
        )

    if credentials:

        risk_flags.append(
            "CREDENTIAL_OR_OTP_REQUEST"
        )

    if kyc:

        risk_flags.append(
            "KYC_REFERENCE"
        )

    if payments:

        risk_flags.append(
            "PAYMENT_OR_TRANSFER_REQUEST"
        )

    if personal_info:

        risk_flags.append(
            "PERSONAL_INFORMATION_REQUEST"
        )

    risk_flags.extend(
        combination_flags
    )

    # Remove duplicates
    risk_flags = list(
        dict.fromkeys(
            risk_flags
        )
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if risk_score >= 75:

        risk_level = "CRITICAL"

    elif risk_score >= 50:

        risk_level = "HIGH"

    elif risk_score >= 25:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    if risk_level == "CRITICAL":

        explanation = (
            "Conversation contains multiple high-risk "
            "signals such as urgency, threats, financial "
            "pressure, credential requests or suspicious "
            "combinations."
        )

    elif risk_level == "HIGH":

        explanation = (
            "Conversation contains significant scam-risk "
            "signals that require caution."
        )

    elif risk_level == "MEDIUM":

        explanation = (
            "Conversation contains some potentially "
            "suspicious requests or pressure signals."
        )

    else:

        explanation = (
            "No strong scam-risk combination was detected "
            "in the conversation."
        )

    return {

        "risk_score": risk_score,

        "risk_level": risk_level,

        "risk_flags": risk_flags,

        "matched_signals": {

            "urgency": len(urgency),

            "threats": len(threats),

            "financial_pressure": len(financial),

            "credentials": len(credentials),

            "kyc": len(kyc),

            "payment": len(payments),

            "personal_information": len(
                personal_info
            ),

            "contextual_combinations": len(
                combination_flags
            )
        },

        "component_scores": {

            "urgency": round(
                urgency_score,
                2
            ),

            "threat": round(
                threat_score,
                2
            ),

            "financial": round(
                financial_score,
                2
            ),

            "credential": round(
                credential_score,
                2
            ),

            "kyc": round(
                kyc_score,
                2
            ),

            "payment": round(
                payment_score,
                2
            ),

            "personal_information": round(
                personal_score,
                2
            ),

            "contextual_bonus": round(
                combination_bonus,
                2
            )
        },

        "explanation": explanation
    }