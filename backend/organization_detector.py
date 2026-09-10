import re


ORG_ALIASES = {
    "state bank of india": [
        "sbi",
        "sbi bank",
        "state bank of india"
    ],

    "sbi bank": [
        "sbi",
        "sbi bank",
        "state bank of india"
    ],

    "hdfc bank": [
        "hdfc",
        "hdfc bank"
    ],

    "icici bank": [
        "icici",
        "icici bank"
    ],

    "axis bank": [
        "axis",
        "axis bank"
    ],

    "bank of baroda": [
        "bob",
        "bank of baroda"
    ],

    "bob": [
        "bob",
        "bank of baroda"
    ]
}


def normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def contains_phrase(
    text: str,
    phrase: str
) -> bool:

    """
    Match complete words only.

    Prevents:
        sbi -> matching inside random words
        axis -> accidental partial matches
    """

    return bool(
        re.search(
            rf"\b{re.escape(phrase)}\b",
            text
        )
    )


def detect_organization(
    transcript: str,
    organizations
):

    if not transcript:
        return None

    normalized_transcript = normalize_text(
        transcript
    )

    for organization in organizations:

        organization_name = (
            organization.organization or ""
        )

        domain = (
            organization.domain or ""
        )

        normalized_name = normalize_text(
            organization_name
        )

        # ============================================
        # 1. EXACT ORGANIZATION NAME
        # ============================================

        if (
            normalized_name
            and contains_phrase(
                normalized_transcript,
                normalized_name
            )
        ):

            return {
                "organization_id": organization.id,
                "organization": organization_name,
                "domain": domain,
                "matched_by": "organization_name"
            }

        # ============================================
        # 2. ALIAS MATCHING
        # ============================================

        aliases = ORG_ALIASES.get(
            normalized_name,
            []
        )

        for alias in aliases:

            normalized_alias = normalize_text(
                alias
            )

            if (
                normalized_alias
                and contains_phrase(
                    normalized_transcript,
                    normalized_alias
                )
            ):

                return {
                    "organization_id": organization.id,
                    "organization": organization_name,
                    "domain": domain,
                    "matched_by": "alias"
                }

        # ============================================
        # 3. DOMAIN MATCHING
        # ============================================

        normalized_domain = (
            domain
            .lower()
            .replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .split("/")[0]
        )

        domain_name = (
            normalized_domain
            .split(".")[0]
        )

        if (
            domain_name
            and contains_phrase(
                normalized_transcript,
                domain_name
            )
        ):

            return {
                "organization_id": organization.id,
                "organization": organization_name,
                "domain": domain,
                "matched_by": "domain"
            }

    return None