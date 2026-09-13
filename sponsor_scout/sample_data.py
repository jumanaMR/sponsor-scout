# sample_data.py
#
# SYNTHETIC / ILLUSTRATIVE dataset.
# ----------------------------------
# These are made-up companies and made-up postings, written only to exercise
# the chunking -> embedding -> retrieval -> sponsorship-signal pipeline below.
# They are NOT real leads and must not be treated as evidence that any real
# company sponsors visas. Replace this with real postings you collect
# yourself (see sponsor_leads_template.csv) before trusting any output.

SAMPLE_POSTINGS = [
    {
        "id": "syn-001",
        "company": "Nimbus Data Labs",
        "country": "Australia",
        "role_title": "AI Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-08-01",
        "text": (
            "Nimbus Data Labs is hiring an AI Engineer to join our Sydney-based "
            "ML platform team. We are an approved sponsor and can support a "
            "Temporary Skill Shortage (subclass 482) visa for the right "
            "candidate, including offshore applicants relocating from overseas. "
            "You'll work on RAG pipelines, vector search, and LLM evaluation. "
            "3+ years experience with Python and ML frameworks required."
        ),
    },
    {
        "id": "syn-002",
        "company": "Aurora Analytics Pty Ltd",
        "country": "Australia",
        "role_title": "Machine Learning Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-07-18",
        "text": (
            "Aurora Analytics is expanding our applied ML team in Melbourne. "
            "Unfortunately, we are not able to offer visa sponsorship at this "
            "time and can only consider candidates with existing Australian "
            "work rights. Strong background in NLP and production ML systems."
        ),
    },
    {
        "id": "syn-003",
        "company": "Harbor Robotics",
        "country": "Australia",
        "role_title": "Automation & AI Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-08-20",
        "text": (
            "Harbor Robotics builds warehouse automation software. We welcome "
            "applications from international candidates and will sponsor a "
            "skilled visa (subclass 482 or equivalent) for exceptional "
            "offshore hires in robotics, computer vision, or AI engineering."
        ),
    },
    {
        "id": "syn-004",
        "company": "Delta Cloud Systems",
        "country": "Netherlands",
        "role_title": "AI/ML Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-06-30",
        "text": (
            "Delta Cloud Systems (Amsterdam) is a recognized sponsor under the "
            "Dutch highly skilled migrant scheme. We actively sponsor "
            "residence and work permits for offshore hires joining our AI "
            "engineering team, covering relocation costs for the employee."
        ),
    },
    {
        "id": "syn-005",
        "company": "Nordwind Systeme GmbH",
        "country": "Germany",
        "role_title": "Machine Learning Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-07-02",
        "text": (
            "Nordwind Systeme in Berlin is hiring. We support the EU Blue "
            "Card process for qualified non-EU applicants and have sponsored "
            "several offshore engineers in the past two years for our "
            "computer vision team."
        ),
    },
    {
        "id": "syn-006",
        "company": "Thistle & Byte",
        "country": "United Kingdom",
        "role_title": "AI Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-05-14",
        "text": (
            "Thistle & Byte is a licensed Skilled Worker sponsor based in "
            "London. This role is eligible for Skilled Worker visa "
            "sponsorship and we regularly hire offshore candidates relocating "
            "from South Asia and the Middle East."
        ),
    },
    {
        "id": "syn-007",
        "company": "Cliffside Softworks",
        "country": "Ireland",
        "role_title": "ML Platform Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-08-05",
        "text": (
            "Cliffside Softworks (Dublin) is not currently registered as a "
            "sponsor for employment permits and can only hire candidates who "
            "already have the right to work in Ireland or the EU."
        ),
    },
    {
        "id": "syn-008",
        "company": "Maple Ridge AI",
        "country": "Canada",
        "role_title": "AI Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-08-22",
        "text": (
            "Maple Ridge AI in Toronto participates in the Global Talent "
            "Stream and can support a work permit plus LMIA sponsorship for "
            "strong offshore candidates with RAG, vector database, and LLM "
            "production experience."
        ),
    },
    {
        "id": "syn-009",
        "company": "Quillfeather Studio",
        "country": "Australia",
        "role_title": "AI/Automation Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-09-01",
        "text": (
            "Quillfeather Studio is a small Brisbane startup building "
            "automation tooling with n8n and LLM integrations. No mention of "
            "visa sponsorship in this posting; sponsorship status unclear, "
            "recommend contacting recruiter directly to confirm."
        ),
    },
    {
        "id": "syn-010",
        "company": "Silverline Cognition",
        "country": "Australia",
        "role_title": "Senior AI Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-07-27",
        "text": (
            "Silverline Cognition is an approved 482 and ENS sponsor. We "
            "have a dedicated immigration team supporting relocation for "
            "offshore AI and automation engineering hires, including "
            "visa costs and a relocation allowance."
        ),
    },
    {
        "id": "syn-011",
        "company": "Ferngully Data Co",
        "country": "Germany",
        "role_title": "AI Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-06-11",
        "text": (
            "Ferngully Data Co in Munich requires candidates to already hold "
            "an EU work permit. We do not sponsor visas for this position."
        ),
    },
    {
        "id": "syn-012",
        "company": "Brightcask Technologies",
        "country": "Netherlands",
        "role_title": "Machine Learning / Automation Engineer",
        "source": "synthetic-example",
        "date_posted": "2026-08-14",
        "text": (
            "Brightcask Technologies (Rotterdam) offers visa sponsorship and "
            "30% ruling support for qualifying international hires joining "
            "our AI automation team. Experience with vector databases, "
            "chunking pipelines, and RAG systems is a strong plus."
        ),
    },
]
