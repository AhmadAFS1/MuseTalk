#!/usr/bin/env python3
"""Regenerate config/languages.json from the compact seed table.

The roster below is a SEED. Replace it with Lingua's authoritative supported-language
list before any production run; every downstream ID is derived from `code`, so a late
change to a code forces a rebuild of that language's characters.

Columns: (code, english_name, native_name, script, rtl, region_key)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "languages.json"

# fmt: off
LANGUAGES: list[tuple[str, str, str, str, bool, str]] = [
    ("af",     "Afrikaans",            "Afrikaans",        "Latin",      False, "southern_africa"),
    ("sq",     "Albanian",             "Shqip",            "Latin",      False, "southeast_europe"),
    ("am",     "Amharic",              "አማርኛ",             "Ethiopic",   False, "horn_africa"),
    ("ar",     "Arabic",               "العربية",            "Arabic",     True,  "arab"),
    ("hy",     "Armenian",             "Հայերեն",           "Armenian",   False, "caucasus"),
    ("az",     "Azerbaijani",          "Azərbaycan dili",  "Latin",      False, "caucasus"),
    ("eu",     "Basque",               "Euskara",          "Latin",      False, "iberia"),
    ("be",     "Belarusian",           "Беларуская",       "Cyrillic",   False, "east_europe"),
    ("bn",     "Bengali",              "বাংলা",              "Bengali",    False, "south_asia"),
    ("bs",     "Bosnian",              "Bosanski",         "Latin",      False, "southeast_europe"),
    ("bg",     "Bulgarian",            "Български",        "Cyrillic",   False, "southeast_europe"),
    ("ca",     "Catalan",              "Català",           "Latin",      False, "iberia"),
    ("ceb",    "Cebuano",              "Cebuano",          "Latin",      False, "southeast_asia"),
    ("ny",     "Chichewa",             "Chichewa",         "Latin",      False, "southern_africa"),
    ("zh",     "Chinese",              "中文",              "Han",        False, "east_asia"),
    ("hr",     "Croatian",             "Hrvatski",         "Latin",      False, "southeast_europe"),
    ("cs",     "Czech",                "Čeština",          "Latin",      False, "central_europe"),
    ("da",     "Danish",               "Dansk",            "Latin",      False, "north_europe"),
    ("nl",     "Dutch",                "Nederlands",       "Latin",      False, "west_europe"),
    ("en",     "English",              "English",          "Latin",      False, "anglosphere"),
    ("et",     "Estonian",             "Eesti",            "Latin",      False, "north_europe"),
    ("fil",    "Filipino",             "Filipino",         "Latin",      False, "southeast_asia"),
    ("fi",     "Finnish",              "Suomi",            "Latin",      False, "north_europe"),
    ("fr",     "French",               "Français",         "Latin",      False, "west_europe"),
    ("gl",     "Galician",             "Galego",           "Latin",      False, "iberia"),
    ("ka",     "Georgian",             "ქართული",          "Georgian",   False, "caucasus"),
    ("de",     "German",               "Deutsch",          "Latin",      False, "west_europe"),
    ("el",     "Greek",                "Ελληνικά",         "Greek",      False, "south_europe"),
    ("gu",     "Gujarati",             "ગુજરાતી",            "Gujarati",   False, "south_asia"),
    ("ht",     "Haitian Creole",       "Kreyòl ayisyen",   "Latin",      False, "caribbean"),
    ("ha",     "Hausa",                "Harshen Hausa",    "Latin",      False, "west_africa"),
    ("haw",    "Hawaiian",             "ʻŌlelo Hawaiʻi",   "Latin",      False, "oceania"),
    ("he",     "Hebrew",               "עברית",             "Hebrew",     True,  "israel"),
    ("hi",     "Hindi",                "हिन्दी",              "Devanagari", False, "south_asia"),
    ("hmn",    "Hmong",                "Hmoob",            "Latin",      False, "southeast_asia"),
    ("hu",     "Hungarian",            "Magyar",           "Latin",      False, "central_europe"),
    ("is",     "Icelandic",            "Íslenska",         "Latin",      False, "north_europe"),
    ("ig",     "Igbo",                 "Asụsụ Igbo",       "Latin",      False, "west_africa"),
    ("id",     "Indonesian",           "Bahasa Indonesia", "Latin",      False, "southeast_asia"),
    ("ga",     "Irish",                "Gaeilge",          "Latin",      False, "anglosphere"),
    ("it",     "Italian",              "Italiano",         "Latin",      False, "south_europe"),
    ("ja",     "Japanese",             "日本語",             "Japanese",   False, "east_asia"),
    ("jv",     "Javanese",             "Basa Jawa",        "Latin",      False, "southeast_asia"),
    ("kn",     "Kannada",              "ಕನ್ನಡ",             "Kannada",    False, "south_asia"),
    ("kk",     "Kazakh",               "Қазақ тілі",       "Cyrillic",   False, "central_asia"),
    ("km",     "Khmer",                "ភាសាខ្មែរ",           "Khmer",      False, "southeast_asia"),
    ("rw",     "Kinyarwanda",          "Ikinyarwanda",     "Latin",      False, "east_africa"),
    ("ko",     "Korean",               "한국어",             "Hangul",     False, "east_asia"),
    ("ku",     "Kurdish (Kurmanji)",   "Kurmancî",         "Latin",      False, "arab"),
    ("ky",     "Kyrgyz",               "Кыргызча",         "Cyrillic",   False, "central_asia"),
    ("lo",     "Lao",                  "ພາສາລາວ",          "Lao",        False, "southeast_asia"),
    ("lv",     "Latvian",              "Latviešu",         "Latin",      False, "north_europe"),
    ("lt",     "Lithuanian",           "Lietuvių",         "Latin",      False, "north_europe"),
    ("lb",     "Luxembourgish",        "Lëtzebuergesch",   "Latin",      False, "west_europe"),
    ("mk",     "Macedonian",           "Македонски",       "Cyrillic",   False, "southeast_europe"),
    ("mg",     "Malagasy",             "Malagasy",         "Latin",      False, "east_africa"),
    ("ms",     "Malay",                "Bahasa Melayu",    "Latin",      False, "southeast_asia"),
    ("ml",     "Malayalam",            "മലയാളം",           "Malayalam",  False, "south_asia"),
    ("mt",     "Maltese",              "Malti",            "Latin",      False, "south_europe"),
    ("mi",     "Maori",                "Te Reo Māori",     "Latin",      False, "oceania"),
    ("mr",     "Marathi",              "मराठी",             "Devanagari", False, "south_asia"),
    ("mn",     "Mongolian",            "Монгол",           "Cyrillic",   False, "east_asia"),
    ("my",     "Burmese",              "မြန်မာဘာသာ",         "Myanmar",    False, "southeast_asia"),
    ("ne",     "Nepali",               "नेपाली",             "Devanagari", False, "south_asia"),
    ("no",     "Norwegian",            "Norsk",            "Latin",      False, "north_europe"),
    ("or",     "Odia",                 "ଓଡ଼ିଆ",              "Odia",       False, "south_asia"),
    ("ps",     "Pashto",               "پښتو",              "Arabic",     True,  "central_asia"),
    ("fa",     "Persian",              "فارسی",             "Arabic",     True,  "iran"),
    ("pl",     "Polish",               "Polski",           "Latin",      False, "central_europe"),
    ("pt",     "Portuguese",           "Português",        "Latin",      False, "lusophone"),
    ("pa",     "Punjabi",              "ਪੰਜਾਬੀ",             "Gurmukhi",   False, "south_asia"),
    ("ro",     "Romanian",             "Română",           "Latin",      False, "southeast_europe"),
    ("ru",     "Russian",              "Русский",          "Cyrillic",   False, "east_europe"),
    ("sm",     "Samoan",               "Gagana Samoa",     "Latin",      False, "oceania"),
    ("gd",     "Scots Gaelic",         "Gàidhlig",         "Latin",      False, "anglosphere"),
    ("sr",     "Serbian",              "Српски",           "Cyrillic",   False, "southeast_europe"),
    ("st",     "Sesotho",              "Sesotho",          "Latin",      False, "southern_africa"),
    ("sn",     "Shona",                "ChiShona",         "Latin",      False, "southern_africa"),
    ("sd",     "Sindhi",               "سنڌي",              "Arabic",     True,  "south_asia"),
    ("si",     "Sinhala",              "සිංහල",             "Sinhala",    False, "south_asia"),
    ("sk",     "Slovak",               "Slovenčina",       "Latin",      False, "central_europe"),
    ("sl",     "Slovenian",            "Slovenščina",      "Latin",      False, "central_europe"),
    ("so",     "Somali",               "Soomaali",         "Latin",      False, "horn_africa"),
    ("es",     "Spanish",              "Español",          "Latin",      False, "hispanophone"),
    ("su",     "Sundanese",            "Basa Sunda",       "Latin",      False, "southeast_asia"),
    ("sw",     "Swahili",              "Kiswahili",        "Latin",      False, "east_africa"),
    ("sv",     "Swedish",              "Svenska",          "Latin",      False, "north_europe"),
    ("tg",     "Tajik",                "Тоҷикӣ",           "Cyrillic",   False, "central_asia"),
    ("ta",     "Tamil",                "தமிழ்",             "Tamil",      False, "south_asia"),
    ("tt",     "Tatar",                "Татарча",          "Cyrillic",   False, "east_europe"),
    ("te",     "Telugu",               "తెలుగు",             "Telugu",     False, "south_asia"),
    ("th",     "Thai",                 "ภาษาไทย",           "Thai",       False, "southeast_asia"),
    ("tr",     "Turkish",              "Türkçe",           "Latin",      False, "turkic"),
    ("tk",     "Turkmen",              "Türkmençe",        "Latin",      False, "central_asia"),
    ("uk",     "Ukrainian",            "Українська",       "Cyrillic",   False, "east_europe"),
    ("ur",     "Urdu",                 "اردو",              "Arabic",     True,  "south_asia"),
    ("ug",     "Uyghur",               "ئۇيغۇرچە",          "Arabic",     True,  "central_asia"),
    ("uz",     "Uzbek",                "Oʻzbekcha",        "Latin",      False, "central_asia"),
    ("vi",     "Vietnamese",           "Tiếng Việt",       "Latin",      False, "southeast_asia"),
    ("cy",     "Welsh",                "Cymraeg",          "Latin",      False, "anglosphere"),
    ("xh",     "Xhosa",                "isiXhosa",         "Latin",      False, "southern_africa"),
    ("yi",     "Yiddish",              "ייִדיש",             "Hebrew",     True,  "israel"),
    ("yo",     "Yoruba",               "Èdè Yorùbá",       "Latin",      False, "west_africa"),
    ("zu",     "Zulu",                 "isiZulu",          "Latin",      False, "southern_africa"),
]
# fmt: on

# Locale variants do not get their own characters: casting is per-language, while the
# locale only changes caption script, TTS voice, and dialogue wording. A language whose
# regional castings should genuinely differ belongs in LANGUAGES as its own row.
LOCALE_VARIANTS: dict[str, list[dict[str, str]]] = {
    "zh": [
        {"locale": "zh-Hans", "label": "Chinese (Simplified)", "script": "Han (Simplified)"},
        {"locale": "zh-Hant", "label": "Chinese (Traditional)", "script": "Han (Traditional)"},
    ],
    "pt": [
        {"locale": "pt-BR", "label": "Portuguese (Brazil)", "script": "Latin"},
        {"locale": "pt-PT", "label": "Portuguese (Portugal)", "script": "Latin"},
    ],
    "es": [
        {"locale": "es-ES", "label": "Spanish (Spain)", "script": "Latin"},
        {"locale": "es-419", "label": "Spanish (Latin America)", "script": "Latin"},
    ],
}

EXCLUDED_CANDIDATES = [
    {"name": "Latin", "reason": "No native conversational speakers; a speaking companion has no authentic register."},
    {"name": "Esperanto", "reason": "Constructed language with no regional identity to cast for."},
    {"name": "Corsican", "reason": "Trimmed to reach the stated 104; reinstate if Lingua ships it."},
    {"name": "Frisian", "reason": "Trimmed to reach the stated 104; reinstate if Lingua ships it."},
]


def main() -> int:
    codes = [row[0] for row in LANGUAGES]
    if len(codes) != len(set(codes)):
        raise SystemExit("Duplicate language codes in the seed table.")
    if len(LANGUAGES) != 104:
        raise SystemExit(f"Seed table holds {len(LANGUAGES)} languages; the roster must be exactly 104.")

    document = {
        "schema_version": 1,
        "status": "seed_requires_confirmation",
        "note": (
            "Seed roster of 104 languages. Replace with Lingua's authoritative supported-language "
            "list before production. Character IDs derive from `code`, so changing a code after "
            "generation orphans that language's pose banks."
        ),
        "characters_per_language": 3,
        "excluded_candidates": EXCLUDED_CANDIDATES,
        "languages": [
            {
                "code": code,
                "name": name,
                "native_name": native,
                "script": script,
                "rtl": rtl,
                "region_key": region,
                **({"locale_variants": LOCALE_VARIANTS[code]} if code in LOCALE_VARIANTS else {}),
            }
            for code, name, native, script, rtl, region in LANGUAGES
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    regions = sorted({row[5] for row in LANGUAGES})
    print(f"Wrote {OUT} with {len(LANGUAGES)} languages across {len(regions)} regions.")
    print("Regions: " + ", ".join(regions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
