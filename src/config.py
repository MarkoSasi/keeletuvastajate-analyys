
SUBREDDITS = ["Eesti"]  # Säilita ainult r/Eesti

# Peamine subreddit (tagasiühilduvuseks)
SUBREDDIT = "Eesti"

# Soovitud unikaalsete lausete arv
TARGET_SENTENCES = 1000

# Minimaalne sõnade arv lauses
MIN_WORDS = 5

# Päringuviivitus sekundites (päringutiheduse piiramise vältimiseks)
REQUEST_DELAY = 2.5

# User-agent stringid rotateerumiseks
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Mitmekesised otsingusõnad, jaotatud kategooriatesse
# Laiendatud, et maksimeerida katvust
SEARCH_KEYWORDS = {
    "general_estonian": [
        # Levinud eesti sõnad, mis esinevad paljudes postitustes
        "minu", "meie", "tema", "nende", "kõik", "palju", "väga", "hästi",
        "arvan", "mõtlen", "tegelikult", "ilmselt", "vist", "kindlasti",
        "probleem", "küsimus", "vastus", "teema", "asi", "elu", "inimesed",
    ],
    "questions_discussions": [
        # Küsimuste algused ja arutluseteemad
        "kuidas", "miks", "kas keegi", "kas on", "mis arvate", "soovitan",
        "kogemus", "nõuanne", "abi", "help", "advice", "recommendation",
    ],
    "technology": [
        "IT", "startup", "programmeerija", "coding", "tech", "developer",
        "arvuti", "software", "app", "AI", "crypto", "blockchain",
        "telefon", "internet", "online", "website", "server", "database",
    ],
    "entertainment": [
        "Netflix", "film", "muusika", "gaming", "YouTube", "Spotify",
        "seriaal", "mängud", "anime", "podcast", "TikTok", "stream",
        "movie", "show", "series", "concert", "festival",
    ],
    "daily_life": [
        "töö", "kool", "ülikool", "toidukott", "shopping", "rent",
        "korter", "palk", "töökoht", "job", "interview", "remote",
        "elu", "päev", "hommik", "õhtu", "nädalavahetus", "puhkus",
    ],
    "emotions_opinions": [
        # Emotsioone ja arvamusi väljendavad sõnad
        "viha", "rõõm", "kurb", "ilus", "naljakas", "imelik", "hull",
        "parim", "halvim", "lemmik", "vihkan", "armastan", "meeldib",
        "cringe", "weird", "funny", "crazy", "best", "worst",
    ],
    "relationships": [
        "sõber", "pere", "vanemad", "lapsed", "abikaasa", "kallim",
        "suhe", "dating", "tinder", "relationship", "breakup", "wedding",
    ],
    "money_finance": [
        "raha", "palk", "hind", "odav", "kallis", "soodne", "ost",
        "pank", "laen", "krediit", "investeering", "säästmine",
        "salary", "price", "expensive", "cheap", "budget", "loan",
    ],
    "sports_fitness": [
        "jalgpall", "NBA", "fitness", "maraton", "gym", "training",
        "jooks", "sport", "treening", "võistlus", "UEFA", "workout",
    ],
    "casual_slang": [
        "lmao", "btw", "tbh", "ngl", "cringe", "vibe", "mood",
        "literally", "random", "nice", "cool", "based", "sus",
        "basically", "actually", "honestly", "seriously",
    ],
    "food_lifestyle": [
        "restoran", "kohvik", "recipe", "vegan", "burger", "pizza",
        "õlu", "cocktail", "brunch", "delivery", "Wolt", "Bolt",
        "söök", "toit", "retsept", "küpsetamine", "cooking",
    ],
    "travel": [
        "reisimine", "travel", "Tallinn", "lennuk", "flight", "vacation",
        "hotel", "airbnb", "tourist", "viisa", "passport", "trip",
    ],
    "health_mental": [
        "tervis", "arst", "haigla", "diagnoos", "ravi", "ravim",
        "vaimne", "depressioon", "ärevus", "stress", "burnout",
        "doctor", "health", "therapy", "mental", "anxiety",
    ],
    "housing_living": [
        "korter", "maja", "üür", "ost", "müük", "remont", "sisustus",
        "naaber", "ühistu", "elamispind", "rent", "apartment", "house",
    ],
    "education": [
        "õppimine", "eksam", "kraad", "bakalaureuse", "magistri",
        "kursus", "loeng", "professor", "study", "degree", "university",
    ],
}

# Tõmmatavate Reddit postituste kategooriad
REDDIT_CATEGORIES = ["hot", "new", "top", "rising"]

# Ajafiltrid 'top' ja 'controversial' kategooriate jaoks
TIME_FILTERS = ["day", "week", "month", "year", "all"]

# Väljundfail
OUTPUT_FILE = "eesti_corpus.xlsx"
