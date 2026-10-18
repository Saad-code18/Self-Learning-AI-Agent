"""
seed_memories.py - fill the memory store with 1,000+ realistic facts across many users.

    python scripts/seed_memories.py                 # 1,100 facts for 50 users
    python scripts/seed_memories.py --count 2000 --users 80

Why?
    * To prove the self-hosted stack handles "1,000+ persisted memories across sessions".
    * To give the benchmark a realistic, non-empty Qdrant collection (search must
      still be fast and accurate when the store is big).

How?
    Facts are generated from templates with a fixed random seed (same data every
    run) and stored with `infer=False`, i.e. ONLY embeddings are computed - no LLM
    calls, so seeding is cheap (about 1,100 embedding requests).
    Seeded users are named `seed_user_001` ... so they never mix with your own
    memories.
"""

import _bootstrap  # noqa: F401

import argparse
import random
import time

from memory_agent.memory_store import MemoryStore
from memory_agent.settings import configure_logging, get_settings

# --------------------------------------------------------------------------
# Building blocks for synthetic facts (feel free to extend)
# --------------------------------------------------------------------------
NAMES = ["Alex", "Priya", "Marco", "Yuki", "Fatima", "Liam", "Sofia", "Noah", "Amara", "Diego", "Hana", "Omar",
         "Elena", "Kwame", "Mei", "Lucas", "Zara", "Ivan", "Chloe", "Tariq"]
CITIES = ["Toronto", "Berlin", "Austin", "Singapore", "Lisbon", "Nairobi", "Seoul", "Denver", "Dublin", "Sydney",
          "Mumbai", "Chicago", "Amsterdam", "São Paulo", "Vancouver", "Tokyo", "Madrid", "Cape Town"]
JOBS = ["data analyst", "product manager", "nurse", "backend engineer", "teacher", "UX designer", "accountant",
        "chef", "civil engineer", "marketing manager", "PhD student", "sales representative", "pharmacist"]
DIETS = ["vegetarian", "vegan", "pescatarian", "gluten-free", "lactose-intolerant", "on a low-carb diet",
         "allergic to peanuts", "allergic to shellfish"]
CUISINES = ["Thai", "Italian", "Mexican", "Japanese", "Ethiopian", "Indian", "Korean", "Greek", "Lebanese"]
HOBBIES = ["trail running", "chess", "pottery", "bouldering", "photography", "baking sourdough", "salsa dancing",
           "gardening", "cycling", "playing the piano", "swimming", "yoga", "board games", "birdwatching"]
PETS = [("dog", "Mochi"), ("cat", "Luna"), ("dog", "Biscuit"), ("parrot", "Kiwi"), ("cat", "Pepper"),
        ("rabbit", "Clover"), ("dog", "Rex"), ("hamster", "Peanut")]
GOALS = ["run a half marathon in May", "learn Spanish this year", "finish a master's thesis by December",
         "buy a house next year", "visit Japan in autumn", "switch to a machine-learning role",
         "read 30 books this year", "get a scuba diving license", "launch a side project"]
TOOLS = ["Python", "Excel", "Figma", "Notion", "Jira", "PostgreSQL", "React", "Tableau", "Docker"]
LANGUAGES = ["Spanish", "French", "Hindi", "Mandarin", "German", "Portuguese", "Arabic", "Korean"]
STYLES = ["short answers with bullet points", "detailed step-by-step explanations", "a casual, friendly tone",
          "formal language", "examples before theory"]
MUSIC = ["jazz", "lo-fi hip hop", "classical piano", "K-pop", "indie rock", "reggaeton", "techno"]
FAMILY = ["two kids", "a daughter named Ava", "a son named Leo", "twin boys", "a younger sister", "an older brother"]
DEVICES = ["an iPhone", "a Pixel phone", "a Windows laptop", "a MacBook", "a Linux desktop"]
DRINKS = ["oat-milk lattes", "black coffee", "green tea", "matcha", "cold brew"]
SPORTS = ["Arsenal", "the Lakers", "Real Madrid", "the Toronto Raptors", "the Yankees", "Bayern Munich"]
TIMES = ["early mornings", "late evenings", "weekends"]


def facts_for_user(rng: random.Random, name: str) -> list[str]:
    """Create ~22 distinct facts for one imaginary user."""
    pet_kind, pet_name = rng.choice(PETS)
    facts = [
        f"User's name is {name}.",
        f"User lives in {rng.choice(CITIES)}.",
        f"User works as a {rng.choice(JOBS)}.",
        f"User is {rng.choice(DIETS)}.",
        f"User loves {rng.choice(CUISINES)} food.",
        f"User enjoys {rng.choice(HOBBIES)} in their free time.",
        f"User has a {pet_kind} named {pet_name}.",
        f"User wants to {rng.choice(GOALS)}.",
        f"User uses {rng.choice(TOOLS)} daily at work.",
        f"User is learning {rng.choice(LANGUAGES)}.",
        f"User prefers {rng.choice(STYLES)}.",
        f"User listens to {rng.choice(MUSIC)} while working.",
        f"User has {rng.choice(FAMILY)}.",
        f"User's birthday is on {rng.choice(['March', 'June', 'September', 'November', 'January'])} {rng.randint(1, 28)}.",
        f"User owns {rng.choice(DEVICES)}.",
        f"User drinks {rng.choice(DRINKS)} every morning.",
        f"User supports {rng.choice(SPORTS)}.",
        f"User prefers to exercise in the {rng.choice(TIMES)}.",
        f"User dislikes {rng.choice(['horror movies', 'cilantro', 'loud restaurants', 'spicy food', 'cold weather'])}.",
        f"User is planning a trip to {rng.choice(CITIES)} in {rng.choice(['spring', 'summer', 'autumn', 'winter'])}.",
        f"User commutes by {rng.choice(['bike', 'train', 'car', 'bus'])}.",
        f"User's favorite movie genre is {rng.choice(['sci-fi', 'documentaries', 'romantic comedies', 'thrillers'])}.",
    ]
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1100, help="how many facts to insert in total (default 1100)")
    parser.add_argument("--users", type=int, default=50, help="spread the facts over this many users (default 50)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (same seed = same facts)")
    args = parser.parse_args()

    settings = get_settings()
    settings.validate()
    configure_logging(settings)
    store = MemoryStore(settings)
    if not store.ping():
        raise SystemExit("Qdrant is not reachable. Start it with: docker compose up -d qdrant")

    rng = random.Random(args.seed)
    per_user = max(1, -(-args.count // args.users))  # ceiling division
    print(f"Seeding {args.count} facts over {args.users} users (~{per_user} each) into '{settings.qdrant_collection}' ...")
    before = store.count()

    started = time.perf_counter()
    inserted = 0
    for i in range(args.users):
        if inserted >= args.count:
            break
        user_id = f"seed_user_{i + 1:03d}"
        name = rng.choice(NAMES)
        facts = facts_for_user(rng, name)
        # Add a few extra variations if we need more than 22 facts per user.
        while len(facts) < per_user:
            facts.append(f"User visited {rng.choice(CITIES)} in {rng.randint(2015, 2025)}.")
        facts = facts[: min(per_user, args.count - inserted)]

        store.add_facts(facts, user_id=user_id, metadata={"source": "seed"})
        inserted += len(facts)
        elapsed = time.perf_counter() - started
        print(f"  {user_id}: +{len(facts)}  (total {inserted}/{args.count}, {elapsed:.0f}s)", flush=True)

    total = store.count()
    print(f"\nDone: inserted {inserted} facts in {time.perf_counter() - started:.1f}s. "
          f"Collection now holds {total} memories (was {before}).")


if __name__ == "__main__":
    main()
