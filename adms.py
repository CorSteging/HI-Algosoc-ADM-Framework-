import json
import re
import random
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "data" / "applicant_cvs.json"
FIRST_NAMES_MALE = [
    "Alex",
    "Ben",
    "Carlos",
    "David",
    "Ethan",
    "Felix",
    "George",
    "Hugo",
    "Ivan",
    "Jamal",
]
FIRST_NAMES_FEMALE = [
    "Anna",
    "Bianca",
    "Clara",
    "Diana",
    "Elena",
    "Fatima",
    "Grace",
    "Hana",
    "Isabel",
    "Julia",
]


def parse_cv(cv_text: str) -> dict:
    match = re.search(
        r"is a (\d+)-year-old .*? from ([A-Za-z ]+?) with a final score of ([0-9]+(?:\.[0-9]+)?)",
        cv_text,
    )
    if not match:
        return {
            "age": None,
            "country": None,
            "final_score": None,
        }

    age_str, country, score_str = match.groups()
    return {
        "age": int(age_str),
        "country": country,
        "final_score": float(score_str),
    }


def load_applicants(path: Path = DATA_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    applicants = []
    for entry in raw:
        extracted = parse_cv(entry.get("cv", ""))
        applicants.append(
            {
                "name": entry.get("name"),
                "age": extracted["age"],
                "country": extracted["country"],
                "final_score": extracted["final_score"],
                "gender": infer_gender(entry.get("name", "")),
            }
        )

    return applicants


def infer_gender(full_name: str) -> str | None:
    first_name = full_name.split(" ", 1)[0] if full_name else ""
    if first_name in FIRST_NAMES_MALE:
        return "male"
    if first_name in FIRST_NAMES_FEMALE:
        return "female"
    return None


def _rank_by_score(applicants: list[dict]) -> list[dict]:
    return sorted(
        applicants,
        key=lambda a: (a["final_score"] is None, -(a["final_score"] or 0.0)),
    )


def model_1(applicants: list[dict], k: int = 10) -> list[str]:
    ranked = _rank_by_score(applicants)
    return [a["name"] for a in ranked[:k]]


def model_2(applicants: list[dict], k: int = 10) -> list[str]:
    shuffled = list(applicants)
    random.shuffle(shuffled)
    return [a["name"] for a in shuffled[:k]]


def model_3(applicants: list[dict], k: int = 10) -> list[str]:
    filtered = [a for a in applicants if a["country"] == "Netherlands"]
    ranked = _rank_by_score(filtered)
    return [a["name"] for a in ranked[:k]]


def model_4(applicants: list[dict], k: int = 10) -> list[str]:
    ranked = _rank_by_score(applicants)
    target_male = k // 2
    target_female = k // 2
    if k % 2 == 1:
        target_extra = 1
    else:
        target_extra = 0

    selected = []
    male_count = 0
    female_count = 0

    for applicant in ranked:
        if len(selected) >= k:
            break
        gender = applicant.get("gender")
        if gender == "male" and male_count < target_male:
            selected.append(applicant)
            male_count += 1
            continue
        if gender == "female" and female_count < target_female:
            selected.append(applicant)
            female_count += 1
            continue

    if target_extra and len(selected) < k:
        for applicant in ranked:
            if len(selected) >= k:
                break
            if applicant in selected:
                continue
            gender = applicant.get("gender")
            if gender == "male" and male_count < target_male + 1:
                selected.append(applicant)
                male_count += 1
                continue
            if gender == "female" and female_count < target_female + 1:
                selected.append(applicant)
                female_count += 1
                continue

    if len(selected) < k:
        for applicant in ranked:
            if len(selected) >= k:
                break
            if applicant not in selected:
                selected.append(applicant)

    return [a["name"] for a in selected[:k]]


def model_5(applicants: list[dict], k: int = 10) -> list[str]:
    filtered = [a for a in applicants if a.get("gender") == "male"]
    ranked = _rank_by_score(filtered)
    return [a["name"] for a in ranked[:k]]


MODEL_REGISTRY = {
    "model_1": model_1,
    "model_2": model_2,
    "model_3": model_3,
    "model_4": model_4,
    "model_5": model_5,
}


def run_model(name: str, k: int = 10) -> list[str]:
    applicants = load_applicants()
    try:
        model = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'.") from exc
    return model(applicants, k=k)


def run_all_models(k: int = 10) -> dict[str, list[str]]:
    applicants = load_applicants()
    return {name: model(applicants, k=k) for name, model in MODEL_REGISTRY.items()}


def main() -> None:
    output = run_all_models()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
