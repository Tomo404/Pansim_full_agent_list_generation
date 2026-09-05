from pathlib import Path
import csv
import heapq
import random
import re
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

CURRENT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/64_agents_with_activity_education_activity_calibrated.csv"
)

FOREIGN_EMPLOYED_HIER_LONG_CSV = Path(
    "../outputs/agent_attribute_generation/foreign_domestic_workplace_plan/84_foreign_employed_hier_long.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/foreign_workplace_assignment"
)

AGENTS_WITH_FOREIGN_DOMESTIC_STATUS_CSV = (
    OUTPUT_FOLDER / "90_agents_with_foreign_domestic_workplace_status.csv"
)

FOREIGN_ASSIGNMENT_PLAN_CSV = (
    OUTPUT_FOLDER / "91_foreign_assignment_plan.csv"
)

FOREIGN_ASSIGNMENT_VALIDATION_CSV = (
    OUTPUT_FOLDER / "92_foreign_assignment_validation.csv"
)

FOREIGN_ASSIGNMENT_SUMMARY_CSV = (
    OUTPUT_FOLDER / "93_foreign_assignment_summary.csv"
)

FOREIGN_ASSIGNMENT_METHOD_NOTES_CSV = (
    OUTPUT_FOLDER / "94_foreign_assignment_method_notes.csv"
)

RANDOM_SEED = 42

# Teszthez pl. 100_000.
# Teljes futtatáshoz None.
MAX_AGENTS_TO_PROCESS = None

EMPLOYED_ACTIVITY = "Foglalkoztatott"


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def normalize_text(raw_value) -> str:
    if raw_value is None:
        return ""

    try:
        if pd.isna(raw_value):
            return ""
    except TypeError:
        pass

    text = str(raw_value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def convert_count_to_integer(raw_value) -> int:
    if raw_value is None:
        return 0

    try:
        if pd.isna(raw_value):
            return 0
    except TypeError:
        pass

    text = str(raw_value).strip()

    if text == "":
        return 0

    if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")

    return int(round(float(text)))


def normalize_county_name(raw_county_name: str) -> str:
    county_name = normalize_text(raw_county_name)

    if county_name.endswith(" vármegye"):
        county_name = county_name.replace(" vármegye", "").strip()

    manual = {
        "Gyor-Moson-Sopron": "Győr-Moson-Sopron",
        "Győr-Moson-Sopron": "Győr-Moson-Sopron",
        "fováros": "Budapest",
        "Fováros": "Budapest",
        "főváros": "Budapest",
        "Főváros": "Budapest",
        "Budapest": "Budapest",
    }

    return manual.get(county_name, county_name)


def normalize_settlement_type(raw_settlement_type: str) -> str:
    settlement_type = normalize_text(raw_settlement_type)

    manual = {
        "Fováros": "Főváros",
        "fováros": "Főváros",
        "főváros": "Főváros",
        "Főváros": "Főváros",
    }

    return manual.get(settlement_type, settlement_type)


def normalize_sex_label(raw_sex: str) -> str:
    sex = normalize_text(raw_sex).lower()

    if sex in ["férfi", "ferfi", "male"]:
        return "male"

    if sex in ["nő", "no", "female"]:
        return "female"

    return sex


def normalize_age_group(raw_age_group: str) -> str:
    text = normalize_text(raw_age_group)
    text = text.replace("-", "–")
    return text


def normalize_education(raw_education: str) -> str:
    return normalize_text(raw_education)


def normalize_activity(raw_activity: str) -> str:
    activity = normalize_text(raw_activity)

    manual = {
        "Foglalkoztatott": "Foglalkoztatott",
        "employed": "Foglalkoztatott",
        "Munkanélküli": "Munkanélküli",
        "unemployed": "Munkanélküli",
        "Ellátásban részesülő inaktív": "Ellátásban részesülő inaktív",
        "inactive_benefit": "Ellátásban részesülő inaktív",
        "Eltartott": "Eltartott",
        "dependent": "Eltartott",
        "15 évesnél fiatalabb": "15 évesnél fiatalabb",
    }

    return manual.get(activity, activity)


def map_exact_age_to_worker_age_group(exact_age: int) -> str:
    age = int(exact_age)

    if age < 15:
        return "15 évesnél fiatalabb"

    if age < 20:
        return "15–19 éves"

    if age < 25:
        return "20–24 éves"

    if age < 30:
        return "25–29 éves"

    if age < 35:
        return "30–34 éves"

    if age < 40:
        return "35–39 éves"

    if age < 45:
        return "40–44 éves"

    if age < 50:
        return "45–49 éves"

    if age < 55:
        return "50–54 éves"

    if age < 60:
        return "55–59 éves"

    if age < 65:
        return "60–64 éves"

    if age < 70:
        return "65–69 éves"

    return "70 éves és idősebb"


def make_foreign_match_key(
    county_name: str,
    settlement_type: str,
    sex: str,
    age_group: str,
    education: str,
) -> tuple[str, str, str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_sex_label(sex),
        normalize_age_group(age_group),
        normalize_education(education),
    )


def make_agent_foreign_match_key(row: dict) -> tuple[str, str, str, str, str]:
    exact_age = convert_count_to_integer(row["exact_age"])

    return make_foreign_match_key(
        county_name=row["county_name"],
        settlement_type=row["settlement_type"],
        sex=row["sex"],
        age_group=map_exact_age_to_worker_age_group(exact_age),
        education=row["education_level_calibrated"],
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# 3. FOREIGN TARGET BETÖLTÉSE
# ============================================================

def load_foreign_targets(foreign_hier_long_csv: Path) -> pd.DataFrame:
    foreign_long = pd.read_csv(foreign_hier_long_csv)

    foreign_long["county_name"] = foreign_long["county_name"].apply(normalize_county_name)
    foreign_long["settlement_type"] = foreign_long["settlement_type"].apply(normalize_settlement_type)
    foreign_long["sex"] = foreign_long["sex"].apply(normalize_sex_label)
    foreign_long["age_group"] = foreign_long["age_group"].apply(normalize_age_group)
    foreign_long["education"] = foreign_long["education"].apply(normalize_education)
    foreign_long["foreign_employed_count"] = foreign_long["foreign_employed_count"].apply(
        convert_count_to_integer
    )

    targets = (
        foreign_long
        .groupby(
            [
                "county_name",
                "settlement_type",
                "sex",
                "age_group",
                "education",
            ],
            dropna=False,
        )["foreign_employed_count"]
        .sum()
        .reset_index()
        .rename(columns={"foreign_employed_count": "foreign_target_count"})
    )

    return targets


def build_foreign_target_lookup(foreign_targets: pd.DataFrame) -> dict:
    lookup = {}

    for _, row in foreign_targets.iterrows():
        key = make_foreign_match_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            sex=row["sex"],
            age_group=row["age_group"],
            education=row["education"],
        )

        lookup[key] = convert_count_to_integer(row["foreign_target_count"])

    return lookup


# ============================================================
# 4. FOREIGN KIJELÖLENDŐ AGENTEK KIVÁLASZTÁSA
# ============================================================

def create_foreign_candidate_score(
    row: dict,
    random_generator: random.Random,
) -> float:
    """
    Ugyanazon demográfiai kulcson belül választunk agenteket.

    Itt nem akarunk erős torzítást bevinni, mert maga a kulcs már tartalmazza:
    county, settlement_type, sex, age_group, education.

    A random score csak azt szolgálja, hogy determinisztikusan,
    de ne sorbarendezési artefaktum alapján válasszunk.
    """
    return random_generator.random()


def select_foreign_agent_rows(
    current_agents_csv: Path,
    foreign_target_lookup: dict,
) -> dict:
    """
    Első pass:
    kiválasztja, mely sorok legyenek foreign_workplace státuszúak.

    Exact kulcs:
    county × settlement_type × sex × age_group × education

    Ha egy exact kulcsnál nincs elég current employed agent,
    akkor az adott target maradéka fallback kijelölésre kerül.
    """
    random_generator = random.Random(RANDOM_SEED)

    heaps_by_key = {
        key: []
        for key in foreign_target_lookup
        if foreign_target_lookup[key] > 0
    }

    available_employed_count_by_key = defaultdict(int)
    total_agents_seen = 0
    employed_agents_seen = 0

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row_number, row in enumerate(reader, start=1):
            if MAX_AGENTS_TO_PROCESS is not None and total_agents_seen >= MAX_AGENTS_TO_PROCESS:
                break

            activity = normalize_activity(row["economic_activity_status_calibrated"])

            if activity == EMPLOYED_ACTIVITY:
                employed_agents_seen += 1

                key = make_agent_foreign_match_key(row)
                available_employed_count_by_key[key] += 1

                if key in foreign_target_lookup:
                    target_count = foreign_target_lookup[key]

                    if target_count > 0:
                        score = create_foreign_candidate_score(
                            row=row,
                            random_generator=random_generator,
                        )

                        heap = heaps_by_key[key]
                        candidate = (score, row_number)

                        if len(heap) < target_count:
                            heapq.heappush(heap, candidate)
                        else:
                            if candidate[0] > heap[0][0]:
                                heapq.heapreplace(heap, candidate)

            total_agents_seen += 1

            if total_agents_seen % 1_000_000 == 0:
                print(f"Foreign kiválasztási pass agentek: {total_agents_seen:,}")

    selected_row_numbers = set()
    exact_assigned_by_key = defaultdict(int)
    fallback_needed_by_key = defaultdict(int)

    for key, target_count in foreign_target_lookup.items():
        selected_for_key = heaps_by_key.get(key, [])

        exact_count = len(selected_for_key)

        for _, row_number in selected_for_key:
            selected_row_numbers.add(row_number)

        exact_assigned_by_key[key] = exact_count

        if exact_count < target_count:
            fallback_needed_by_key[key] = target_count - exact_count

    return {
        "selected_row_numbers": selected_row_numbers,
        "exact_assigned_by_key": dict(exact_assigned_by_key),
        "fallback_needed_by_key": dict(fallback_needed_by_key),
        "available_employed_count_by_key": dict(available_employed_count_by_key),
        "total_agents_seen": total_agents_seen,
        "employed_agents_seen": employed_agents_seen,
    }


# ============================================================
# 5. FALLBACK FOREIGN KIJELÖLÉS
# ============================================================

def create_fallback_target_list(fallback_needed_by_key: dict) -> list[tuple]:
    fallback_targets = []

    for key, needed_count in fallback_needed_by_key.items():
        for _ in range(needed_count):
            fallback_targets.append(key)

    return fallback_targets


def create_fallback_candidate_score(
    row: dict,
    target_key: tuple,
    random_generator: random.Random,
) -> float:
    """
    Fallbacknél hasonlósági pontszám.

    target_key:
    county, settlement_type, sex, age_group, education

    Pontozás:
    - ugyanaz a megye: nagy súly
    - ugyanaz a településtípus: közepes súly
    - ugyanaz a nem: közepes súly
    - ugyanaz a korcsoport: közepes súly
    - ugyanaz a végzettség: közepes súly

    Mivel várhatóan csak néhány fő fallback kell, ez elég.
    """
    candidate_key = make_agent_foreign_match_key(row)

    target_county, target_type, target_sex, target_age, target_education = target_key
    cand_county, cand_type, cand_sex, cand_age, cand_education = candidate_key

    score = 0.0

    if cand_county == target_county:
        score += 1000.0

    if cand_type == target_type:
        score += 200.0

    if cand_sex == target_sex:
        score += 200.0

    if cand_age == target_age:
        score += 200.0

    if cand_education == target_education:
        score += 200.0

    score += random_generator.random()

    return score


def select_fallback_foreign_rows(
    current_agents_csv: Path,
    already_selected_rows: set[int],
    fallback_needed_by_key: dict,
) -> dict:
    fallback_targets = create_fallback_target_list(fallback_needed_by_key)

    if len(fallback_targets) == 0:
        return {
            "fallback_selected_rows": {},
            "fallback_assigned_count": 0,
        }

    random_generator = random.Random(RANDOM_SEED)

    fallback_selected_rows = {}
    used_rows = set(already_selected_rows)

    # Kevés fallbackre számítunk, ezért egyszerű, több célponton végigmenő kiválasztás.
    for fallback_index, target_key in enumerate(fallback_targets, start=1):
        best_candidate = None
        best_score = -1.0

        total_seen = 0

        with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            for row_number, row in enumerate(reader, start=1):
                if MAX_AGENTS_TO_PROCESS is not None and total_seen >= MAX_AGENTS_TO_PROCESS:
                    break

                if row_number in used_rows:
                    total_seen += 1
                    continue

                activity = normalize_activity(row["economic_activity_status_calibrated"])

                if activity != EMPLOYED_ACTIVITY:
                    total_seen += 1
                    continue

                score = create_fallback_candidate_score(
                    row=row,
                    target_key=target_key,
                    random_generator=random_generator,
                )

                if score > best_score:
                    best_score = score
                    best_candidate = row_number

                total_seen += 1

        if best_candidate is None:
            raise ValueError(
                f"Nem találtam fallback foreign candidate-et ehhez a targethez: {target_key}"
            )

        fallback_selected_rows[best_candidate] = target_key
        used_rows.add(best_candidate)

        print(
            f"Fallback foreign kijelölés {fallback_index}/{len(fallback_targets)}: "
            f"row={best_candidate}, target={target_key}, score={best_score}"
        )

    return {
        "fallback_selected_rows": fallback_selected_rows,
        "fallback_assigned_count": len(fallback_selected_rows),
    }


# ============================================================
# 6. OUTPUT AGENTFÁJL ÍRÁSA
# ============================================================

def write_agents_with_foreign_domestic_status(
    current_agents_csv: Path,
    output_agents_csv: Path,
    exact_selected_rows: set[int],
    fallback_selected_rows: dict,
) -> dict:
    total_written = 0
    employed_total = 0
    foreign_assigned = 0
    foreign_exact_assigned = 0
    foreign_fallback_assigned = 0
    domestic_assigned = 0
    not_employed_count = 0
    missing_workplace_assignment_count = 0

    status_counts = defaultdict(int)

    all_foreign_rows = set(exact_selected_rows) | set(fallback_selected_rows.keys())

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)

        input_fieldnames = list(reader.fieldnames)

        new_columns = [
            "employment_layer_status",
            "workplace_assignment_type",
            "workplace_assignment_source",
            "foreign_workplace_assignment_method",
            "workplace_id",
            "workplace_country",
            "workplace_county",
            "workplace_settlement",
            "workplace_settlement_type",
            "workplace_teaor_code",
            "workplace_size",
            "previous_worker_record_id",
        ]

        output_fieldnames = input_fieldnames + [
            column
            for column in new_columns
            if column not in input_fieldnames
        ]

        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for row_number, row in enumerate(reader, start=1):
            if MAX_AGENTS_TO_PROCESS is not None and total_written >= MAX_AGENTS_TO_PROCESS:
                break

            activity = normalize_activity(row["economic_activity_status_calibrated"])

            if activity == EMPLOYED_ACTIVITY:
                employed_total += 1

                if row_number in all_foreign_rows:
                    row["employment_layer_status"] = "employed_with_foreign_workplace"
                    row["workplace_assignment_type"] = "foreign_workplace"
                    row["workplace_assignment_source"] = "hier_kulfold_foglalkoztatott"

                    if row_number in fallback_selected_rows:
                        row["foreign_workplace_assignment_method"] = "fallback"
                        foreign_fallback_assigned += 1
                    else:
                        row["foreign_workplace_assignment_method"] = "exact_key"
                        foreign_exact_assigned += 1

                    row["workplace_id"] = "FOREIGN_WORKPLACE"
                    row["workplace_country"] = "FOREIGN"
                    row["workplace_county"] = "FOREIGN"
                    row["workplace_settlement"] = "FOREIGN"
                    row["workplace_settlement_type"] = "FOREIGN"
                    row["workplace_teaor_code"] = "FOREIGN_OR_UNKNOWN_TEAOR"
                    row["workplace_size"] = 0
                    row["previous_worker_record_id"] = "NO_PREVIOUS_WORKER_RECORD"

                    foreign_assigned += 1

                else:
                    row["employment_layer_status"] = "employed_pending_domestic_workplace"
                    row["workplace_assignment_type"] = "domestic_workplace_pending"
                    row["workplace_assignment_source"] = "pending_previous_worker_record_matching"
                    row["foreign_workplace_assignment_method"] = "not_foreign"

                    row["workplace_id"] = "PENDING_DOMESTIC_WORKPLACE"
                    row["workplace_country"] = "HUNGARY"
                    row["workplace_county"] = "PENDING_DOMESTIC_WORKPLACE"
                    row["workplace_settlement"] = "PENDING_DOMESTIC_WORKPLACE"
                    row["workplace_settlement_type"] = "PENDING_DOMESTIC_WORKPLACE"
                    row["workplace_teaor_code"] = "PENDING_DOMESTIC_TEAOR"
                    row["workplace_size"] = 0
                    row["previous_worker_record_id"] = "PENDING_PREVIOUS_WORKER_RECORD"

                    domestic_assigned += 1

            else:
                row["employment_layer_status"] = "not_employed"
                row["workplace_assignment_type"] = "not_applicable"
                row["workplace_assignment_source"] = "not_employed_agent"
                row["foreign_workplace_assignment_method"] = "not_applicable"

                row["workplace_id"] = "NO_WORKPLACE"
                row["workplace_country"] = "NOT_APPLICABLE"
                row["workplace_county"] = "NO_WORKPLACE"
                row["workplace_settlement"] = "NO_WORKPLACE"
                row["workplace_settlement_type"] = "NO_WORKPLACE"
                row["workplace_teaor_code"] = "NO_TEAOR"
                row["workplace_size"] = 0
                row["previous_worker_record_id"] = "NO_PREVIOUS_WORKER_RECORD"

                not_employed_count += 1

            status_counts[row["employment_layer_status"]] += 1

            if row["employment_layer_status"] == "missing_workplace_assignment":
                missing_workplace_assignment_count += 1

            writer.writerow(row)

            total_written += 1

            if total_written % 1_000_000 == 0:
                print(f"Kiírt agentek foreign/domestic státusszal: {total_written:,}")

    return {
        "total_written": total_written,
        "employed_total": employed_total,
        "foreign_assigned": foreign_assigned,
        "foreign_exact_assigned": foreign_exact_assigned,
        "foreign_fallback_assigned": foreign_fallback_assigned,
        "domestic_assigned": domestic_assigned,
        "not_employed_count": not_employed_count,
        "missing_workplace_assignment_count": missing_workplace_assignment_count,
        "status_counts": dict(status_counts),
    }


# ============================================================
# 7. VALIDÁCIÓS OUTPUTOK
# ============================================================

def create_foreign_assignment_plan(
    foreign_targets: pd.DataFrame,
    selection_result: dict,
) -> pd.DataFrame:
    rows = []

    exact_assigned_by_key = selection_result["exact_assigned_by_key"]
    fallback_needed_by_key = selection_result["fallback_needed_by_key"]
    available_by_key = selection_result["available_employed_count_by_key"]

    for _, row in foreign_targets.iterrows():
        key = make_foreign_match_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            sex=row["sex"],
            age_group=row["age_group"],
            education=row["education"],
        )

        target = convert_count_to_integer(row["foreign_target_count"])
        available = available_by_key.get(key, 0)
        exact_assigned = exact_assigned_by_key.get(key, 0)
        fallback_needed = fallback_needed_by_key.get(key, 0)

        rows.append({
            "county_name": key[0],
            "settlement_type": key[1],
            "sex": key[2],
            "age_group": key[3],
            "education": key[4],
            "foreign_target_count": target,
            "current_employed_available_count": available,
            "foreign_exact_assigned_count": exact_assigned,
            "foreign_fallback_needed_count": fallback_needed,
            "difference_exact_assigned_minus_target": exact_assigned - target,
        })

    plan = pd.DataFrame(rows)

    plan = plan.sort_values(
        [
            "foreign_fallback_needed_count",
            "foreign_target_count",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return plan


def create_assignment_validation(
    foreign_targets: pd.DataFrame,
    output_agents_csv: Path,
) -> pd.DataFrame:
    assigned_counts_by_key = defaultdict(int)

    total_rows = 0

    with output_agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_rows >= MAX_AGENTS_TO_PROCESS:
                break

            if row["workplace_assignment_type"] == "foreign_workplace":
                key = make_agent_foreign_match_key(row)
                assigned_counts_by_key[key] += 1

            total_rows += 1

    assigned_rows = []

    for key, count in assigned_counts_by_key.items():
        assigned_rows.append({
            "county_name": key[0],
            "settlement_type": key[1],
            "sex": key[2],
            "age_group": key[3],
            "education": key[4],
            "foreign_assigned_count": count,
        })

    assigned_df = pd.DataFrame(assigned_rows)

    if assigned_df.empty:
        assigned_df = pd.DataFrame(
            columns=[
                "county_name",
                "settlement_type",
                "sex",
                "age_group",
                "education",
                "foreign_assigned_count",
            ]
        )

    target_df = foreign_targets.copy()

    validation = target_df.merge(
        assigned_df,
        on=[
            "county_name",
            "settlement_type",
            "sex",
            "age_group",
            "education",
        ],
        how="outer",
    )

    validation["foreign_target_count"] = pd.to_numeric(
        validation["foreign_target_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["foreign_assigned_count"] = pd.to_numeric(
        validation["foreign_assigned_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_assigned_minus_target"] = (
        validation["foreign_assigned_count"] - validation["foreign_target_count"]
    )

    validation["absolute_difference"] = validation[
        "difference_assigned_minus_target"
    ].abs()

    validation = validation.sort_values(
        [
            "absolute_difference",
            "foreign_target_count",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return validation


def create_summary(
    foreign_targets: pd.DataFrame,
    selection_result: dict,
    fallback_result: dict,
    write_result: dict,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    foreign_target_total = foreign_targets["foreign_target_count"].sum()

    rows = [
        {
            "metric": "output_agents_total",
            "value": write_result["total_written"],
            "note": "Output agentek száma a 90-es fájlban.",
        },
        {
            "metric": "employed_total",
            "value": write_result["employed_total"],
            "note": "Foglalkoztatott agentek száma.",
        },
        {
            "metric": "foreign_workplace_target_total",
            "value": foreign_target_total,
            "note": "Foreign workplace target a hier_kulfold_foglalkoztatott tábla alapján.",
        },
        {
            "metric": "foreign_workplace_assigned_total",
            "value": write_result["foreign_assigned"],
            "note": "Foreign workplace státuszra kijelölt agentek száma.",
        },
        {
            "metric": "foreign_workplace_difference_assigned_minus_target",
            "value": write_result["foreign_assigned"] - foreign_target_total,
            "note": "Ideálisan 0.",
        },
        {
            "metric": "foreign_exact_assigned_total",
            "value": write_result["foreign_exact_assigned"],
            "note": "Pontos county × settlement_type × sex × age_group × education kulcson kijelölt foreign agentek.",
        },
        {
            "metric": "foreign_fallback_assigned_total",
            "value": write_result["foreign_fallback_assigned"],
            "note": "Fallback módszerrel kijelölt foreign agentek.",
        },
        {
            "metric": "domestic_workplace_pending_total",
            "value": write_result["domestic_assigned"],
            "note": "Belföldi workplace assignmentre váró foglalkoztatott agentek száma.",
        },
        {
            "metric": "not_employed_total",
            "value": write_result["not_employed_count"],
            "note": "Nem foglalkoztatott agentek száma.",
        },
        {
            "metric": "missing_workplace_assignment_count",
            "value": write_result["missing_workplace_assignment_count"],
            "note": "Ideálisan 0.",
        },
        {
            "metric": "validation_sum_absolute_difference",
            "value": validation["absolute_difference"].sum(),
            "note": "Foreign target vs assigned abszolút eltérések összege.",
        },
        {
            "metric": "validation_max_absolute_difference",
            "value": validation["absolute_difference"].max(),
            "note": "Legnagyobb foreign target vs assigned cellaeltérés.",
        },
        {
            "metric": "exact_selection_employed_seen",
            "value": selection_result["employed_agents_seen"],
            "note": "Első passban látott foglalkoztatott agentek száma.",
        },
        {
            "metric": "fallback_needed_total_before_fallback_assignment",
            "value": sum(selection_result["fallback_needed_by_key"].values()),
            "note": "Exact kulcson nem fedett foreign targetek száma fallback előtt.",
        },
        {
            "metric": "fallback_assigned_total",
            "value": fallback_result["fallback_assigned_count"],
            "note": "Fallbackként ténylegesen kijelölt foreign agentek száma.",
        },
    ]

    return pd.DataFrame(rows)


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script a 64-es agentfájlból kijelöli a foreign_workplace státuszú foglalkoztatott agenteket.",
        },
        {
            "order": 2,
            "note": "A foreign target a hier_kulfold_foglalkoztatott táblából előállított 84_foreign_employed_hier_long.csv alapján jön.",
        },
        {
            "order": 3,
            "note": "Elsődleges kijelölési kulcs: county × settlement_type × sex × 5 éves age_group × education.",
        },
        {
            "order": 4,
            "note": "Ahol pontos kulcson nincs elég foglalkoztatott agent, ott fallback kijelölés történik hasonlósági pontszám alapján.",
        },
        {
            "order": 5,
            "note": "A foreign_workplace agentek nem kapnak magyarországi workplace_id-t vagy TEÁOR-kódot; ezeket explicit FOREIGN jelöléssel látjuk el.",
        },
        {
            "order": 6,
            "note": "A domestic foglalkoztatott agentek ebben a lépésben még csak pending domestic workplace státuszt kapnak.",
        },
        {
            "order": 7,
            "note": "A nem foglalkoztatott agentek explicit NO_WORKPLACE / NO_TEAOR jelölést kapnak, nem üres cellát.",
        },
        {
            "order": 8,
            "note": "A következő lépésben csak a domestic_workplace_pending agentekhez kell régi agents_with_workplaces rekordot rendelni.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        FOREIGN_ASSIGNMENT_METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 8. FŐ FUTTATÁS
# ============================================================

def run_foreign_workplace_assignment() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Foreign targetek betöltése...")
    foreign_targets = load_foreign_targets(
        foreign_hier_long_csv=FOREIGN_EMPLOYED_HIER_LONG_CSV
    )

    foreign_target_lookup = build_foreign_target_lookup(
        foreign_targets=foreign_targets
    )

    print("Foreign exact agent sorok kijelölése...")
    selection_result = select_foreign_agent_rows(
        current_agents_csv=CURRENT_AGENTS_CSV,
        foreign_target_lookup=foreign_target_lookup,
    )

    print("Foreign fallback agent sorok kijelölése...")
    fallback_result = select_fallback_foreign_rows(
        current_agents_csv=CURRENT_AGENTS_CSV,
        already_selected_rows=selection_result["selected_row_numbers"],
        fallback_needed_by_key=selection_result["fallback_needed_by_key"],
    )

    all_exact_selected_rows = selection_result["selected_row_numbers"]
    fallback_selected_rows = fallback_result["fallback_selected_rows"]

    print("Agentfájl írása foreign/domestic workplace státusszal...")
    write_result = write_agents_with_foreign_domestic_status(
        current_agents_csv=CURRENT_AGENTS_CSV,
        output_agents_csv=AGENTS_WITH_FOREIGN_DOMESTIC_STATUS_CSV,
        exact_selected_rows=all_exact_selected_rows,
        fallback_selected_rows=fallback_selected_rows,
    )

    print("Foreign assignment plan készítése...")
    assignment_plan = create_foreign_assignment_plan(
        foreign_targets=foreign_targets,
        selection_result=selection_result,
    )

    print("Foreign assignment validation készítése...")
    validation = create_assignment_validation(
        foreign_targets=foreign_targets,
        output_agents_csv=AGENTS_WITH_FOREIGN_DOMESTIC_STATUS_CSV,
    )

    print("Summary készítése...")
    summary = create_summary(
        foreign_targets=foreign_targets,
        selection_result=selection_result,
        fallback_result=fallback_result,
        write_result=write_result,
        validation=validation,
    )

    print("Method notes írása...")
    write_method_notes()

    print("Outputok mentése...")

    assignment_plan.to_csv(
        FOREIGN_ASSIGNMENT_PLAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    validation.to_csv(
        FOREIGN_ASSIGNMENT_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        FOREIGN_ASSIGNMENT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Output agentek: {write_result['total_written']:,}")
    print(f"Foglalkoztatott agentek: {write_result['employed_total']:,}")
    print(f"Foreign workplace target: {foreign_targets['foreign_target_count'].sum():,}")
    print(f"Foreign assigned: {write_result['foreign_assigned']:,}")
    print(f"Domestic pending: {write_result['domestic_assigned']:,}")
    print(f"Nem foglalkoztatott: {write_result['not_employed_count']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_foreign_workplace_assignment()