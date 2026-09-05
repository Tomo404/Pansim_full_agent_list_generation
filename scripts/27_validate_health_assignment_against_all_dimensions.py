from pathlib import Path
import csv
import math
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

AGENTS_WITH_HEALTH_STATUS_CSV = Path(
    "../outputs/agent_attribute_generation/health_status_assignment/"
    "130_agents_with_age_based_health_status.csv"
)

HEALTH_LONG_CSV = Path(
    "../outputs/agent_attribute_generation/health_status_assignment/"
    "121_health_status_hier_long.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/health_status_assignment"
)

VALIDATION_BY_DIMENSION_CSV = OUTPUT_FOLDER / "140_health_assignment_validation_by_all_dimensions.csv"
VALIDATION_SUMMARY_CSV = OUTPUT_FOLDER / "141_health_assignment_validation_by_all_dimensions_summary.csv"
DIMENSION_AGENT_COUNTS_CSV = OUTPUT_FOLDER / "142_health_validation_agent_counts_by_dimension.csv"
METHOD_NOTES_CSV = OUTPUT_FOLDER / "143_health_assignment_all_dimension_validation_method_notes.csv"

MAX_ROWS_TO_PROCESS = None

UNDER_5_NOT_IN_SOURCE = "UNDER_5_NOT_IN_SOURCE"

HEALTH_QUESTION_TO_AGENT_COLUMN = {
    "disability_or_severe_limitation": "health_disability_status",
    "chronic_disease": "health_chronic_disease_status",
    "limitation_level": "health_limitation_status",
}


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def normalize_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value).strip()


def convert_count_to_integer(value) -> int:
    text = normalize_text(value)

    if text == "":
        return 0

    return int(round(float(text)))


def normalize_county_name(value) -> str:
    county_name = normalize_text(value)

    if county_name.endswith(" vármegye"):
        county_name = county_name.replace(" vármegye", "").strip()

    manual = {
        "Gyor-Moson-Sopron": "Győr-Moson-Sopron",
        "Győr-Moson-Sopron": "Győr-Moson-Sopron",
        "Főváros": "Budapest",
        "főváros": "Budapest",
        "Fováros": "Budapest",
        "fováros": "Budapest",
        "Budapest": "Budapest",
    }

    return manual.get(county_name, county_name)


def normalize_settlement_type(value) -> str:
    settlement_type = normalize_text(value)

    manual = {
        "Főváros": "Főváros",
        "főváros": "Főváros",
        "Fováros": "Főváros",
        "fováros": "Főváros",
        "Megyei jogú város(ok)": "Megyei jogú város(ok)",
        "Egyéb város(ok)": "Egyéb város(ok)",
        "Község(ek)": "Község(ek)",
    }

    return manual.get(settlement_type, settlement_type)


def normalize_sex_to_health_dimension_category(value) -> str:
    sex = normalize_text(value).lower()

    if sex in ["male", "férfi", "ferfi"]:
        return "Férfi"

    if sex in ["female", "nő", "no"]:
        return "Nő"

    return normalize_text(value)


def map_exact_age_to_health_age_group(exact_age: int) -> str:
    age = int(exact_age)

    if age < 5:
        return UNDER_5_NOT_IN_SOURCE

    if age <= 14:
        return "5–14 éves"

    if age <= 24:
        return "15–24 éves"

    if age <= 34:
        return "25–34 éves"

    if age <= 44:
        return "35–44 éves"

    if age <= 54:
        return "45–54 éves"

    if age <= 64:
        return "55–64 éves"

    if age <= 74:
        return "65–74 éves"

    if age <= 84:
        return "75–84 éves"

    return "85 éves és idősebb"


def map_agent_to_dimension_category(row: dict, dimension_group: str) -> str:
    exact_age = convert_count_to_integer(row["exact_age"])

    if dimension_group == "age_group":
        return map_exact_age_to_health_age_group(exact_age)

    if dimension_group == "sex":
        return normalize_sex_to_health_dimension_category(row["sex"])

    if dimension_group == "education":
        if exact_age < 15:
            return "15 évesnél fiatalabb személy"

        return normalize_text(row["education_level_calibrated"])

    if dimension_group == "economic_activity":
        if exact_age < 15:
            return "15 évesnél fiatalabb személy"

        return normalize_text(row["economic_activity_status_calibrated"])

    raise ValueError(f"Ismeretlen dimension_group: {dimension_group}")


def make_dimension_key(
    county_name: str,
    settlement_type: str,
    dimension_group: str,
    dimension_category: str,
) -> tuple[str, str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_text(dimension_group),
        normalize_text(dimension_category),
    )


def make_full_validation_key(
    county_name: str,
    settlement_type: str,
    dimension_group: str,
    dimension_category: str,
    health_question: str,
    health_response_category: str,
) -> tuple[str, str, str, str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_text(dimension_group),
        normalize_text(dimension_category),
        normalize_text(health_question),
        normalize_text(health_response_category),
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def largest_remainder_allocation(
    response_weights: dict[str, int],
    target_total: int,
) -> dict[str, int]:
    source_total = sum(response_weights.values())

    if target_total == 0:
        return {
            response_category: 0
            for response_category in response_weights
        }

    if source_total == 0:
        raise ValueError(
            "0 source_total mellett nem lehet nem nulla target_totalra skálázni."
        )

    raw_allocations = []

    for response_category, source_count in response_weights.items():
        raw_value = source_count * target_total / source_total
        floor_value = math.floor(raw_value)
        remainder = raw_value - floor_value

        raw_allocations.append({
            "response_category": response_category,
            "floor_value": floor_value,
            "remainder": remainder,
        })

    allocated_total = sum(item["floor_value"] for item in raw_allocations)
    remaining = target_total - allocated_total

    raw_allocations = sorted(
        raw_allocations,
        key=lambda item: item["remainder"],
        reverse=True,
    )

    result = {}

    for index, item in enumerate(raw_allocations):
        extra = 1 if index < remaining else 0
        result[item["response_category"]] = item["floor_value"] + extra

    return result


# ============================================================
# 3. HEALTH SOURCE TARGETEK
# ============================================================

def load_health_long(health_long_csv: Path) -> pd.DataFrame:
    health_long = pd.read_csv(
        health_long_csv,
        keep_default_na=False,
    )

    required_columns = [
        "county_name",
        "settlement_type",
        "dimension_group",
        "dimension_category",
        "health_question",
        "health_response_category",
        "health_count",
    ]

    for column_name in required_columns:
        if column_name not in health_long.columns:
            raise ValueError(
                f"Hiányzó oszlop a health long fájlból: {column_name}"
            )

    health_long["county_name"] = health_long["county_name"].apply(normalize_county_name)
    health_long["settlement_type"] = health_long["settlement_type"].apply(normalize_settlement_type)
    health_long["dimension_group"] = health_long["dimension_group"].apply(normalize_text)
    health_long["dimension_category"] = health_long["dimension_category"].apply(normalize_text)
    health_long["health_question"] = health_long["health_question"].apply(normalize_text)
    health_long["health_response_category"] = health_long["health_response_category"].apply(normalize_text)
    health_long["health_count"] = health_long["health_count"].apply(convert_count_to_integer)

    allowed_dimension_groups = {
        "age_group",
        "sex",
        "education",
        "economic_activity",
    }

    health_long = health_long[
        health_long["dimension_group"].isin(allowed_dimension_groups)
    ].copy()

    return health_long


def build_health_response_weights(health_long: pd.DataFrame) -> dict:
    """
    Source response eloszlás:
    county × settlement_type × dimension_group × dimension_category × health_question
    szerint.

    A 15 évesnél fiatalabb személy duplikáció itt nem keveredik össze,
    mert a dimension_group része a kulcsnak.
    """
    weights = defaultdict(lambda: defaultdict(int))

    for _, row in health_long.iterrows():
        dimension_key = make_dimension_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            dimension_group=row["dimension_group"],
            dimension_category=row["dimension_category"],
        )

        health_question = normalize_text(row["health_question"])
        response_category = normalize_text(row["health_response_category"])
        health_count = convert_count_to_integer(row["health_count"])

        weights[(dimension_key, health_question)][response_category] += health_count

    return {
        key: dict(counter)
        for key, counter in weights.items()
    }


# ============================================================
# 4. AGENTEK SZÁMLÁLÁSA
# ============================================================

def count_agents_and_assigned_health_by_dimensions(
    agents_csv: Path,
) -> dict:
    total_rows = 0
    age_5plus_rows = 0
    under_5_rows = 0

    agent_counts_by_dimension_key = defaultdict(int)
    assigned_counts_by_full_key = defaultdict(int)

    with agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        required_columns = [
            "county_name",
            "settlement_type",
            "sex",
            "exact_age",
            "education_level_calibrated",
            "economic_activity_status_calibrated",
        ]

        for health_question, column_name in HEALTH_QUESTION_TO_AGENT_COLUMN.items():
            required_columns.append(column_name)

        for column_name in required_columns:
            if column_name not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop az agent fájlból: {column_name}"
                )

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])

            if exact_age < 5:
                under_5_rows += 1
                total_rows += 1
                continue

            age_5plus_rows += 1

            for dimension_group in ["age_group", "sex", "education", "economic_activity"]:
                dimension_category = map_agent_to_dimension_category(
                    row=row,
                    dimension_group=dimension_group,
                )

                dimension_key = make_dimension_key(
                    county_name=row["county_name"],
                    settlement_type=row["settlement_type"],
                    dimension_group=dimension_group,
                    dimension_category=dimension_category,
                )

                agent_counts_by_dimension_key[dimension_key] += 1

                for health_question, agent_health_column in HEALTH_QUESTION_TO_AGENT_COLUMN.items():
                    response_category = normalize_text(row[agent_health_column])

                    full_key = make_full_validation_key(
                        county_name=row["county_name"],
                        settlement_type=row["settlement_type"],
                        dimension_group=dimension_group,
                        dimension_category=dimension_category,
                        health_question=health_question,
                        health_response_category=response_category,
                    )

                    assigned_counts_by_full_key[full_key] += 1

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"Health all-dimension validation agent pass: {total_rows:,}")

    return {
        "total_rows": total_rows,
        "age_5plus_rows": age_5plus_rows,
        "under_5_rows": under_5_rows,
        "agent_counts_by_dimension_key": dict(agent_counts_by_dimension_key),
        "assigned_counts_by_full_key": dict(assigned_counts_by_full_key),
    }


# ============================================================
# 5. SKÁLÁZOTT DIMENZIÓ TARGETEK
# ============================================================

def create_scaled_dimension_targets(
    health_long: pd.DataFrame,
    agent_counts_by_dimension_key: dict,
) -> dict:
    response_weights = build_health_response_weights(
        health_long=health_long
    )

    scaled_target_lookup = {}

    for key, response_weight_dict in response_weights.items():
        dimension_key, health_question = key

        generated_dimension_total = agent_counts_by_dimension_key.get(dimension_key, 0)

        allocation = largest_remainder_allocation(
            response_weights=response_weight_dict,
            target_total=generated_dimension_total,
        )

        for response_category, scaled_count in allocation.items():
            full_key = (
                dimension_key[0],
                dimension_key[1],
                dimension_key[2],
                dimension_key[3],
                health_question,
                response_category,
            )

            scaled_target_lookup[full_key] = scaled_count

    return scaled_target_lookup


# ============================================================
# 6. OUTPUTOK
# ============================================================

def create_validation_table(
    scaled_target_lookup: dict,
    assigned_counts_by_full_key: dict,
) -> pd.DataFrame:
    all_keys = set(scaled_target_lookup.keys()) | set(assigned_counts_by_full_key.keys())

    rows = []

    for key in all_keys:
        scaled_target_count = scaled_target_lookup.get(key, 0)
        assigned_count = assigned_counts_by_full_key.get(key, 0)

        rows.append({
            "county_name": key[0],
            "settlement_type": key[1],
            "dimension_group": key[2],
            "dimension_category": key[3],
            "health_question": key[4],
            "health_response_category": key[5],
            "scaled_target_count": scaled_target_count,
            "assigned_count": assigned_count,
            "difference_assigned_minus_scaled_target": assigned_count - scaled_target_count,
            "absolute_difference": abs(assigned_count - scaled_target_count),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "absolute_difference",
                "dimension_group",
                "health_question",
                "scaled_target_count",
            ],
            ascending=[False, True, True, False],
        )
        .reset_index(drop=True)
    )


def create_dimension_agent_count_table(agent_counts_by_dimension_key: dict) -> pd.DataFrame:
    rows = []

    for key, count in agent_counts_by_dimension_key.items():
        rows.append({
            "county_name": key[0],
            "settlement_type": key[1],
            "dimension_group": key[2],
            "dimension_category": key[3],
            "generated_5plus_agent_count": count,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dimension_group",
                "county_name",
                "settlement_type",
                "dimension_category",
            ]
        )
        .reset_index(drop=True)
    )


def create_summary(validation: pd.DataFrame, count_result: dict) -> pd.DataFrame:
    rows = [
        {
            "metric": "agents_total",
            "dimension_group": "ALL",
            "health_question": "ALL",
            "value": count_result["total_rows"],
            "note": "Teljes agentfájl sorainak száma.",
        },
        {
            "metric": "age_5plus_agents_total",
            "dimension_group": "ALL",
            "health_question": "ALL",
            "value": count_result["age_5plus_rows"],
            "note": "Health validációba bevont 5+ agentek száma.",
        },
        {
            "metric": "under_5_agents_total",
            "dimension_group": "ALL",
            "health_question": "ALL",
            "value": count_result["under_5_rows"],
            "note": "Health source-ból kimaradó 5 év alatti agentek száma.",
        },
    ]

    grouped = (
        validation
        .groupby(["dimension_group", "health_question"], dropna=False)
        .agg(
            scaled_target_total=("scaled_target_count", "sum"),
            assigned_total=("assigned_count", "sum"),
            sum_absolute_difference=("absolute_difference", "sum"),
            max_absolute_difference=("absolute_difference", "max"),
            nonzero_rows=("absolute_difference", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )

    for _, row in grouped.iterrows():
        dimension_group = normalize_text(row["dimension_group"])
        health_question = normalize_text(row["health_question"])

        rows.append({
            "metric": "scaled_target_total",
            "dimension_group": dimension_group,
            "health_question": health_question,
            "value": int(row["scaled_target_total"]),
            "note": "Generated 5+ agent countokra skálázott target total.",
        })

        rows.append({
            "metric": "assigned_total",
            "dimension_group": dimension_group,
            "health_question": health_question,
            "value": int(row["assigned_total"]),
            "note": "Jelenlegi health assignmentből számolt total.",
        })

        rows.append({
            "metric": "sum_absolute_difference",
            "dimension_group": dimension_group,
            "health_question": health_question,
            "value": int(row["sum_absolute_difference"]),
            "note": "Eltérések abszolút összege. Minél kisebb, annál jobb.",
        })

        rows.append({
            "metric": "max_absolute_difference",
            "dimension_group": dimension_group,
            "health_question": health_question,
            "value": int(row["max_absolute_difference"]),
            "note": "Legnagyobb cellaszintű eltérés.",
        })

        rows.append({
            "metric": "nonzero_validation_rows",
            "dimension_group": dimension_group,
            "health_question": health_question,
            "value": int(row["nonzero_rows"]),
            "note": "Nem nulla eltérésű validációs sorok száma.",
        })

    return pd.DataFrame(rows)


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script a jelenlegi age-based health assignmentet validálja mind a négy health source dimenzióval szemben.",
        },
        {
            "order": 2,
            "note": "Vizsgált dimenziók: age_group, sex, education, economic_activity.",
        },
        {
            "order": 3,
            "note": "A validáció a 130_agents_with_age_based_health_status.csv agentfájlon fut.",
        },
        {
            "order": 4,
            "note": "A source eloszlások a 121_health_status_hier_long.csv fájlból jönnek.",
        },
        {
            "order": 5,
            "note": "Mivel a generated 5+ agent total eltér az eredeti health source 5+ bázistól, a source response arányokat minden dimenziókategórián belül a generated agent countokra skálázzuk.",
        },
        {
            "order": 6,
            "note": "Ez a validáció nem módosítja az agenteket, csak megmutatja, hogy az age-based kiosztás mennyire tér el sex, education és economic_activity szerint.",
        },
        {
            "order": 7,
            "note": "A 15 évesnél fiatalabb személy kategória külön kezelődik education és economic_activity dimenzióban, de nem keveredik össze, mert a dimension_group része a validációs kulcsnak.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 7. FŐ FUTTATÁS
# ============================================================

def run_all_dimension_health_validation() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Health long source betöltése...")
    health_long = load_health_long(
        health_long_csv=HEALTH_LONG_CSV
    )

    print("Agentek és assigned health értékek számlálása minden dimenzióban...")
    count_result = count_agents_and_assigned_health_by_dimensions(
        agents_csv=AGENTS_WITH_HEALTH_STATUS_CSV
    )

    print("Skálázott dimenzió targetek készítése...")
    scaled_target_lookup = create_scaled_dimension_targets(
        health_long=health_long,
        agent_counts_by_dimension_key=count_result["agent_counts_by_dimension_key"],
    )

    print("Validációs tábla készítése...")
    validation = create_validation_table(
        scaled_target_lookup=scaled_target_lookup,
        assigned_counts_by_full_key=count_result["assigned_counts_by_full_key"],
    )

    print("Summary készítése...")
    summary = create_summary(
        validation=validation,
        count_result=count_result,
    )

    print("Dimension agent count tábla készítése...")
    dimension_agent_counts = create_dimension_agent_count_table(
        agent_counts_by_dimension_key=count_result["agent_counts_by_dimension_key"],
    )

    print("Outputok mentése...")

    validation.to_csv(
        VALIDATION_BY_DIMENSION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        VALIDATION_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    dimension_agent_counts.to_csv(
        DIMENSION_AGENT_COUNTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    write_method_notes()

    print()
    print("Kész.")
    print(f"Agentek összesen: {count_result['total_rows']:,}")
    print(f"5+ agentek: {count_result['age_5plus_rows']:,}")
    print(f"5 év alatti agentek: {count_result['under_5_rows']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_all_dimension_health_validation()