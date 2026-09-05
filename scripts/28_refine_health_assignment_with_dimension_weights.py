from pathlib import Path
import csv
import random
import math
from collections import defaultdict
from array import array

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

INPUT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/health_status_assignment/"
    "130_agents_with_age_based_health_status.csv"
)

HEALTH_LONG_CSV = Path(
    "../outputs/agent_attribute_generation/health_status_assignment/"
    "121_health_status_hier_long.csv"
)

SCALED_AGE_HEALTH_TARGETS_CSV = Path(
    "../outputs/agent_attribute_generation/health_status_assignment/"
    "137_generated_scaled_age_based_health_targets.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/health_status_assignment"
)

REFINED_AGENTS_CSV = OUTPUT_FOLDER / "144_agents_with_refined_health_status.csv"
REFINED_VALIDATION_ALL_DIMENSIONS_CSV = OUTPUT_FOLDER / "145_refined_health_validation_by_all_dimensions.csv"
REFINED_VALIDATION_SUMMARY_CSV = OUTPUT_FOLDER / "146_refined_health_validation_summary.csv"
REFINED_AGE_CELL_VALIDATION_CSV = OUTPUT_FOLDER / "147_refined_health_age_cell_validation.csv"
REFINED_METHOD_NOTES_CSV = OUTPUT_FOLDER / "148_refined_health_method_notes.csv"

RANDOM_SEED = 42
MAX_ROWS_TO_PROCESS = None

UNDER_5_NOT_IN_SOURCE = "UNDER_5_NOT_IN_SOURCE"
NAPP = "NAPP"

DIMENSION_GROUPS_TO_IMPROVE = [
    "sex",
    "education",
    "economic_activity",
]

# Súlyok: age_group nem szerepel itt, mert azt keményen fixen tartjuk
# a county × settlement_type × health_age_group response targetekkel.
DIMENSION_WEIGHTS = {
    "sex": 1.0,
    "education": 2.0,
    "economic_activity": 2.0,
}

HEALTH_QUESTION_TO_AGENT_COLUMN = {
    "disability_or_severe_limitation": "health_disability_status",
    "chronic_disease": "health_chronic_disease_status",
    "limitation_level": "health_limitation_status",
}

HEALTH_QUESTION_TO_METHOD_COLUMN = {
    "disability_or_severe_limitation": "health_disability_assignment_method",
    "chronic_disease": "health_chronic_disease_assignment_method",
    "limitation_level": "health_limitation_assignment_method",
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


def make_age_cell_key(
    county_name: str,
    settlement_type: str,
    health_age_group: str,
) -> tuple[str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_text(health_age_group),
    )


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


def make_full_dimension_health_key(
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
# 3. AGENTEK BEOLVASÁSA SLIM STRUKTÚRÁBA
# ============================================================

def collect_agents_for_refinement(input_agents_csv: Path) -> dict:
    total_rows = 0
    age_5plus_rows = 0
    under_5_rows = 0

    agents_by_age_cell = defaultdict(list)
    agent_counts_by_dimension_key = defaultdict(int)

    dimension_key_to_id = {}
    id_to_dimension_key = {}

    def get_dimension_key_id(dimension_key: tuple[str, str, str, str]) -> int:
        if dimension_key not in dimension_key_to_id:
            new_id = len(dimension_key_to_id) + 1
            dimension_key_to_id[dimension_key] = new_id
            id_to_dimension_key[new_id] = dimension_key

        return dimension_key_to_id[dimension_key]

    with input_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        required_columns = [
            "county_name",
            "settlement_type",
            "sex",
            "exact_age",
            "education_level_calibrated",
            "economic_activity_status_calibrated",
        ]

        for column_name in required_columns:
            if column_name not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop az input agentfájlból: {column_name}"
                )

        for row_number, row in enumerate(reader, start=1):
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])

            if exact_age < 5:
                under_5_rows += 1
                total_rows += 1
                continue

            health_age_group = map_exact_age_to_health_age_group(exact_age)

            age_cell_key = make_age_cell_key(
                county_name=row["county_name"],
                settlement_type=row["settlement_type"],
                health_age_group=health_age_group,
            )

            dimension_ids_for_agent = []

            for dimension_group in DIMENSION_GROUPS_TO_IMPROVE:
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

                dimension_id = get_dimension_key_id(dimension_key)
                dimension_ids_for_agent.append(dimension_id)

                agent_counts_by_dimension_key[dimension_key] += 1

            agents_by_age_cell[age_cell_key].append(
                (
                    row_number,
                    tuple(dimension_ids_for_agent),
                )
            )

            age_5plus_rows += 1
            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"Health refinement agent collection: {total_rows:,} sor")

    return {
        "total_rows": total_rows,
        "age_5plus_rows": age_5plus_rows,
        "under_5_rows": under_5_rows,
        "agents_by_age_cell": dict(agents_by_age_cell),
        "agent_counts_by_dimension_key": dict(agent_counts_by_dimension_key),
        "dimension_key_to_id": dimension_key_to_id,
        "id_to_dimension_key": id_to_dimension_key,
    }


# ============================================================
# 4. TARGETEK BETÖLTÉSE
# ============================================================

def load_scaled_age_health_targets(targets_csv: Path) -> pd.DataFrame:
    targets = pd.read_csv(
        targets_csv,
        keep_default_na=False,
    )

    required_columns = [
        "county_name",
        "settlement_type",
        "health_age_group",
        "health_question",
        "health_response_category",
        "target_count",
    ]

    for column_name in required_columns:
        if column_name not in targets.columns:
            raise ValueError(
                f"Hiányzó oszlop a scaled age health target fájlból: {column_name}"
            )

    targets["county_name"] = targets["county_name"].apply(normalize_county_name)
    targets["settlement_type"] = targets["settlement_type"].apply(normalize_settlement_type)
    targets["health_age_group"] = targets["health_age_group"].apply(normalize_text)
    targets["health_question"] = targets["health_question"].apply(normalize_text)
    targets["health_response_category"] = targets["health_response_category"].apply(normalize_text)
    targets["target_count"] = targets["target_count"].apply(convert_count_to_integer)

    return targets


def build_age_cell_response_targets(
    scaled_age_targets: pd.DataFrame,
) -> dict:
    targets_by_question_and_age_cell = defaultdict(lambda: defaultdict(dict))

    for _, row in scaled_age_targets.iterrows():
        age_cell_key = make_age_cell_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            health_age_group=row["health_age_group"],
        )

        health_question = normalize_text(row["health_question"])
        response_category = normalize_text(row["health_response_category"])
        target_count = convert_count_to_integer(row["target_count"])

        targets_by_question_and_age_cell[health_question][age_cell_key][response_category] = (
            targets_by_question_and_age_cell[health_question][age_cell_key].get(response_category, 0)
            + target_count
        )

    return {
        health_question: dict(age_cell_dict)
        for health_question, age_cell_dict in targets_by_question_and_age_cell.items()
    }


def load_health_long(health_long_csv: Path) -> pd.DataFrame:
    health_long = pd.read_csv(
        health_long_csv,
        keep_default_na=False,
    )

    health_long["county_name"] = health_long["county_name"].apply(normalize_county_name)
    health_long["settlement_type"] = health_long["settlement_type"].apply(normalize_settlement_type)
    health_long["dimension_group"] = health_long["dimension_group"].apply(normalize_text)
    health_long["dimension_category"] = health_long["dimension_category"].apply(normalize_text)
    health_long["health_question"] = health_long["health_question"].apply(normalize_text)
    health_long["health_response_category"] = health_long["health_response_category"].apply(normalize_text)
    health_long["health_count"] = health_long["health_count"].apply(convert_count_to_integer)

    return health_long


def build_scaled_dimension_response_targets(
    health_long: pd.DataFrame,
    agent_counts_by_dimension_key: dict,
) -> dict:
    """
    Sex / education / economic_activity dimenziók céljai.

    Mivel az eredeti health source total nem pont azonos a generated 5+ agent total-lal,
    minden dimension_key-n belül a response arányokat megtartjuk,
    de a generated agent countokra skálázzuk.
    """

    health_long = health_long[
        health_long["dimension_group"].isin(DIMENSION_GROUPS_TO_IMPROVE)
    ].copy()

    response_weights = defaultdict(lambda: defaultdict(int))

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

        response_weights[(dimension_key, health_question)][response_category] += health_count

    scaled_targets = defaultdict(int)

    for key, weights in response_weights.items():
        dimension_key, health_question = key
        generated_dimension_total = agent_counts_by_dimension_key.get(dimension_key, 0)

        allocation = largest_remainder_allocation(
            response_weights=dict(weights),
            target_total=generated_dimension_total,
        )

        for response_category, scaled_count in allocation.items():
            full_key = (
                dimension_key,
                health_question,
                response_category,
            )

            scaled_targets[full_key] += scaled_count

    return dict(scaled_targets)


# ============================================================
# 5. REFINEMENT ASSIGNMENT
# ============================================================

def build_response_code_maps(
    scaled_age_targets: pd.DataFrame,
) -> dict:
    response_to_code_by_question = {}
    code_to_response_by_question = {}

    for health_question, group in scaled_age_targets.groupby("health_question"):
        responses = sorted(group["health_response_category"].unique().tolist())

        responses.append(UNDER_5_NOT_IN_SOURCE)

        response_to_code = {}
        code_to_response = {}

        for code, response_category in enumerate(responses, start=1):
            response_to_code[response_category] = code
            code_to_response[code] = response_category

        response_to_code_by_question[health_question] = response_to_code
        code_to_response_by_question[health_question] = code_to_response

    return {
        "response_to_code_by_question": response_to_code_by_question,
        "code_to_response_by_question": code_to_response_by_question,
    }


def score_response_for_agent(
    dimension_ids_for_agent: tuple[int, ...],
    response_category: str,
    health_question: str,
    remaining_need_by_dimension_id_response: dict,
    id_to_dimension_key: dict,
    random_generator: random.Random,
) -> float:
    score = 0.0

    for dimension_id in dimension_ids_for_agent:
        dimension_key = id_to_dimension_key[dimension_id]
        dimension_group = dimension_key[2]

        weight = DIMENSION_WEIGHTS.get(dimension_group, 1.0)

        need_key = (
            dimension_id,
            health_question,
            response_category,
        )

        remaining_need = remaining_need_by_dimension_id_response.get(need_key, 0)

        if remaining_need > 0:
            score += weight * remaining_need

    score += random_generator.random()

    return score


def refine_one_health_question(
    health_question: str,
    total_rows: int,
    agents_by_age_cell: dict,
    age_cell_response_targets_by_question: dict,
    scaled_dimension_targets: dict,
    dimension_key_to_id: dict,
    id_to_dimension_key: dict,
    response_to_code: dict,
) -> dict:
    random_generator = random.Random(RANDOM_SEED)

    assigned_codes = array("H", [0]) * (total_rows + 1)

    remaining_need_by_dimension_id_response = defaultdict(int)

    for full_key, target_count in scaled_dimension_targets.items():
        dimension_key, target_health_question, response_category = full_key

        if target_health_question != health_question:
            continue

        dimension_id = dimension_key_to_id.get(dimension_key)

        if dimension_id is None:
            continue

        remaining_need_by_dimension_id_response[
            (
                dimension_id,
                health_question,
                response_category,
            )
        ] = target_count

    assigned_counts_by_dimension_target = defaultdict(int)
    assigned_counts_by_age_cell_response = defaultdict(int)

    method_counts = defaultdict(int)

    age_cell_targets = age_cell_response_targets_by_question[health_question]

    cells_processed = 0
    agents_assigned_5plus = 0

    # Nagyobb cellákkal kezdünk, mert ott van a legtöbb optimalizációs tér.
    sorted_age_cells = sorted(
        agents_by_age_cell.keys(),
        key=lambda cell_key: len(agents_by_age_cell[cell_key]),
        reverse=True,
    )

    for age_cell_key in sorted_age_cells:
        agents_in_cell = list(agents_by_age_cell[age_cell_key])

        random_generator.shuffle(agents_in_cell)

        response_quota = dict(age_cell_targets.get(age_cell_key, {}))

        cell_agent_count = len(agents_in_cell)
        quota_total = sum(response_quota.values())

        if quota_total != cell_agent_count:
            raise ValueError(
                f"Age cell quota mismatch ennél: {age_cell_key}, "
                f"agents={cell_agent_count}, quota={quota_total}, "
                f"health_question={health_question}"
            )

        available_responses = [
            response_category
            for response_category, quota in response_quota.items()
            if quota > 0
        ]

        for row_number, dimension_ids_for_agent in agents_in_cell:
            best_response = None
            best_score = None

            for response_category in available_responses:
                if response_quota.get(response_category, 0) <= 0:
                    continue

                score = score_response_for_agent(
                    dimension_ids_for_agent=dimension_ids_for_agent,
                    response_category=response_category,
                    health_question=health_question,
                    remaining_need_by_dimension_id_response=remaining_need_by_dimension_id_response,
                    id_to_dimension_key=id_to_dimension_key,
                    random_generator=random_generator,
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_response = response_category

            if best_response is None:
                raise ValueError(
                    f"Nem találtam kiosztható response kategóriát: "
                    f"{health_question}, {age_cell_key}"
                )

            response_quota[best_response] -= 1

            if response_quota[best_response] == 0:
                available_responses = [
                    response_category
                    for response_category in available_responses
                    if response_quota.get(response_category, 0) > 0
                ]

            assigned_codes[row_number] = response_to_code[best_response]

            assigned_counts_by_age_cell_response[
                (
                    age_cell_key,
                    health_question,
                    best_response,
                )
            ] += 1

            for dimension_id in dimension_ids_for_agent:
                need_key = (
                    dimension_id,
                    health_question,
                    best_response,
                )

                remaining_need_by_dimension_id_response[need_key] -= 1

                dimension_key = id_to_dimension_key[dimension_id]

                assigned_counts_by_dimension_target[
                    (
                        dimension_key,
                        health_question,
                        best_response,
                    )
                ] += 1

            method_counts["REFINED_AGE_CELL_WEIGHTED"] += 1
            agents_assigned_5plus += 1

        cells_processed += 1

        if cells_processed % 100 == 0:
            print(
                f"{health_question}: refined age cells {cells_processed:,}/"
                f"{len(sorted_age_cells):,}"
            )

    return {
        "assigned_codes": assigned_codes,
        "assigned_counts_by_dimension_target": dict(assigned_counts_by_dimension_target),
        "assigned_counts_by_age_cell_response": dict(assigned_counts_by_age_cell_response),
        "method_counts": dict(method_counts),
        "agents_assigned_5plus": agents_assigned_5plus,
    }


def refine_all_health_questions(
    total_rows: int,
    agents_by_age_cell: dict,
    age_cell_response_targets_by_question: dict,
    scaled_dimension_targets: dict,
    dimension_key_to_id: dict,
    id_to_dimension_key: dict,
    response_code_maps: dict,
) -> dict:
    results_by_question = {}

    for health_question in HEALTH_QUESTION_TO_AGENT_COLUMN:
        print(f"Refinement indul: {health_question}")

        response_to_code = response_code_maps["response_to_code_by_question"][health_question]

        result = refine_one_health_question(
            health_question=health_question,
            total_rows=total_rows,
            agents_by_age_cell=agents_by_age_cell,
            age_cell_response_targets_by_question=age_cell_response_targets_by_question,
            scaled_dimension_targets=scaled_dimension_targets,
            dimension_key_to_id=dimension_key_to_id,
            id_to_dimension_key=id_to_dimension_key,
            response_to_code=response_to_code,
        )

        results_by_question[health_question] = result

    return results_by_question


# ============================================================
# 6. OUTPUT AGENTFÁJL ÍRÁSA
# ============================================================

def write_refined_agent_file(
    input_agents_csv: Path,
    output_agents_csv: Path,
    refinement_results: dict,
    response_code_maps: dict,
) -> dict:
    total_written = 0
    under_5_written = 0
    age_5plus_written = 0

    with input_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)
        input_fieldnames = list(reader.fieldnames)

        output_fieldnames = list(input_fieldnames)

        for health_question in HEALTH_QUESTION_TO_AGENT_COLUMN:
            status_column = HEALTH_QUESTION_TO_AGENT_COLUMN[health_question]
            method_column = HEALTH_QUESTION_TO_METHOD_COLUMN[health_question]

            if status_column not in output_fieldnames:
                output_fieldnames.append(status_column)

            if method_column not in output_fieldnames:
                output_fieldnames.append(method_column)

        if "health_refinement_source" not in output_fieldnames:
            output_fieldnames.append("health_refinement_source")

        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for row_number, row in enumerate(reader, start=1):
            if MAX_ROWS_TO_PROCESS is not None and total_written >= MAX_ROWS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])

            if exact_age < 5:
                row["health_refinement_source"] = UNDER_5_NOT_IN_SOURCE

                for health_question in HEALTH_QUESTION_TO_AGENT_COLUMN:
                    status_column = HEALTH_QUESTION_TO_AGENT_COLUMN[health_question]
                    method_column = HEALTH_QUESTION_TO_METHOD_COLUMN[health_question]

                    row[status_column] = UNDER_5_NOT_IN_SOURCE
                    row[method_column] = NAPP

                under_5_written += 1

            else:
                row["health_refinement_source"] = "AGE_CELL_FIXED_MULTIDIM_WEIGHTED"

                for health_question in HEALTH_QUESTION_TO_AGENT_COLUMN:
                    status_column = HEALTH_QUESTION_TO_AGENT_COLUMN[health_question]
                    method_column = HEALTH_QUESTION_TO_METHOD_COLUMN[health_question]

                    code = refinement_results[health_question]["assigned_codes"][row_number]

                    if code == 0:
                        raise ValueError(
                            f"Nincs refined health code: row={row_number}, "
                            f"question={health_question}"
                        )

                    response_category = response_code_maps["code_to_response_by_question"][
                        health_question
                    ][code]

                    row[status_column] = response_category
                    row[method_column] = "REFINED_AGE_CELL_WEIGHTED"

                age_5plus_written += 1

            writer.writerow(row)
            total_written += 1

            if total_written % 1_000_000 == 0:
                print(f"Refined health agent output: {total_written:,} sor")

    return {
        "total_written": total_written,
        "age_5plus_written": age_5plus_written,
        "under_5_written": under_5_written,
    }


# ============================================================
# 7. VALIDÁCIÓK
# ============================================================

def create_all_dimension_validation(
    scaled_dimension_targets: dict,
    refinement_results: dict,
) -> pd.DataFrame:
    assigned_lookup = defaultdict(int)

    for health_question, result in refinement_results.items():
        for key, count in result["assigned_counts_by_dimension_target"].items():
            assigned_lookup[key] += count

    all_keys = set(scaled_dimension_targets.keys()) | set(assigned_lookup.keys())

    rows = []

    for key in all_keys:
        dimension_key, health_question, response_category = key

        target_count = scaled_dimension_targets.get(key, 0)
        assigned_count = assigned_lookup.get(key, 0)

        rows.append({
            "county_name": dimension_key[0],
            "settlement_type": dimension_key[1],
            "dimension_group": dimension_key[2],
            "dimension_category": dimension_key[3],
            "health_question": health_question,
            "health_response_category": response_category,
            "scaled_target_count": target_count,
            "assigned_count": assigned_count,
            "difference_assigned_minus_scaled_target": assigned_count - target_count,
            "absolute_difference": abs(assigned_count - target_count),
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


def create_age_cell_validation(
    scaled_age_targets: pd.DataFrame,
    refinement_results: dict,
) -> pd.DataFrame:
    target_lookup = defaultdict(int)

    for _, row in scaled_age_targets.iterrows():
        age_cell_key = make_age_cell_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            health_age_group=row["health_age_group"],
        )

        key = (
            age_cell_key,
            normalize_text(row["health_question"]),
            normalize_text(row["health_response_category"]),
        )

        target_lookup[key] += convert_count_to_integer(row["target_count"])

    assigned_lookup = defaultdict(int)

    for health_question, result in refinement_results.items():
        for key, count in result["assigned_counts_by_age_cell_response"].items():
            assigned_lookup[key] += count

    all_keys = set(target_lookup.keys()) | set(assigned_lookup.keys())

    rows = []

    for key in all_keys:
        age_cell_key, health_question, response_category = key

        target_count = target_lookup.get(key, 0)
        assigned_count = assigned_lookup.get(key, 0)

        rows.append({
            "county_name": age_cell_key[0],
            "settlement_type": age_cell_key[1],
            "health_age_group": age_cell_key[2],
            "health_question": health_question,
            "health_response_category": response_category,
            "scaled_age_target_count": target_count,
            "assigned_count": assigned_count,
            "difference_assigned_minus_scaled_age_target": assigned_count - target_count,
            "absolute_difference": abs(assigned_count - target_count),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "absolute_difference",
                "health_question",
                "scaled_age_target_count",
            ],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )


def create_summary(
    collection_result: dict,
    write_result: dict,
    all_dimension_validation: pd.DataFrame,
    age_cell_validation: pd.DataFrame,
    refinement_results: dict,
) -> pd.DataFrame:
    rows = [
        {
            "metric": "output_agents_total",
            "dimension_group": "ALL",
            "health_question": "ALL",
            "value": write_result["total_written"],
            "note": "Refined output agentfájl sorainak száma.",
        },
        {
            "metric": "age_5plus_agents_total",
            "dimension_group": "ALL",
            "health_question": "ALL",
            "value": write_result["age_5plus_written"],
            "note": "Health refinementbe bevont 5+ agentek száma.",
        },
        {
            "metric": "under_5_agents_total",
            "dimension_group": "ALL",
            "health_question": "ALL",
            "value": write_result["under_5_written"],
            "note": "5 év alatti agentek, akik UNDER_5_NOT_IN_SOURCE jelölést kaptak.",
        },
    ]

    for health_question, result in refinement_results.items():
        rows.append({
            "metric": "refined_assigned_5plus",
            "dimension_group": "ALL",
            "health_question": health_question,
            "value": result["agents_assigned_5plus"],
            "note": "Refined health response-t kapott 5+ agentek száma az adott kérdésben.",
        })

        for method, count in result["method_counts"].items():
            rows.append({
                "metric": f"assignment_method__{method}",
                "dimension_group": "ALL",
                "health_question": health_question,
                "value": count,
                "note": "Refined assignment method count.",
            })

    all_dim_grouped = (
        all_dimension_validation
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

    for _, row in all_dim_grouped.iterrows():
        dimension_group = normalize_text(row["dimension_group"])
        health_question = normalize_text(row["health_question"])

        rows.append({
            "metric": "sum_absolute_difference",
            "dimension_group": dimension_group,
            "health_question": health_question,
            "value": int(row["sum_absolute_difference"]),
            "note": "Sex/education/economic_activity skálázott targethez mért abszolút eltérés.",
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

    age_grouped = (
        age_cell_validation
        .groupby("health_question", dropna=False)
        .agg(
            age_sum_absolute_difference=("absolute_difference", "sum"),
            age_max_absolute_difference=("absolute_difference", "max"),
            age_nonzero_rows=("absolute_difference", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )

    for _, row in age_grouped.iterrows():
        health_question = normalize_text(row["health_question"])

        rows.append({
            "metric": "age_cell_sum_absolute_difference",
            "dimension_group": "age_group_FIXED_CONSTRAINT",
            "health_question": health_question,
            "value": int(row["age_sum_absolute_difference"]),
            "note": "Age cell constraint eltérés. Ideálisan 0.",
        })

        rows.append({
            "metric": "age_cell_max_absolute_difference",
            "dimension_group": "age_group_FIXED_CONSTRAINT",
            "health_question": health_question,
            "value": int(row["age_max_absolute_difference"]),
            "note": "Age cell constraint max eltérés. Ideálisan 0.",
        })

        rows.append({
            "metric": "age_cell_nonzero_rows",
            "dimension_group": "age_group_FIXED_CONSTRAINT",
            "health_question": health_question,
            "value": int(row["age_nonzero_rows"]),
            "note": "Age cell nem nulla eltérésű sorok. Ideálisan 0.",
        })

    return pd.DataFrame(rows)


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script a korábbi age-based health assignmentet finomítja.",
        },
        {
            "order": 2,
            "note": "A county × settlement_type × health_age_group × health_question × response eloszlást kemény constraintként megtartja.",
        },
        {
            "order": 3,
            "note": "A finomítás age cellán belül történik: ugyanazon age cellán belül variáljuk, melyik agent melyik health response kategóriát kapja.",
        },
        {
            "order": 4,
            "note": "A cél a sex, education és economic_activity szerinti marginális health targetekhez való jobb illeszkedés.",
        },
        {
            "order": 5,
            "note": "A sex, education és economic_activity targeteket a health source response arányai alapján, a generated 5+ agent countokra skálázva használjuk.",
        },
        {
            "order": 6,
            "note": "Az 5 év alatti agentek továbbra is UNDER_5_NOT_IN_SOURCE jelölést kapnak.",
        },
        {
            "order": 7,
            "note": "A módszer greedy weighted refinement: minden age cellán belül olyan response kiosztást keres, amely csökkenti a többi dimenzió fennmaradó célhiányát.",
        },
        {
            "order": 8,
            "note": "Nem garantál matematikai optimumot, de megőrzi az életkori kiosztást, és várhatóan jelentősen javítja a sex/education/economic_activity illeszkedést az age-only random kiosztáshoz képest.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        REFINED_METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 8. FŐ FUTTATÁS
# ============================================================

def run_health_refinement() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Scaled age health targetek betöltése...")
    scaled_age_targets = load_scaled_age_health_targets(
        targets_csv=SCALED_AGE_HEALTH_TARGETS_CSV
    )

    print("Health long source betöltése...")
    health_long = load_health_long(
        health_long_csv=HEALTH_LONG_CSV
    )

    print("Agentek slim gyűjtése refinementhez...")
    collection_result = collect_agents_for_refinement(
        input_agents_csv=INPUT_AGENTS_CSV
    )

    print("Age cell response targetek építése...")
    age_cell_response_targets_by_question = build_age_cell_response_targets(
        scaled_age_targets=scaled_age_targets
    )

    print("Skálázott sex/education/economic_activity targetek építése...")
    scaled_dimension_targets = build_scaled_dimension_response_targets(
        health_long=health_long,
        agent_counts_by_dimension_key=collection_result["agent_counts_by_dimension_key"],
    )

    print("Response code mapek építése...")
    response_code_maps = build_response_code_maps(
        scaled_age_targets=scaled_age_targets
    )

    print("Health kérdések finomítása...")
    refinement_results = refine_all_health_questions(
        total_rows=collection_result["total_rows"],
        agents_by_age_cell=collection_result["agents_by_age_cell"],
        age_cell_response_targets_by_question=age_cell_response_targets_by_question,
        scaled_dimension_targets=scaled_dimension_targets,
        dimension_key_to_id=collection_result["dimension_key_to_id"],
        id_to_dimension_key=collection_result["id_to_dimension_key"],
        response_code_maps=response_code_maps,
    )

    print("Refined agentfájl írása...")
    write_result = write_refined_agent_file(
        input_agents_csv=INPUT_AGENTS_CSV,
        output_agents_csv=REFINED_AGENTS_CSV,
        refinement_results=refinement_results,
        response_code_maps=response_code_maps,
    )

    print("All-dimension validáció készítése...")
    all_dimension_validation = create_all_dimension_validation(
        scaled_dimension_targets=scaled_dimension_targets,
        refinement_results=refinement_results,
    )

    all_dimension_validation.to_csv(
        REFINED_VALIDATION_ALL_DIMENSIONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Age-cell validáció készítése...")
    age_cell_validation = create_age_cell_validation(
        scaled_age_targets=scaled_age_targets,
        refinement_results=refinement_results,
    )

    age_cell_validation.to_csv(
        REFINED_AGE_CELL_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Summary készítése...")
    summary = create_summary(
        collection_result=collection_result,
        write_result=write_result,
        all_dimension_validation=all_dimension_validation,
        age_cell_validation=age_cell_validation,
        refinement_results=refinement_results,
    )

    summary.to_csv(
        REFINED_VALIDATION_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Method notes írása...")
    write_method_notes()

    print()
    print("Kész.")
    print(f"Output agentek: {write_result['total_written']:,}")
    print(f"5+ agentek: {write_result['age_5plus_written']:,}")
    print(f"5 év alatti agentek: {write_result['under_5_written']:,}")

    for health_question, result in refinement_results.items():
        print(f"{health_question}: {result['method_counts']}")

    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_health_refinement()