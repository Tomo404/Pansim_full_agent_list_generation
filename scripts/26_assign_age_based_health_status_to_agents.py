from pathlib import Path
import csv
import random
from collections import defaultdict, deque

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

CURRENT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/school_attendance_assignment/"
    "116_agents_with_school_attendance.csv"
)

AGE_BASED_HEALTH_TARGETS_CSV = Path(
    "../outputs/agent_attribute_generation/health_status_assignment/"
    "137_generated_scaled_age_based_health_targets.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/health_status_assignment"
)

AGENTS_WITH_HEALTH_STATUS_CSV = OUTPUT_FOLDER / "130_agents_with_age_based_health_status.csv"
HEALTH_ASSIGNMENT_PLAN_CSV = OUTPUT_FOLDER / "131_health_assignment_plan_by_age_cell.csv"
HEALTH_VALIDATION_BY_TARGET_CELL_CSV = OUTPUT_FOLDER / "132_health_assignment_validation_by_age_cell.csv"
HEALTH_ASSIGNMENT_SUMMARY_CSV = OUTPUT_FOLDER / "133_health_assignment_summary.csv"
HEALTH_BY_EMPLOYMENT_STATUS_CSV = OUTPUT_FOLDER / "134_health_status_by_employment_status.csv"
HEALTH_BY_SCHOOL_STATUS_CSV = OUTPUT_FOLDER / "135_health_status_by_school_status.csv"
HEALTH_ASSIGNMENT_METHOD_NOTES_CSV = OUTPUT_FOLDER / "136_health_assignment_method_notes.csv"

RANDOM_SEED = 42
MAX_ROWS_TO_PROCESS = None

UNDER_5_NOT_IN_SOURCE = "UNDER_5_NOT_IN_SOURCE"
NAPP = "NAPP"


HEALTH_QUESTION_TO_OUTPUT_COLUMNS = {
    "disability_or_severe_limitation": {
        "status_column": "health_disability_status",
        "method_column": "health_disability_assignment_method",
    },
    "chronic_disease": {
        "status_column": "health_chronic_disease_status",
        "method_column": "health_chronic_disease_assignment_method",
    },
    "limitation_level": {
        "status_column": "health_limitation_status",
        "method_column": "health_limitation_assignment_method",
    },
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


def make_health_demo_key(
    county_name: str,
    settlement_type: str,
    health_age_group: str,
) -> tuple[str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_text(health_age_group),
    )


def make_agent_health_demo_key(row: dict) -> tuple[str, str, str]:
    exact_age = convert_count_to_integer(row["exact_age"])

    return make_health_demo_key(
        county_name=row["county_name"],
        settlement_type=row["settlement_type"],
        health_age_group=map_exact_age_to_health_age_group(exact_age),
    )


def make_full_health_key(
    county_name: str,
    settlement_type: str,
    health_age_group: str,
    health_question: str,
    health_response_category: str,
) -> tuple[str, str, str, str, str]:
    demo_key = make_health_demo_key(
        county_name=county_name,
        settlement_type=settlement_type,
        health_age_group=health_age_group,
    )

    return (
        demo_key[0],
        demo_key[1],
        demo_key[2],
        normalize_text(health_question),
        normalize_text(health_response_category),
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# 3. TARGETEK BETÖLTÉSE
# ============================================================

def load_age_based_health_targets(targets_csv: Path) -> pd.DataFrame:
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
                f"Hiányzó oszlop a health target fájlból: {column_name}"
            )

    targets["county_name"] = targets["county_name"].apply(normalize_county_name)
    targets["settlement_type"] = targets["settlement_type"].apply(normalize_settlement_type)
    targets["health_age_group"] = targets["health_age_group"].apply(normalize_text)
    targets["health_question"] = targets["health_question"].apply(normalize_text)
    targets["health_response_category"] = targets["health_response_category"].apply(normalize_text)
    targets["target_count"] = targets["target_count"].apply(convert_count_to_integer)

    return targets


# ============================================================
# 4. CURRENT AGENTEK SZÁMLÁLÁSA HEALTH AGE CELLÁKHOZ
# ============================================================

def count_current_agents_by_health_demo_key(current_agents_csv: Path) -> dict:
    total_rows = 0
    age_5plus_total = 0
    under_5_total = 0

    candidate_counts_by_demo_key = defaultdict(int)

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        required_columns = [
            "county_name",
            "settlement_type",
            "exact_age",
        ]

        for column_name in required_columns:
            if column_name not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop a current agent fájlban: {column_name}"
                )

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])

            if exact_age < 5:
                under_5_total += 1
            else:
                demo_key = make_agent_health_demo_key(row)
                candidate_counts_by_demo_key[demo_key] += 1
                age_5plus_total += 1

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"Health candidate count pass: {total_rows:,} agent")

    return {
        "total_rows": total_rows,
        "age_5plus_total": age_5plus_total,
        "under_5_total": under_5_total,
        "candidate_counts_by_demo_key": dict(candidate_counts_by_demo_key),
    }


# ============================================================
# 5. ASSIGNMENT QUEUE-K ÉPÍTÉSE
# ============================================================

def build_health_assignment_queues(
    targets: pd.DataFrame,
    candidate_counts_by_demo_key: dict,
) -> dict:
    """
    Kérdésenként külön queue-kat építünk.

    Exact queue:
        ugyanarra a county × settlement_type × health_age_group cellára.

    Fallback queue:
        azok a target response-ok, amelyek olyan cellában maradnak,
        ahol target > candidate.
    """

    random_generator = random.Random(RANDOM_SEED)

    exact_response_queues_by_question_and_demo_key = defaultdict(lambda: defaultdict(deque))
    fallback_response_queues_by_question = defaultdict(deque)

    target_total_by_question = defaultdict(int)
    exact_capacity_by_question = defaultdict(int)
    fallback_response_count_by_question = defaultdict(int)
    local_shortage_by_question = defaultdict(int)
    local_surplus_agents_by_question = defaultdict(int)

    grouped = (
        targets
        .groupby(
            [
                "health_question",
                "county_name",
                "settlement_type",
                "health_age_group",
            ],
            dropna=False,
        )
    )

    for group_key, group in grouped:
        health_question, county_name, settlement_type, health_age_group = group_key

        demo_key = make_health_demo_key(
            county_name=county_name,
            settlement_type=settlement_type,
            health_age_group=health_age_group,
        )

        response_list = []

        for _, row in group.iterrows():
            response_category = normalize_text(row["health_response_category"])
            target_count = convert_count_to_integer(row["target_count"])

            response_list.extend([response_category] * target_count)
            target_total_by_question[health_question] += target_count

        random_generator.shuffle(response_list)

        candidate_count = candidate_counts_by_demo_key.get(demo_key, 0)
        target_count_for_demo = len(response_list)

        exact_count = min(candidate_count, target_count_for_demo)

        exact_response_queues_by_question_and_demo_key[health_question][demo_key] = deque(
            response_list[:exact_count]
        )

        exact_capacity_by_question[health_question] += exact_count

        if target_count_for_demo > candidate_count:
            remaining_responses = response_list[exact_count:]

            for response_category in remaining_responses:
                fallback_response_queues_by_question[health_question].append(response_category)

            fallback_response_count_by_question[health_question] += len(remaining_responses)
            local_shortage_by_question[health_question] += len(remaining_responses)

        if candidate_count > target_count_for_demo:
            local_surplus_agents_by_question[health_question] += (
                candidate_count - target_count_for_demo
            )

    return {
        "exact_response_queues_by_question_and_demo_key": exact_response_queues_by_question_and_demo_key,
        "fallback_response_queues_by_question": fallback_response_queues_by_question,
        "target_total_by_question": dict(target_total_by_question),
        "exact_capacity_by_question": dict(exact_capacity_by_question),
        "fallback_response_count_by_question": dict(fallback_response_count_by_question),
        "local_shortage_by_question": dict(local_shortage_by_question),
        "local_surplus_agents_by_question": dict(local_surplus_agents_by_question),
    }


# ============================================================
# 6. AGENTFÁJL ÍRÁSA HEALTH ATTRIBÚTUMOKKAL
# ============================================================

def write_agents_with_health_status(
    current_agents_csv: Path,
    output_agents_csv: Path,
    queue_result: dict,
) -> dict:
    exact_queues = queue_result["exact_response_queues_by_question_and_demo_key"]
    fallback_queues = queue_result["fallback_response_queues_by_question"]

    total_written = 0
    under_5_written = 0
    age_5plus_written = 0

    assigned_counts_by_full_key = defaultdict(int)
    method_counts_by_question = defaultdict(lambda: defaultdict(int))
    response_counts_by_question = defaultdict(lambda: defaultdict(int))

    health_by_employment_status_counts = defaultdict(int)
    health_by_school_status_counts = defaultdict(int)

    fallback_used_by_question = defaultdict(int)
    missing_health_assignment_by_question = defaultdict(int)

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)
        input_fieldnames = list(reader.fieldnames)

        new_columns = []

        for health_question, column_info in HEALTH_QUESTION_TO_OUTPUT_COLUMNS.items():
            new_columns.append(column_info["status_column"])
            new_columns.append(column_info["method_column"])

        new_columns.append("health_assignment_source")

        output_fieldnames = input_fieldnames + [
            column for column in new_columns
            if column not in input_fieldnames
        ]

        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_written >= MAX_ROWS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])
            employment_status = normalize_text(row.get("employment_layer_status", ""))
            school_status = normalize_text(row.get("school_attendance_status", ""))

            if exact_age < 5:
                row["health_assignment_source"] = UNDER_5_NOT_IN_SOURCE

                for health_question, column_info in HEALTH_QUESTION_TO_OUTPUT_COLUMNS.items():
                    status_column = column_info["status_column"]
                    method_column = column_info["method_column"]

                    row[status_column] = UNDER_5_NOT_IN_SOURCE
                    row[method_column] = NAPP

                    method_counts_by_question[health_question][NAPP] += 1
                    response_counts_by_question[health_question][UNDER_5_NOT_IN_SOURCE] += 1

                under_5_written += 1

            else:
                demo_key = make_agent_health_demo_key(row)
                row["health_assignment_source"] = "AGE_BASED_HEALTH_TARGET"

                for health_question, column_info in HEALTH_QUESTION_TO_OUTPUT_COLUMNS.items():
                    status_column = column_info["status_column"]
                    method_column = column_info["method_column"]

                    response_category = None
                    assignment_method = None

                    exact_queue = exact_queues[health_question][demo_key]

                    if len(exact_queue) > 0:
                        response_category = exact_queue.popleft()
                        assignment_method = "EXACT_AGE_CELL"

                    elif len(fallback_queues[health_question]) > 0:
                        response_category = fallback_queues[health_question].popleft()
                        assignment_method = "FALLBACK_OTHER_AGE_CELL"
                        fallback_used_by_question[health_question] += 1

                    else:
                        response_category = "MISSING_HEALTH_ASSIGNMENT"
                        assignment_method = "FAILED"
                        missing_health_assignment_by_question[health_question] += 1

                    row[status_column] = response_category
                    row[method_column] = assignment_method

                    method_counts_by_question[health_question][assignment_method] += 1
                    response_counts_by_question[health_question][response_category] += 1

                    full_key = make_full_health_key(
                        county_name=row["county_name"],
                        settlement_type=row["settlement_type"],
                        health_age_group=map_exact_age_to_health_age_group(exact_age),
                        health_question=health_question,
                        health_response_category=response_category,
                    )

                    assigned_counts_by_full_key[full_key] += 1

                    health_by_employment_status_counts[
                        (
                            health_question,
                            employment_status,
                            response_category,
                        )
                    ] += 1

                    health_by_school_status_counts[
                        (
                            health_question,
                            school_status,
                            response_category,
                        )
                    ] += 1

                age_5plus_written += 1

            writer.writerow(row)
            total_written += 1

            if total_written % 1_000_000 == 0:
                print(f"Kiírt agentek health attribútumokkal: {total_written:,}")

    return {
        "total_written": total_written,
        "under_5_written": under_5_written,
        "age_5plus_written": age_5plus_written,
        "assigned_counts_by_full_key": dict(assigned_counts_by_full_key),
        "method_counts_by_question": {
            question: dict(counter)
            for question, counter in method_counts_by_question.items()
        },
        "response_counts_by_question": {
            question: dict(counter)
            for question, counter in response_counts_by_question.items()
        },
        "health_by_employment_status_counts": dict(health_by_employment_status_counts),
        "health_by_school_status_counts": dict(health_by_school_status_counts),
        "fallback_used_by_question": dict(fallback_used_by_question),
        "missing_health_assignment_by_question": dict(missing_health_assignment_by_question),
    }


# ============================================================
# 7. VALIDÁCIÓK
# ============================================================

def create_assignment_plan(
    targets: pd.DataFrame,
    candidate_counts_by_demo_key: dict,
    queue_result: dict,
) -> pd.DataFrame:
    target_demo_totals = (
        targets
        .groupby(
            [
                "health_question",
                "county_name",
                "settlement_type",
                "health_age_group",
            ],
            dropna=False,
        )["target_count"]
        .sum()
        .reset_index()
    )

    rows = []

    for _, row in target_demo_totals.iterrows():
        demo_key = make_health_demo_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            health_age_group=row["health_age_group"],
        )

        health_question = normalize_text(row["health_question"])
        target_count = convert_count_to_integer(row["target_count"])
        candidate_count = candidate_counts_by_demo_key.get(demo_key, 0)

        rows.append({
            "health_question": health_question,
            "county_name": demo_key[0],
            "settlement_type": demo_key[1],
            "health_age_group": demo_key[2],
            "target_count": target_count,
            "candidate_agent_count": candidate_count,
            "difference_candidate_minus_target": candidate_count - target_count,
            "absolute_difference": abs(candidate_count - target_count),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["absolute_difference", "target_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def create_target_cell_validation(
    targets: pd.DataFrame,
    assigned_counts_by_full_key: dict,
) -> pd.DataFrame:
    target_lookup = {}

    for _, row in targets.iterrows():
        full_key = make_full_health_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            health_age_group=row["health_age_group"],
            health_question=row["health_question"],
            health_response_category=row["health_response_category"],
        )

        target_lookup[full_key] = target_lookup.get(full_key, 0) + convert_count_to_integer(
            row["target_count"]
        )

    all_keys = set(target_lookup.keys()) | set(assigned_counts_by_full_key.keys())

    rows = []

    for key in all_keys:
        target_count = target_lookup.get(key, 0)
        assigned_count = assigned_counts_by_full_key.get(key, 0)

        rows.append({
            "county_name": key[0],
            "settlement_type": key[1],
            "health_age_group": key[2],
            "health_question": key[3],
            "health_response_category": key[4],
            "target_count": target_count,
            "assigned_count": assigned_count,
            "difference_assigned_minus_target": assigned_count - target_count,
            "absolute_difference": abs(assigned_count - target_count),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["absolute_difference", "target_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def write_health_by_employment_status(write_result: dict) -> None:
    rows = []

    for key, count in write_result["health_by_employment_status_counts"].items():
        health_question, employment_status, response_category = key

        rows.append({
            "health_question": health_question,
            "employment_layer_status": employment_status,
            "health_response_category": response_category,
            "agent_count": count,
        })

    (
        pd.DataFrame(rows)
        .sort_values(["health_question", "agent_count"], ascending=[True, False])
        .to_csv(
            HEALTH_BY_EMPLOYMENT_STATUS_CSV,
            index=False,
            encoding="utf-8-sig",
        )
    )


def write_health_by_school_status(write_result: dict) -> None:
    rows = []

    for key, count in write_result["health_by_school_status_counts"].items():
        health_question, school_status, response_category = key

        rows.append({
            "health_question": health_question,
            "school_attendance_status": school_status,
            "health_response_category": response_category,
            "agent_count": count,
        })

    (
        pd.DataFrame(rows)
        .sort_values(["health_question", "agent_count"], ascending=[True, False])
        .to_csv(
            HEALTH_BY_SCHOOL_STATUS_CSV,
            index=False,
            encoding="utf-8-sig",
        )
    )


def write_summary(
    targets: pd.DataFrame,
    current_count_result: dict,
    queue_result: dict,
    write_result: dict,
    validation: pd.DataFrame,
) -> None:
    rows = [
        {
            "metric": "output_agents_total",
            "value": write_result["total_written"],
            "note": "Output agentfájl sorainak száma.",
        },
        {
            "metric": "age_5plus_agents_total",
            "value": write_result["age_5plus_written"],
            "note": "5 éves és idősebb agentek, akik health attribútumot kaptak.",
        },
        {
            "metric": "under_5_agents_total",
            "value": write_result["under_5_written"],
            "note": "5 év alatti agentek, akik nem szerepelnek a health forrástáblában.",
        },
    ]

    target_total_by_question = (
        targets
        .groupby("health_question", dropna=False)["target_count"]
        .sum()
        .reset_index()
    )

    for _, row in target_total_by_question.iterrows():
        health_question = normalize_text(row["health_question"])
        target_total = convert_count_to_integer(row["target_count"])

        rows.append({
            "metric": f"target_total__{health_question}",
            "value": target_total,
            "note": "Health question célösszege.",
        })

        rows.append({
            "metric": f"assigned_5plus_total__{health_question}",
            "value": write_result["age_5plus_written"],
            "note": "Ehhez a health questionhöz kiosztott 5+ agentek száma.",
        })

        rows.append({
            "metric": f"difference_assigned_minus_target__{health_question}",
            "value": write_result["age_5plus_written"] - target_total,
            "note": "Ideálisan 0.",
        })

        method_counts = write_result["method_counts_by_question"].get(health_question, {})

        for method, count in method_counts.items():
            rows.append({
                "metric": f"assignment_method__{health_question}__{method}",
                "value": count,
                "note": "Assignment method count for one health question.",
            })

        rows.append({
            "metric": f"local_shortage_before_fallback__{health_question}",
            "value": queue_result["local_shortage_by_question"].get(health_question, 0),
            "note": "Cellaszinten target > candidate hiányok összege fallback előtt.",
        })

        rows.append({
            "metric": f"local_surplus_agents_before_fallback__{health_question}",
            "value": queue_result["local_surplus_agents_by_question"].get(health_question, 0),
            "note": "Cellaszinten candidate > target többletek összege fallback előtt.",
        })

        rows.append({
            "metric": f"missing_assignment__{health_question}",
            "value": write_result["missing_health_assignment_by_question"].get(health_question, 0),
            "note": "Ideálisan 0.",
        })

    validation_by_question = (
        validation
        .groupby("health_question", dropna=False)
        .agg(
            validation_sum_absolute_difference=("absolute_difference", "sum"),
            validation_max_absolute_difference=("absolute_difference", "max"),
            validation_nonzero_rows=("absolute_difference", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )

    for _, row in validation_by_question.iterrows():
        health_question = normalize_text(row["health_question"])

        rows.append({
            "metric": f"validation_sum_absolute_difference__{health_question}",
            "value": int(row["validation_sum_absolute_difference"]),
            "note": "Target cella vs assigned cella abszolút eltérések összege.",
        })

        rows.append({
            "metric": f"validation_max_absolute_difference__{health_question}",
            "value": int(row["validation_max_absolute_difference"]),
            "note": "Legnagyobb cellaeltérés.",
        })

        rows.append({
            "metric": f"validation_nonzero_rows__{health_question}",
            "value": int(row["validation_nonzero_rows"]),
            "note": "Nem nulla eltérésű validációs sorok száma.",
        })

    pd.DataFrame(rows).to_csv(
        HEALTH_ASSIGNMENT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script az age-based health targetek alapján három külön health attribútumot rendel az agentekhez.",
        },
        {
            "order": 2,
            "note": "A három health attribútum: health_disability_status, health_chronic_disease_status, health_limitation_status.",
        },
        {
            "order": 3,
            "note": "A forrás target: 127_age_based_health_targets.csv.",
        },
        {
            "order": 4,
            "note": "Az assignment fő kulcsa: vármegye × településtípus × health_age_group.",
        },
        {
            "order": 5,
            "note": "Az 5 év alatti agentek UNDER_5_NOT_IN_SOURCE értéket kapnak, mert a forrástábla 5 éves kortól indul.",
        },
        {
            "order": 6,
            "note": "A sex, education és economic_activity dimenziók ebben az első verzióban nem kényszerített targetek, hanem későbbi validációs szempontok.",
        },
        {
            "order": 7,
            "note": "A 15 évesnél fiatalabb személy duplikált kategóriáját nem használjuk assignment targetként; a target kizárólag az age_group dimenzióból jön.",
        },
        {
            "order": 8,
            "note": "Ha egy county × settlement_type × age_group cellában eltér a target és a candidate agent count, fallback response assignment történik más age cellákból.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        HEALTH_ASSIGNMENT_METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 8. FŐ FUTTATÁS
# ============================================================

def run_age_based_health_assignment() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Age-based health targetek betöltése...")
    targets = load_age_based_health_targets(
        targets_csv=AGE_BASED_HEALTH_TARGETS_CSV
    )

    print("Current agentek számlálása health age cellák szerint...")
    current_count_result = count_current_agents_by_health_demo_key(
        current_agents_csv=CURRENT_AGENTS_CSV
    )

    print("Health assignment queue-k építése...")
    queue_result = build_health_assignment_queues(
        targets=targets,
        candidate_counts_by_demo_key=current_count_result["candidate_counts_by_demo_key"],
    )

    print("Assignment plan írása...")
    assignment_plan = create_assignment_plan(
        targets=targets,
        candidate_counts_by_demo_key=current_count_result["candidate_counts_by_demo_key"],
        queue_result=queue_result,
    )

    assignment_plan.to_csv(
        HEALTH_ASSIGNMENT_PLAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Agentfájl írása health attribútumokkal...")
    write_result = write_agents_with_health_status(
        current_agents_csv=CURRENT_AGENTS_CSV,
        output_agents_csv=AGENTS_WITH_HEALTH_STATUS_CSV,
        queue_result=queue_result,
    )

    print("Health target cell validation írása...")
    validation = create_target_cell_validation(
        targets=targets,
        assigned_counts_by_full_key=write_result["assigned_counts_by_full_key"],
    )

    validation.to_csv(
        HEALTH_VALIDATION_BY_TARGET_CELL_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Health × employment validáció írása...")
    write_health_by_employment_status(
        write_result=write_result
    )

    print("Health × school validáció írása...")
    write_health_by_school_status(
        write_result=write_result
    )

    print("Summary írása...")
    write_summary(
        targets=targets,
        current_count_result=current_count_result,
        queue_result=queue_result,
        write_result=write_result,
        validation=validation,
    )

    print("Method notes írása...")
    write_method_notes()

    print()
    print("Kész.")
    print(f"Output agentek: {write_result['total_written']:,}")
    print(f"5+ agentek: {write_result['age_5plus_written']:,}")
    print(f"5 év alatti agentek: {write_result['under_5_written']:,}")

    for question in HEALTH_QUESTION_TO_OUTPUT_COLUMNS:
        method_counts = write_result["method_counts_by_question"].get(question, {})
        print(f"{question}: {method_counts}")

    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_age_based_health_assignment()