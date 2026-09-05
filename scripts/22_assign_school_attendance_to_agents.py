from pathlib import Path
import csv
import heapq
import random
import re
from collections import defaultdict, deque

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

CURRENT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/domestic_workplace_assignment/"
    "99_agents_with_domestic_workplaces_no_na_placeholders.csv"
)

SCHOOL_ATTENDANCE_XLSX = Path(
    "../data/raw_hier_tables/hier_iskolaba_jaro_nepesseg.xlsx"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/school_attendance_assignment"
)

SCHOOL_ATTENDANCE_LONG_CSV = OUTPUT_FOLDER / "114_school_attendance_hier_long.csv"
SCHOOL_ASSIGNMENT_PLAN_CSV = OUTPUT_FOLDER / "115_school_attendance_assignment_plan.csv"
AGENTS_WITH_SCHOOL_ATTENDANCE_CSV = OUTPUT_FOLDER / "116_agents_with_school_attendance.csv"
SCHOOL_VALIDATION_BY_TARGET_CELL_CSV = OUTPUT_FOLDER / "117_school_attendance_validation_by_target_cell.csv"
SCHOOL_ASSIGNMENT_SUMMARY_CSV = OUTPUT_FOLDER / "118_school_attendance_assignment_summary.csv"
SCHOOL_BY_EMPLOYMENT_STATUS_CSV = OUTPUT_FOLDER / "119_school_attendance_by_employment_status.csv"
SCHOOL_ASSIGNMENT_METHOD_NOTES_CSV = OUTPUT_FOLDER / "120_school_attendance_assignment_method_notes.csv"

RANDOM_SEED = 42

# Teljes futtatáshoz None.
# Teszthez lehet pl. 100_000.
MAX_ROWS_TO_PROCESS = None

SCHOOL_STATUS_ATTENDS = "SCHOOL"
SCHOOL_STATUS_NOT_ATTENDING = "NO_SCHOOL"
NOT_APPLICABLE = "NAPP"


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
        "Főváros": "Főváros",
        "főváros": "Főváros",
        "Fováros": "Főváros",
        "fováros": "Főváros",
        "Megyei jogú város(ok)": "Megyei jogú város(ok)",
        "Egyéb város(ok)": "Egyéb város(ok)",
        "Község(ek)": "Község(ek)",
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


def normalize_schooling_type(raw_schooling_type: str) -> str:
    return normalize_text(raw_schooling_type)


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

    if text == "…":
        # A KSH jelmagyarázat szerint ez adatvédelmi okból elnyomott érték.
        # Ebben az első verzióban 0-ként kezeljük, nem imputáljuk.
        return 0

    if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")

    return int(round(float(text)))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def map_exact_age_to_school_age_group(exact_age: int) -> str:
    age = int(exact_age)

    if age < 6:
        return "UNDER_6_NOT_IN_TABLE"

    if age <= 9:
        return "6–9 éves"

    if age <= 14:
        return "10–14 éves"

    if age <= 19:
        return "15–19 éves"

    if age <= 24:
        return "20–24 éves"

    if age <= 29:
        return "25–29 éves"

    if age <= 34:
        return "30–34 éves"

    if age <= 39:
        return "35–39 éves"

    return "40 éves és idősebb"


def make_demo_key(
    county_name: str,
    settlement_type: str,
    sex: str,
    age_group: str,
) -> tuple[str, str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_sex_label(sex),
        normalize_age_group(age_group),
    )


def make_full_school_key(
    county_name: str,
    settlement_type: str,
    sex: str,
    age_group: str,
    schooling_type: str,
) -> tuple[str, str, str, str, str]:
    demo_key = make_demo_key(
        county_name=county_name,
        settlement_type=settlement_type,
        sex=sex,
        age_group=age_group,
    )

    return demo_key + (normalize_schooling_type(schooling_type),)


def make_agent_demo_key(row: dict) -> tuple[str, str, str, str]:
    exact_age = convert_count_to_integer(row["exact_age"])

    return make_demo_key(
        county_name=row["county_name"],
        settlement_type=row["settlement_type"],
        sex=row["sex"],
        age_group=map_exact_age_to_school_age_group(exact_age),
    )


# ============================================================
# 3. ISKOLÁBA JÁRÁSI TÁBLA LONG FORMÁTUMRA
# ============================================================

def load_school_attendance_table_to_long(school_attendance_xlsx: Path) -> pd.DataFrame:
    raw_table = pd.read_excel(
        school_attendance_xlsx,
        sheet_name="Adattábla",
        header=None,
    )

    first_value_column_index = 3
    first_data_row_index = 2

    county_values = raw_table.iloc[0, first_value_column_index:].ffill()
    settlement_type_values = raw_table.iloc[1, first_value_column_index:].ffill()

    geography = pd.DataFrame({
        "value_column_index": raw_table.columns[first_value_column_index:],
        "county_name": county_values.values,
        "settlement_type": settlement_type_values.values,
    })

    geography["county_name"] = geography["county_name"].apply(normalize_county_name)
    geography["settlement_type"] = geography["settlement_type"].apply(normalize_settlement_type)

    data_rows = raw_table.iloc[first_data_row_index:].copy()

    data_rows[0] = data_rows[0].ffill()
    data_rows[1] = data_rows[1].ffill()

    long_rows = []

    for row_index, row in data_rows.iterrows():
        sex = normalize_sex_label(row[0])
        age_group = normalize_age_group(row[1])
        schooling_type = normalize_schooling_type(row[2])

        if sex == "" or age_group == "" or schooling_type == "":
            continue

        for _, geography_row in geography.iterrows():
            value_column_index = geography_row["value_column_index"]

            count = convert_count_to_integer(row[value_column_index])

            if count == 0:
                continue

            long_rows.append({
                "county_name": geography_row["county_name"],
                "settlement_type": geography_row["settlement_type"],
                "sex": sex,
                "age_group": age_group,
                "schooling_type": schooling_type,
                "school_attendance_target_count": count,
            })

    long_df = pd.DataFrame(long_rows)

    grouped = (
        long_df
        .groupby(
            [
                "county_name",
                "settlement_type",
                "sex",
                "age_group",
                "schooling_type",
            ],
            dropna=False,
        )["school_attendance_target_count"]
        .sum()
        .reset_index()
    )

    return grouped


def build_school_target_lookups(school_long: pd.DataFrame) -> dict:
    full_target_lookup = {}
    demo_target_lookup = defaultdict(int)
    schooling_type_counts_by_demo = defaultdict(lambda: defaultdict(int))

    for _, row in school_long.iterrows():
        full_key = make_full_school_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            sex=row["sex"],
            age_group=row["age_group"],
            schooling_type=row["schooling_type"],
        )

        demo_key = full_key[:4]
        schooling_type = full_key[4]
        count = convert_count_to_integer(row["school_attendance_target_count"])

        full_target_lookup[full_key] = count
        demo_target_lookup[demo_key] += count
        schooling_type_counts_by_demo[demo_key][schooling_type] += count

    return {
        "full_target_lookup": full_target_lookup,
        "demo_target_lookup": dict(demo_target_lookup),
        "schooling_type_counts_by_demo": {
            demo_key: dict(counter)
            for demo_key, counter in schooling_type_counts_by_demo.items()
        },
    }


# ============================================================
# 4. EXACT SCHOOL AGENTEK KIJELÖLÉSE
# ============================================================

def select_exact_school_agent_rows(
    current_agents_csv: Path,
    demo_target_lookup: dict,
) -> dict:
    random_generator = random.Random(RANDOM_SEED)

    heaps_by_demo_key = {
        demo_key: []
        for demo_key, target_count in demo_target_lookup.items()
        if target_count > 0
    }

    candidate_count_by_demo_key = defaultdict(int)

    total_rows_seen = 0
    age_6plus_rows_seen = 0

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        required_columns = [
            "county_name",
            "settlement_type",
            "sex",
            "exact_age",
        ]

        for column_name in required_columns:
            if column_name not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop a current agent fájlban: {column_name}"
                )

        for row_number, row in enumerate(reader, start=1):
            if MAX_ROWS_TO_PROCESS is not None and total_rows_seen >= MAX_ROWS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])

            if exact_age >= 6:
                age_6plus_rows_seen += 1

                demo_key = make_agent_demo_key(row)
                candidate_count_by_demo_key[demo_key] += 1

                if demo_key in demo_target_lookup:
                    target_count = demo_target_lookup[demo_key]

                    if target_count > 0:
                        score = random_generator.random()
                        candidate = (score, row_number)
                        heap = heaps_by_demo_key[demo_key]

                        if len(heap) < target_count:
                            heapq.heappush(heap, candidate)
                        else:
                            if candidate[0] > heap[0][0]:
                                heapq.heapreplace(heap, candidate)

            total_rows_seen += 1

            if total_rows_seen % 1_000_000 == 0:
                print(f"School exact kiválasztási pass: {total_rows_seen:,} agent")

    selected_exact_rows_by_demo_key = defaultdict(set)
    selected_row_numbers = set()
    exact_assigned_count_by_demo_key = defaultdict(int)
    fallback_needed_by_demo_key = defaultdict(int)

    for demo_key, target_count in demo_target_lookup.items():
        selected_for_demo = heaps_by_demo_key.get(demo_key, [])

        for _, row_number in selected_for_demo:
            selected_row_numbers.add(row_number)
            selected_exact_rows_by_demo_key[demo_key].add(row_number)

        exact_count = len(selected_for_demo)
        exact_assigned_count_by_demo_key[demo_key] = exact_count

        if exact_count < target_count:
            fallback_needed_by_demo_key[demo_key] = target_count - exact_count

    return {
        "selected_row_numbers": selected_row_numbers,
        "selected_exact_rows_by_demo_key": {
            demo_key: set(row_numbers)
            for demo_key, row_numbers in selected_exact_rows_by_demo_key.items()
        },
        "exact_assigned_count_by_demo_key": dict(exact_assigned_count_by_demo_key),
        "fallback_needed_by_demo_key": dict(fallback_needed_by_demo_key),
        "candidate_count_by_demo_key": dict(candidate_count_by_demo_key),
        "total_rows_seen": total_rows_seen,
        "age_6plus_rows_seen": age_6plus_rows_seen,
    }


# ============================================================
# 5. TYPE QUEUE-K ÉPÍTÉSE
# ============================================================

def build_schooling_type_queues(
    schooling_type_counts_by_demo: dict,
    exact_assigned_count_by_demo_key: dict,
) -> dict:
    random_generator = random.Random(RANDOM_SEED)

    exact_type_queues_by_demo_key = {}
    fallback_type_targets = []

    for demo_key, schooling_type_counts in schooling_type_counts_by_demo.items():
        type_list = []

        for schooling_type, count in schooling_type_counts.items():
            type_list.extend([schooling_type] * int(count))

        random_generator.shuffle(type_list)

        exact_count = exact_assigned_count_by_demo_key.get(demo_key, 0)

        exact_type_queues_by_demo_key[demo_key] = deque(type_list[:exact_count])

        for schooling_type in type_list[exact_count:]:
            fallback_type_targets.append({
                "target_demo_key": demo_key,
                "schooling_type": schooling_type,
            })

    return {
        "exact_type_queues_by_demo_key": exact_type_queues_by_demo_key,
        "fallback_type_targets": fallback_type_targets,
    }


# ============================================================
# 6. FALLBACK SCHOOL AGENTEK
# ============================================================

def score_fallback_candidate(
    candidate_demo_key: tuple[str, str, str, str],
    target_demo_key: tuple[str, str, str, str],
    random_generator: random.Random,
) -> float:
    cand_county, cand_type, cand_sex, cand_age_group = candidate_demo_key
    target_county, target_type, target_sex, target_age_group = target_demo_key

    score = 0.0

    if cand_county == target_county:
        score += 1000.0

    if cand_type == target_type:
        score += 300.0

    if cand_sex == target_sex:
        score += 300.0

    if cand_age_group == target_age_group:
        score += 500.0

    score += random_generator.random()

    return score


def select_fallback_school_agent_rows(
    current_agents_csv: Path,
    already_selected_rows: set[int],
    fallback_type_targets: list[dict],
) -> dict:
    """
    Gyors fallback kijelölés.

    Az előző verzió minden fallback célhoz újraolvasta a teljes agentfájlt.
    Ez 268 fallbacknél nagyon lassú lett volna.

    Ez a verzió egyszer olvassa végig a fájlt, és a nem kiválasztott,
    6+ éves agenteket fallback queue-kba gyűjti.

    Prioritási sorrend:
    1. ugyanaz county × settlement_type × sex × age_group
    2. ugyanaz county × settlement_type × age_group
    3. ugyanaz county × age_group
    4. ugyanaz age_group
    5. bármely 6+ agent
    """

    if len(fallback_type_targets) == 0:
        return {
            "fallback_selected_rows": {},
            "fallback_assigned_count": 0,
        }

    already_selected_rows = set(already_selected_rows)

    candidates_by_exact_demo = defaultdict(deque)
    candidates_by_county_type_age = defaultdict(deque)
    candidates_by_county_age = defaultdict(deque)
    candidates_by_age = defaultdict(deque)
    any_candidate_rows = deque()

    total_rows_seen = 0
    fallback_candidate_count = 0

    print(
        f"Gyors fallback candidate pool építése "
        f"{len(fallback_type_targets):,} fallback célhoz..."
    )

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        for row_number, row in enumerate(reader, start=1):
            if MAX_ROWS_TO_PROCESS is not None and total_rows_seen >= MAX_ROWS_TO_PROCESS:
                break

            if row_number in already_selected_rows:
                total_rows_seen += 1
                continue

            exact_age = convert_count_to_integer(row["exact_age"])

            if exact_age < 6:
                total_rows_seen += 1
                continue

            demo_key = make_agent_demo_key(row)

            county_name, settlement_type, sex, age_group = demo_key

            candidates_by_exact_demo[demo_key].append(row_number)
            candidates_by_county_type_age[
                (
                    county_name,
                    settlement_type,
                    age_group,
                )
            ].append(row_number)
            candidates_by_county_age[
                (
                    county_name,
                    age_group,
                )
            ].append(row_number)
            candidates_by_age[age_group].append(row_number)
            any_candidate_rows.append(row_number)

            fallback_candidate_count += 1
            total_rows_seen += 1

            if total_rows_seen % 1_000_000 == 0:
                print(f"Fallback candidate pool pass: {total_rows_seen:,} agent")

    print(f"Fallback candidate pool kész. Candidate-ek: {fallback_candidate_count:,}")

    used_rows = set(already_selected_rows)
    fallback_selected_rows = {}

    def pop_first_unused(candidate_queue: deque) -> int | None:
        while len(candidate_queue) > 0:
            candidate_row_number = candidate_queue.popleft()

            if candidate_row_number not in used_rows:
                return candidate_row_number

        return None

    for fallback_index, fallback_target in enumerate(fallback_type_targets, start=1):
        target_demo_key = fallback_target["target_demo_key"]
        schooling_type = fallback_target["schooling_type"]

        target_county, target_type, target_sex, target_age_group = target_demo_key

        selected_row_number = None
        method = None

        # 1. Exact demo key.
        selected_row_number = pop_first_unused(
            candidates_by_exact_demo[target_demo_key]
        )

        if selected_row_number is not None:
            method = "FALLBACK_SAME_DEMO"

        # 2. Same county × settlement_type × age_group, sex relaxed.
        if selected_row_number is None:
            selected_row_number = pop_first_unused(
                candidates_by_county_type_age[
                    (
                        target_county,
                        target_type,
                        target_age_group,
                    )
                ]
            )

            if selected_row_number is not None:
                method = "FALLBACK_SAME_COUNTY_TYPE_AGE"

        # 3. Same county × age_group.
        if selected_row_number is None:
            selected_row_number = pop_first_unused(
                candidates_by_county_age[
                    (
                        target_county,
                        target_age_group,
                    )
                ]
            )

            if selected_row_number is not None:
                method = "FALLBACK_SAME_COUNTY_AGE"

        # 4. Same age_group nationally.
        if selected_row_number is None:
            selected_row_number = pop_first_unused(
                candidates_by_age[target_age_group]
            )

            if selected_row_number is not None:
                method = "FALLBACK_SAME_AGE"

        # 5. Any 6+ agent.
        if selected_row_number is None:
            selected_row_number = pop_first_unused(any_candidate_rows)

            if selected_row_number is not None:
                method = "FALLBACK_ANY_6PLUS"

        if selected_row_number is None:
            raise ValueError(
                f"Nem találtam fallback school candidate-et ehhez: {fallback_target}"
            )

        fallback_selected_rows[selected_row_number] = {
            "target_demo_key": target_demo_key,
            "schooling_type": schooling_type,
            "fallback_method": method,
        }

        used_rows.add(selected_row_number)

        print(
            f"School fallback kijelölés {fallback_index}/{len(fallback_type_targets)}: "
            f"row={selected_row_number}, target={target_demo_key}, "
            f"type={schooling_type}, method={method}"
        )

    return {
        "fallback_selected_rows": fallback_selected_rows,
        "fallback_assigned_count": len(fallback_selected_rows),
    }


# ============================================================
# 7. OUTPUT AGENTFÁJL ÍRÁSA
# ============================================================

def write_agents_with_school_attendance(
    current_agents_csv: Path,
    output_agents_csv: Path,
    selected_exact_rows_by_demo_key: dict,
    exact_type_queues_by_demo_key: dict,
    fallback_selected_rows: dict,
) -> dict:
    exact_selected_row_to_demo_key = {}

    for demo_key, row_numbers in selected_exact_rows_by_demo_key.items():
        for row_number in row_numbers:
            exact_selected_row_to_demo_key[row_number] = demo_key

    all_school_rows = set(exact_selected_row_to_demo_key.keys()) | set(fallback_selected_rows.keys())

    total_written = 0
    school_assigned_total = 0
    school_exact_total = 0
    school_fallback_total = 0
    no_school_total = 0

    school_status_counts = defaultdict(int)
    school_method_counts = defaultdict(int)
    school_type_counts = defaultdict(int)
    school_by_employment_status_counts = defaultdict(int)

    assigned_full_key_counts = defaultdict(int)

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)
        input_fieldnames = list(reader.fieldnames)

        new_columns = [
            "school_attendance_status",
            "schooling_type",
            "school_attendance_assignment_method",
            "school_attendance_source",
        ]

        output_fieldnames = input_fieldnames + [
            column for column in new_columns
            if column not in input_fieldnames
        ]

        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for row_number, row in enumerate(reader, start=1):
            if MAX_ROWS_TO_PROCESS is not None and total_written >= MAX_ROWS_TO_PROCESS:
                break

            employment_status = normalize_text(row.get("employment_layer_status", ""))

            if row_number in all_school_rows:
                row["school_attendance_status"] = SCHOOL_STATUS_ATTENDS
                row["school_attendance_source"] = "HIER_SCHOOL"

                if row_number in fallback_selected_rows:
                    schooling_type = fallback_selected_rows[row_number]["schooling_type"]
                    row["school_attendance_assignment_method"] = "FALLBACK"
                    school_fallback_total += 1

                else:
                    demo_key = exact_selected_row_to_demo_key[row_number]
                    schooling_type = exact_type_queues_by_demo_key[demo_key].popleft()
                    row["school_attendance_assignment_method"] = "EXACT"
                    school_exact_total += 1

                row["schooling_type"] = schooling_type

                agent_demo_key = make_agent_demo_key(row)
                full_key = agent_demo_key + (normalize_schooling_type(schooling_type),)
                assigned_full_key_counts[full_key] += 1

                school_assigned_total += 1

            else:
                row["school_attendance_status"] = SCHOOL_STATUS_NOT_ATTENDING
                row["schooling_type"] = NOT_APPLICABLE
                row["school_attendance_assignment_method"] = NOT_APPLICABLE
                row["school_attendance_source"] = NOT_APPLICABLE

                no_school_total += 1

            school_status = row["school_attendance_status"]
            school_method = row["school_attendance_assignment_method"]
            schooling_type = row["schooling_type"]

            school_status_counts[school_status] += 1
            school_method_counts[school_method] += 1
            school_type_counts[schooling_type] += 1

            school_by_employment_status_counts[
                (
                    employment_status,
                    school_status,
                    schooling_type,
                )
            ] += 1

            writer.writerow(row)

            total_written += 1

            if total_written % 1_000_000 == 0:
                print(f"Kiírt agentek school attendance-szel: {total_written:,}")

    return {
        "total_written": total_written,
        "school_assigned_total": school_assigned_total,
        "school_exact_total": school_exact_total,
        "school_fallback_total": school_fallback_total,
        "no_school_total": no_school_total,
        "school_status_counts": dict(school_status_counts),
        "school_method_counts": dict(school_method_counts),
        "school_type_counts": dict(school_type_counts),
        "school_by_employment_status_counts": dict(school_by_employment_status_counts),
        "assigned_full_key_counts": dict(assigned_full_key_counts),
    }


# ============================================================
# 8. VALIDÁCIÓK
# ============================================================

def create_assignment_plan(
    school_long: pd.DataFrame,
    exact_selection_result: dict,
) -> pd.DataFrame:
    demo_target_lookup = (
        school_long
        .groupby(["county_name", "settlement_type", "sex", "age_group"], dropna=False)
        ["school_attendance_target_count"]
        .sum()
        .reset_index()
    )

    rows = []

    for _, row in demo_target_lookup.iterrows():
        demo_key = make_demo_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            sex=row["sex"],
            age_group=row["age_group"],
        )

        target_count = convert_count_to_integer(row["school_attendance_target_count"])
        candidate_count = exact_selection_result["candidate_count_by_demo_key"].get(demo_key, 0)
        exact_assigned = exact_selection_result["exact_assigned_count_by_demo_key"].get(demo_key, 0)
        fallback_needed = exact_selection_result["fallback_needed_by_demo_key"].get(demo_key, 0)

        rows.append({
            "county_name": demo_key[0],
            "settlement_type": demo_key[1],
            "sex": demo_key[2],
            "age_group": demo_key[3],
            "school_attendance_target_count": target_count,
            "candidate_agent_count": candidate_count,
            "exact_assigned_count": exact_assigned,
            "fallback_needed_count": fallback_needed,
            "difference_exact_assigned_minus_target": exact_assigned - target_count,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["fallback_needed_count", "school_attendance_target_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def create_target_cell_validation(
    full_target_lookup: dict,
    assigned_full_key_counts: dict,
) -> pd.DataFrame:
    all_keys = set(full_target_lookup.keys()) | set(assigned_full_key_counts.keys())

    rows = []

    for key in all_keys:
        target_count = full_target_lookup.get(key, 0)
        assigned_count = assigned_full_key_counts.get(key, 0)

        rows.append({
            "county_name": key[0],
            "settlement_type": key[1],
            "sex": key[2],
            "age_group": key[3],
            "schooling_type": key[4],
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


def write_school_by_employment_status(write_result: dict) -> None:
    rows = []

    for key, count in write_result["school_by_employment_status_counts"].items():
        employment_status, school_status, schooling_type = key

        rows.append({
            "employment_layer_status": employment_status,
            "school_attendance_status": school_status,
            "schooling_type": schooling_type,
            "agent_count": count,
        })

    (
        pd.DataFrame(rows)
        .sort_values(["school_attendance_status", "agent_count"], ascending=[True, False])
        .to_csv(
            SCHOOL_BY_EMPLOYMENT_STATUS_CSV,
            index=False,
            encoding="utf-8-sig",
        )
    )


def write_summary(
    school_long: pd.DataFrame,
    exact_selection_result: dict,
    fallback_result: dict,
    write_result: dict,
    validation: pd.DataFrame,
) -> None:
    school_target_total = int(school_long["school_attendance_target_count"].sum())

    rows = [
        {
            "metric": "output_agents_total",
            "value": write_result["total_written"],
            "note": "Output agentfájl sorainak száma.",
        },
        {
            "metric": "school_attendance_target_total",
            "value": school_target_total,
            "note": "Hier iskolába járó népesség célösszege.",
        },
        {
            "metric": "school_attendance_assigned_total",
            "value": write_result["school_assigned_total"],
            "note": "SCHOOL státuszt kapott agentek száma.",
        },
        {
            "metric": "school_attendance_difference_assigned_minus_target",
            "value": write_result["school_assigned_total"] - school_target_total,
            "note": "Ideálisan 0.",
        },
        {
            "metric": "school_exact_assigned_total",
            "value": write_result["school_exact_total"],
            "note": "Pontos vármegye × településtípus × nem × korcsoport kulcson assigned.",
        },
        {
            "metric": "school_fallback_assigned_total",
            "value": write_result["school_fallback_total"],
            "note": "Fallbackkal assigned.",
        },
        {
            "metric": "school_exact_assignment_share",
            "value": safe_divide(write_result["school_exact_total"], write_result["school_assigned_total"]),
            "note": "Exact assignment aránya.",
        },
        {
            "metric": "school_fallback_assignment_share",
            "value": safe_divide(write_result["school_fallback_total"], write_result["school_assigned_total"]),
            "note": "Fallback assignment aránya.",
        },
        {
            "metric": "no_school_agents_total",
            "value": write_result["no_school_total"],
            "note": "NO_SCHOOL státuszú agentek.",
        },
        {
            "metric": "fallback_needed_before_fallback_assignment",
            "value": sum(exact_selection_result["fallback_needed_by_demo_key"].values()),
            "note": "Exact kulcson nem fedett targetek száma fallback előtt.",
        },
        {
            "metric": "fallback_assigned_total",
            "value": fallback_result["fallback_assigned_count"],
            "note": "Fallbackként ténylegesen assigned agentek száma.",
        },
        {
            "metric": "validation_sum_absolute_difference",
            "value": int(validation["absolute_difference"].sum()),
            "note": "Target cella vs assigned cella abszolút eltérések összege.",
        },
        {
            "metric": "validation_max_absolute_difference",
            "value": int(validation["absolute_difference"].max()),
            "note": "Legnagyobb target cella vs assigned cella eltérés.",
        },
    ]

    pd.DataFrame(rows).to_csv(
        SCHOOL_ASSIGNMENT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script az iskolába járó népességet assignolja az aktuális agentfájlhoz.",
        },
        {
            "order": 2,
            "note": "A forrástábla: hier_iskolaba_jaro_nepesseg.xlsx.",
        },
        {
            "order": 3,
            "note": "A target dimenziói: vármegye × településtípus × nem × korcsoport × képzéstípus.",
        },
        {
            "order": 4,
            "note": "Az assignment nem tiltja ki a foglalkoztatottakat, mert reális, hogy valaki dolgozik és mellette képzésben vesz részt.",
        },
        {
            "order": 5,
            "note": "Az elsődleges exact kulcs: vármegye × településtípus × nem × korcsoport. A képzéstípus ezen belül kerül kiosztásra a hierarchikus célmegoszlás alapján.",
        },
        {
            "order": 6,
            "note": "A 6 év alatti agentek nem kaphatnak SCHOOL státuszt, mert a forrástábla a 6 éves és idősebb iskolába járó népességet tartalmazza.",
        },
        {
            "order": 7,
            "note": "A nem iskolába járó agentek NO_SCHOOL státuszt és NAPP schooling_type értéket kapnak.",
        },
        {
            "order": 8,
            "note": "A validáció target cella szinten történik: vármegye × településtípus × nem × korcsoport × képzéstípus.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        SCHOOL_ASSIGNMENT_METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 9. FŐ FUTTATÁS
# ============================================================

def run_school_attendance_assignment() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Iskolába járási hier tábla long formátumra alakítása...")
    school_long = load_school_attendance_table_to_long(
        school_attendance_xlsx=SCHOOL_ATTENDANCE_XLSX
    )

    school_long.to_csv(
        SCHOOL_ATTENDANCE_LONG_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    target_lookups = build_school_target_lookups(
        school_long=school_long
    )

    print("Exact school attendance agentek kijelölése...")
    exact_selection_result = select_exact_school_agent_rows(
        current_agents_csv=CURRENT_AGENTS_CSV,
        demo_target_lookup=target_lookups["demo_target_lookup"],
    )

    print("Schooling type queue-k építése...")
    type_queue_result = build_schooling_type_queues(
        schooling_type_counts_by_demo=target_lookups["schooling_type_counts_by_demo"],
        exact_assigned_count_by_demo_key=exact_selection_result["exact_assigned_count_by_demo_key"],
    )

    print("Fallback school attendance agentek kijelölése...")
    fallback_result = select_fallback_school_agent_rows(
        current_agents_csv=CURRENT_AGENTS_CSV,
        already_selected_rows=exact_selection_result["selected_row_numbers"],
        fallback_type_targets=type_queue_result["fallback_type_targets"],
    )

    print("Agentfájl írása school attendance mezőkkel...")
    write_result = write_agents_with_school_attendance(
        current_agents_csv=CURRENT_AGENTS_CSV,
        output_agents_csv=AGENTS_WITH_SCHOOL_ATTENDANCE_CSV,
        selected_exact_rows_by_demo_key=exact_selection_result["selected_exact_rows_by_demo_key"],
        exact_type_queues_by_demo_key=type_queue_result["exact_type_queues_by_demo_key"],
        fallback_selected_rows=fallback_result["fallback_selected_rows"],
    )

    print("Assignment plan írása...")
    assignment_plan = create_assignment_plan(
        school_long=school_long,
        exact_selection_result=exact_selection_result,
    )

    assignment_plan.to_csv(
        SCHOOL_ASSIGNMENT_PLAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Target cell validation írása...")
    validation = create_target_cell_validation(
        full_target_lookup=target_lookups["full_target_lookup"],
        assigned_full_key_counts=write_result["assigned_full_key_counts"],
    )

    validation.to_csv(
        SCHOOL_VALIDATION_BY_TARGET_CELL_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Employment × school státusz validáció írása...")
    write_school_by_employment_status(
        write_result=write_result
    )

    print("Summary írása...")
    write_summary(
        school_long=school_long,
        exact_selection_result=exact_selection_result,
        fallback_result=fallback_result,
        write_result=write_result,
        validation=validation,
    )

    print("Method notes írása...")
    write_method_notes()

    print()
    print("Kész.")
    print(f"Output agentek: {write_result['total_written']:,}")
    print(f"School target: {int(school_long['school_attendance_target_count'].sum()):,}")
    print(f"School assigned: {write_result['school_assigned_total']:,}")
    print(f"Exact school assigned: {write_result['school_exact_total']:,}")
    print(f"Fallback school assigned: {write_result['school_fallback_total']:,}")
    print(f"No school: {write_result['no_school_total']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_school_attendance_assignment()