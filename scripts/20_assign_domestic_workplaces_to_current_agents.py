from pathlib import Path
import csv
import re
from collections import defaultdict, deque

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

CURRENT_COMPACT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/foreign_workplace_assignment/"
    "95_agents_with_compact_foreign_domestic_workplace_status.csv"
)

PREVIOUS_WORKER_WORKPLACE_CSV = Path(
    "../data/raw_reference/agents_with_workplaces.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/domestic_workplace_assignment"
)

AGENTS_WITH_DOMESTIC_WORKPLACES_CSV = (
    OUTPUT_FOLDER / "99_agents_with_domestic_workplaces.csv"
)

DOMESTIC_ASSIGNMENT_VALIDATION_CSV = (
    OUTPUT_FOLDER / "100_domestic_assignment_validation.csv"
)

DOMESTIC_ASSIGNMENT_SUMMARY_CSV = (
    OUTPUT_FOLDER / "101_domestic_assignment_summary.csv"
)

DOMESTIC_ASSIGNMENT_UNUSED_OLD_RECORDS_CSV = (
    OUTPUT_FOLDER / "102_unused_old_worker_records_summary.csv"
)

DOMESTIC_ASSIGNMENT_METHOD_NOTES_CSV = (
    OUTPUT_FOLDER / "103_domestic_assignment_method_notes.csv"
)

MAX_ROWS_TO_PROCESS = None

DOMESTIC_PENDING_STATUS = "EMP_DOM_PENDING"
DOMESTIC_FINAL_STATUS = "EMP_DOM"
FOREIGN_STATUS = "EMP_FOREIGN"
NOT_EMP_STATUS = "NOT_EMP"


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

def find_first_existing_column(fieldnames: list[str], possible_column_names: list[str]) -> str | None:
    """
    Visszaadja az első létező oszlopnevet a lehetséges nevek közül.
    Régi pipeline fájloknál hasznos, ahol ugyanaz az információ
    többféle oszlopnéven szerepelhet.
    """
    for column_name in possible_column_names:
        if column_name in fieldnames:
            return column_name

    return None


def require_first_existing_column(
    fieldnames: list[str],
    possible_column_names: list[str],
    logical_column_name: str,
) -> str:
    existing_column = find_first_existing_column(
        fieldnames=fieldnames,
        possible_column_names=possible_column_names,
    )

    if existing_column is None:
        raise ValueError(
            f"Hiányzó oszlop a régi agents_with_workplaces fájlban ehhez: {logical_column_name}. "
            f"Elfogadott nevek: {possible_column_names}"
        )

    return existing_column

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


def make_match_key(
    county_name: str,
    sex: str,
    age_group: str,
    education: str,
) -> tuple[str, str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_sex_label(sex),
        normalize_age_group(age_group),
        normalize_education(education),
    )


def make_current_agent_match_key(row: dict) -> tuple[str, str, str, str]:
    exact_age = convert_count_to_integer(row["exact_age"])

    return make_match_key(
        county_name=row["county_name"],
        sex=row["sex"],
        age_group=map_exact_age_to_worker_age_group(exact_age),
        education=row["education_level_calibrated"],
    )


def make_old_worker_match_key(row: dict) -> tuple[str, str, str, str]:
    return make_match_key(
        county_name=row["county"],
        sex=row["gender"],
        age_group=row["age_group"],
        education=row["education"],
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# 3. CURRENT DOMESTIC TARGETEK SZÁMLÁLÁSA
# ============================================================

def count_current_domestic_pending_agents(current_agents_csv: Path) -> dict:
    total_rows = 0
    domestic_pending_total = 0
    foreign_total = 0
    not_employed_total = 0

    domestic_counts_by_key = defaultdict(int)
    status_counts = defaultdict(int)

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = [
            "employment_layer_status",
            "county_name",
            "sex",
            "exact_age",
            "education_level_calibrated",
        ]

        for column_name in required_columns:
            if column_name not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop a current agent fájlban: {column_name}"
                )

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            status = normalize_text(row["employment_layer_status"])
            status_counts[status] += 1

            if status == DOMESTIC_PENDING_STATUS:
                key = make_current_agent_match_key(row)
                domestic_counts_by_key[key] += 1
                domestic_pending_total += 1

            elif status == FOREIGN_STATUS:
                foreign_total += 1

            elif status == NOT_EMP_STATUS:
                not_employed_total += 1

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"Current agent count pass: {total_rows:,}")

    return {
        "total_rows": total_rows,
        "domestic_pending_total": domestic_pending_total,
        "foreign_total": foreign_total,
        "not_employed_total": not_employed_total,
        "domestic_counts_by_key": dict(domestic_counts_by_key),
        "status_counts": dict(status_counts),
    }


# ============================================================
# 4. OLD WORKER RECORD STRUKTÚRA
# ============================================================

def old_worker_row_to_assignment_record(row: dict) -> dict:
    """
    Csak azokat az oszlopokat tartjuk meg, amelyeket ténylegesen át akarunk vinni.
    Így kisebb a memóriahasználat, mintha teljes sort tárolnánk.
    """
    return {
        "previous_worker_record_id": normalize_text(row.get("agent_id", "")),
        "old_worker_residence_county": normalize_county_name(row.get("county", "")),
        "old_worker_sector_code": normalize_text(row.get("sector_code", "")),
        "old_worker_gender": normalize_text(row.get("gender", "")),
        "old_worker_age_group": normalize_age_group(row.get("age_group", "")),
        "old_worker_education": normalize_education(row.get("education", "")),
        "target_work_county": normalize_county_name(row.get("target_work_county", "")),
        "final_work_county": normalize_county_name(row.get("final_work_county", "")),
        "old_worker_used_fallback": normalize_text(row.get("used_fallback", "")),
        "old_worker_fallback_level": normalize_text(row.get("fallback_level", "")),
        "old_worker_used_teaor_fallback": normalize_text(row.get("used_teaor_fallback", "")),
        "workplace_id": normalize_text(row.get("workplace_id", "")),
        "workplace_county": normalize_county_name(row.get("workplace_county", "")),
        "workplace_teaor_code": normalize_text(row.get("workplace_teaor_code", "")),
        "workplace_settlement": normalize_text(row.get("workplace_settlement", "")),
        "workplace_settlement_type": normalize_text(row.get("workplace_settlement_type", "")),
        "workplace_size": convert_count_to_integer(row.get("workplace_size", 0)),
    }


# ============================================================
# 5. OLD WORKER POOL BETÖLTÉSE
# ============================================================

def build_old_worker_assignment_pools(
    previous_workers_csv: Path,
    domestic_targets_by_key: dict,
) -> dict:
    """
    A régi worker rekordokat két részre bontja:

    1. exact_pools_by_key:
       legfeljebb annyi régi rekordot tart meg egy exact kulcshoz,
       ahány current domestic agent van ugyanazon a kulcson.

    2. fallback_pool:
       minden többlet régi rekord ide kerül.
       Ezeket használjuk, ha valamely current domestic kulcshoz nincs elég exact régi rekord.

    A régi worker pool nagyobb, mint a domestic target, ezért a végén marad unused rekord.
    """
    exact_pools_by_key = defaultdict(deque)
    exact_selected_count_by_key = defaultdict(int)

    fallback_pool = deque()

    old_total_rows = 0
    old_missing_workplace_id = 0

    old_counts_by_key = defaultdict(int)

    with previous_workers_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        fieldnames = list(reader.fieldnames)

        agent_id_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["agent_id", "worker_id", "old_agent_id"],
            logical_column_name="agent_id",
        )

        county_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["county", "residence_county", "agent_county"],
            logical_column_name="county",
        )

        gender_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["gender", "sex"],
            logical_column_name="gender",
        )

        age_group_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["age_group"],
            logical_column_name="age_group",
        )

        education_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["education", "education_level"],
            logical_column_name="education",
        )

        sector_code_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=[
                "sector_code",
                "teor_code",
                "teaor_code",
                "workplace_teaor_code",
            ],
            logical_column_name="sector_code / TEÁOR-like source code",
        )

        workplace_id_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["workplace_id"],
            logical_column_name="workplace_id",
        )

        workplace_county_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["workplace_county"],
            logical_column_name="workplace_county",
        )

        workplace_teaor_code_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["workplace_teaor_code", "teaor_code", "teor_code"],
            logical_column_name="workplace_teaor_code",
        )

        workplace_settlement_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["workplace_settlement"],
            logical_column_name="workplace_settlement",
        )

        workplace_settlement_type_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["workplace_settlement_type"],
            logical_column_name="workplace_settlement_type",
        )

        workplace_size_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["workplace_size"],
            logical_column_name="workplace_size",
        )

        target_work_county_column = find_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["target_work_county"],
        )

        final_work_county_column = find_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["final_work_county"],
        )

        used_fallback_column = find_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["used_fallback"],
        )

        fallback_level_column = find_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["fallback_level"],
        )

        used_teaor_fallback_column = find_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["used_teaor_fallback"],
        )

        print("Régi worker fájlban felismert oszlopok:")
        print(f"  agent_id: {agent_id_column}")
        print(f"  county: {county_column}")
        print(f"  gender: {gender_column}")
        print(f"  age_group: {age_group_column}")
        print(f"  education: {education_column}")
        print(f"  sector_code: {sector_code_column}")
        print(f"  workplace_teaor_code: {workplace_teaor_code_column}")

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and old_total_rows >= MAX_ROWS_TO_PROCESS:
                break

            key = make_match_key(
                county_name=row[county_column],
                sex=row[gender_column],
                age_group=row[age_group_column],
                education=row[education_column],
            )
            old_counts_by_key[key] += 1

            assignment_record = {
                "previous_worker_record_id": normalize_text(row.get(agent_id_column, "")),
                "old_worker_residence_county": normalize_county_name(row.get(county_column, "")),
                "old_worker_sector_code": normalize_text(row.get(sector_code_column, "")),
                "old_worker_gender": normalize_text(row.get(gender_column, "")),
                "old_worker_age_group": normalize_age_group(row.get(age_group_column, "")),
                "old_worker_education": normalize_education(row.get(education_column, "")),

                "target_work_county": normalize_county_name(
                    row.get(target_work_county_column, "")
                    if target_work_county_column is not None
                    else ""
                ),
                "final_work_county": normalize_county_name(
                    row.get(final_work_county_column, "")
                    if final_work_county_column is not None
                    else ""
                ),
                "old_worker_used_fallback": normalize_text(
                    row.get(used_fallback_column, "")
                    if used_fallback_column is not None
                    else ""
                ),
                "old_worker_fallback_level": normalize_text(
                    row.get(fallback_level_column, "")
                    if fallback_level_column is not None
                    else ""
                ),
                "old_worker_used_teaor_fallback": normalize_text(
                    row.get(used_teaor_fallback_column, "")
                    if used_teaor_fallback_column is not None
                    else ""
                ),

                "workplace_id": normalize_text(row.get(workplace_id_column, "")),
                "workplace_county": normalize_county_name(row.get(workplace_county_column, "")),
                "workplace_teaor_code": normalize_text(row.get(workplace_teaor_code_column, "")),
                "workplace_settlement": normalize_text(row.get(workplace_settlement_column, "")),
                "workplace_settlement_type": normalize_text(row.get(workplace_settlement_type_column, "")),
                "workplace_size": convert_count_to_integer(row.get(workplace_size_column, 0)),
            }

            if assignment_record["workplace_id"] == "":
                old_missing_workplace_id += 1

            current_target_for_key = domestic_targets_by_key.get(key, 0)

            if exact_selected_count_by_key[key] < current_target_for_key:
                exact_pools_by_key[key].append(assignment_record)
                exact_selected_count_by_key[key] += 1
            else:
                fallback_pool.append(assignment_record)

            old_total_rows += 1

            if old_total_rows % 1_000_000 == 0:
                print(f"Old worker pool betöltve: {old_total_rows:,}")

    return {
        "exact_pools_by_key": exact_pools_by_key,
        "fallback_pool": fallback_pool,
        "old_total_rows": old_total_rows,
        "old_missing_workplace_id": old_missing_workplace_id,
        "old_counts_by_key": dict(old_counts_by_key),
        "exact_selected_count_by_key": dict(exact_selected_count_by_key),
    }


# ============================================================
# 6. ASSIGNMENT RECORD ALKALMAZÁSA
# ============================================================

def set_na_old_worker_columns(row: dict) -> None:
    """
    Nem domestic agenteknél a régi worker oszlopok nem alkalmazhatóak.

    Fontos:
    Nem 'NA'-t használunk, mert pandas sokszor missing value-ként értelmezi.
    Ehelyett 'NAPP' = not applicable placeholder.
    """
    row["domestic_worker_assignment_method"] = "NAPP"
    row["old_worker_residence_county"] = "NAPP"
    row["old_worker_sector_code"] = "NAPP"
    row["old_worker_gender"] = "NAPP"
    row["old_worker_age_group"] = "NAPP"
    row["old_worker_education"] = "NAPP"
    row["target_work_county"] = "NAPP"
    row["final_work_county"] = "NAPP"
    row["old_worker_used_fallback"] = "NAPP"
    row["old_worker_fallback_level"] = "NAPP"
    row["old_worker_used_teaor_fallback"] = "NAPP"


def apply_domestic_assignment(row: dict, assignment_record: dict, method: str) -> None:
    row["employment_layer_status"] = DOMESTIC_FINAL_STATUS
    row["workplace_assignment_type"] = "DOMESTIC"
    row["workplace_assignment_source"] = "OLD_WORKER"
    row["domestic_worker_assignment_method"] = method

    row["workplace_country"] = "HU"
    row["workplace_id"] = assignment_record["workplace_id"]
    row["workplace_county"] = assignment_record["workplace_county"]
    row["workplace_settlement"] = assignment_record["workplace_settlement"]
    row["workplace_settlement_type"] = assignment_record["workplace_settlement_type"]
    row["workplace_teaor_code"] = assignment_record["workplace_teaor_code"]
    row["workplace_size"] = assignment_record["workplace_size"]

    row["previous_worker_record_id"] = assignment_record["previous_worker_record_id"]
    row["old_worker_residence_county"] = assignment_record["old_worker_residence_county"]
    row["old_worker_sector_code"] = assignment_record["old_worker_sector_code"]
    row["old_worker_gender"] = assignment_record["old_worker_gender"]
    row["old_worker_age_group"] = assignment_record["old_worker_age_group"]
    row["old_worker_education"] = assignment_record["old_worker_education"]
    row["target_work_county"] = assignment_record["target_work_county"]
    row["final_work_county"] = assignment_record["final_work_county"]
    row["old_worker_used_fallback"] = assignment_record["old_worker_used_fallback"]
    row["old_worker_fallback_level"] = assignment_record["old_worker_fallback_level"]
    row["old_worker_used_teaor_fallback"] = assignment_record["old_worker_used_teaor_fallback"]


# ============================================================
# 7. OUTPUT AGENTFÁJL ÍRÁSA
# ============================================================

def write_agents_with_domestic_workplaces(
    current_agents_csv: Path,
    output_agents_csv: Path,
    old_worker_pools: dict,
) -> dict:
    exact_pools_by_key = old_worker_pools["exact_pools_by_key"]
    fallback_pool = old_worker_pools["fallback_pool"]

    total_written = 0

    status_counts = defaultdict(int)
    assignment_type_counts = defaultdict(int)
    domestic_method_counts = defaultdict(int)
    workplace_country_counts = defaultdict(int)

    domestic_assigned_total = 0
    domestic_exact_assigned = 0
    domestic_fallback_assigned = 0

    missing_domestic_assignment = 0
    empty_workplace_id_domestic = 0

    used_old_worker_records = 0

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)
        input_fieldnames = list(reader.fieldnames)

        new_columns = [
            "domestic_worker_assignment_method",
            "old_worker_residence_county",
            "old_worker_sector_code",
            "old_worker_gender",
            "old_worker_age_group",
            "old_worker_education",
            "target_work_county",
            "final_work_county",
            "old_worker_used_fallback",
            "old_worker_fallback_level",
            "old_worker_used_teaor_fallback",
        ]

        output_fieldnames = input_fieldnames + [
            column for column in new_columns
            if column not in input_fieldnames
        ]

        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_written >= MAX_ROWS_TO_PROCESS:
                break

            status = normalize_text(row["employment_layer_status"])

            if status == DOMESTIC_PENDING_STATUS:
                key = make_current_agent_match_key(row)

                if len(exact_pools_by_key[key]) > 0:
                    assignment_record = exact_pools_by_key[key].popleft()
                    apply_domestic_assignment(
                        row=row,
                        assignment_record=assignment_record,
                        method="EXACT"
                    )
                    domestic_exact_assigned += 1
                    used_old_worker_records += 1

                elif len(fallback_pool) > 0:
                    assignment_record = fallback_pool.popleft()
                    apply_domestic_assignment(
                        row=row,
                        assignment_record=assignment_record,
                        method="FALLBACK"
                    )
                    domestic_fallback_assigned += 1
                    used_old_worker_records += 1

                else:
                    row["employment_layer_status"] = "MISSING_WP"
                    row["workplace_assignment_type"] = "MISSING"
                    row["workplace_assignment_source"] = "NO_OLD_WORKER_RECORD_AVAILABLE"
                    row["domestic_worker_assignment_method"] = "FAILED"
                    missing_domestic_assignment += 1

                domestic_assigned_total += 1

            else:
                set_na_old_worker_columns(row)

            if row.get("employment_layer_status", "") == DOMESTIC_FINAL_STATUS:
                if row.get("workplace_id", "") == "":
                    empty_workplace_id_domestic += 1

            status_counts[row.get("employment_layer_status", "")] += 1
            assignment_type_counts[row.get("workplace_assignment_type", "")] += 1
            domestic_method_counts[row.get("domestic_worker_assignment_method", "")] += 1
            workplace_country_counts[row.get("workplace_country", "")] += 1

            writer.writerow(row)

            total_written += 1

            if total_written % 1_000_000 == 0:
                print(f"Kiírt agentek domestic workplace-kel: {total_written:,}")

    unused_old_worker_records = (
        old_worker_pools["old_total_rows"] - used_old_worker_records
    )

    return {
        "total_written": total_written,
        "status_counts": dict(status_counts),
        "assignment_type_counts": dict(assignment_type_counts),
        "domestic_method_counts": dict(domestic_method_counts),
        "workplace_country_counts": dict(workplace_country_counts),
        "domestic_assigned_total": domestic_assigned_total,
        "domestic_exact_assigned": domestic_exact_assigned,
        "domestic_fallback_assigned": domestic_fallback_assigned,
        "missing_domestic_assignment": missing_domestic_assignment,
        "empty_workplace_id_domestic": empty_workplace_id_domestic,
        "used_old_worker_records": used_old_worker_records,
        "unused_old_worker_records": unused_old_worker_records,
    }


# ============================================================
# 8. VALIDÁCIÓK
# ============================================================

def write_domestic_assignment_validation(
    current_counts: dict,
    old_worker_pools: dict,
    write_result: dict,
) -> None:
    rows = []

    for status, count in write_result["status_counts"].items():
        rows.append({
            "validation_group": "employment_layer_status",
            "category": status,
            "count": count,
        })

    for assignment_type, count in write_result["assignment_type_counts"].items():
        rows.append({
            "validation_group": "workplace_assignment_type",
            "category": assignment_type,
            "count": count,
        })

    for method, count in write_result["domestic_method_counts"].items():
        rows.append({
            "validation_group": "domestic_worker_assignment_method",
            "category": method,
            "count": count,
        })

    for country, count in write_result["workplace_country_counts"].items():
        rows.append({
            "validation_group": "workplace_country",
            "category": country,
            "count": count,
        })

    pd.DataFrame(rows).to_csv(
        DOMESTIC_ASSIGNMENT_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def write_unused_old_worker_records_summary(
    old_worker_pools: dict,
    write_result: dict,
) -> None:
    rows = [
        {
            "metric": "old_worker_records_total",
            "value": old_worker_pools["old_total_rows"],
            "note": "Régi agents_with_workplaces rekordok összesen.",
        },
        {
            "metric": "old_worker_records_used",
            "value": write_result["used_old_worker_records"],
            "note": "Domestic assignmenthez felhasznált régi worker rekordok.",
        },
        {
            "metric": "old_worker_records_unused",
            "value": write_result["unused_old_worker_records"],
            "note": "Fel nem használt régi worker rekordok. Explicit foreign réteg miatt ez várható.",
        },
        {
            "metric": "old_worker_missing_workplace_id",
            "value": old_worker_pools["old_missing_workplace_id"],
            "note": "Régi worker rekordok hiányzó workplace_id-val.",
        },
        {
            "metric": "fallback_pool_remaining_after_assignment",
            "value": len(old_worker_pools["fallback_pool"]),
            "note": "Assignment után megmaradt fallback pool rekordok száma.",
        },
    ]

    pd.DataFrame(rows).to_csv(
        DOMESTIC_ASSIGNMENT_UNUSED_OLD_RECORDS_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def write_summary(
    current_counts: dict,
    old_worker_pools: dict,
    write_result: dict,
) -> None:
    status_counts = write_result["status_counts"]
    assignment_type_counts = write_result["assignment_type_counts"]
    method_counts = write_result["domestic_method_counts"]
    country_counts = write_result["workplace_country_counts"]

    rows = [
        {
            "metric": "output_agents_total",
            "value": write_result["total_written"],
            "note": "Output agentek száma.",
        },
        {
            "metric": "input_domestic_pending_total",
            "value": current_counts["domestic_pending_total"],
            "note": "Input 95 fájlban EMP_DOM_PENDING agentek száma.",
        },
        {
            "metric": "output_employment_status_emp_dom",
            "value": status_counts.get(DOMESTIC_FINAL_STATUS, 0),
            "note": "Domestic workplace-kel ellátott foglalkoztatott agentek.",
        },
        {
            "metric": "output_employment_status_emp_foreign",
            "value": status_counts.get(FOREIGN_STATUS, 0),
            "note": "Külföldi workplace státuszú foglalkoztatott agentek.",
        },
        {
            "metric": "output_employment_status_not_emp",
            "value": status_counts.get(NOT_EMP_STATUS, 0),
            "note": "Nem foglalkoztatott agentek.",
        },
        {
            "metric": "output_employment_status_emp_dom_pending",
            "value": status_counts.get(DOMESTIC_PENDING_STATUS, 0),
            "note": "Ideálisan 0, mert minden domestic pending agentnek assignmentet kell kapnia.",
        },
        {
            "metric": "output_employment_status_missing_wp",
            "value": status_counts.get("MISSING_WP", 0),
            "note": "Ideálisan 0.",
        },
        {
            "metric": "domestic_exact_assigned",
            "value": write_result["domestic_exact_assigned"],
            "note": "Pontos county × sex × age_group × education kulcson assigned.",
        },
        {
            "metric": "domestic_fallback_assigned",
            "value": write_result["domestic_fallback_assigned"],
            "note": "Fallback régi worker rekorddal assigned.",
        },
        {
            "metric": "domestic_assignment_exact_share",
            "value": safe_divide(
                write_result["domestic_exact_assigned"],
                write_result["domestic_assigned_total"],
            ),
            "note": "Domestic assignment exact aránya.",
        },
        {
            "metric": "domestic_assignment_fallback_share",
            "value": safe_divide(
                write_result["domestic_fallback_assigned"],
                write_result["domestic_assigned_total"],
            ),
            "note": "Domestic assignment fallback aránya.",
        },
        {
            "metric": "workplace_assignment_type_domestic",
            "value": assignment_type_counts.get("DOMESTIC", 0),
            "note": "Belföldi workplace assignment típusú agentek.",
        },
        {
            "metric": "workplace_assignment_type_foreign",
            "value": assignment_type_counts.get("FOREIGN", 0),
            "note": "Foreign workplace típusú agentek.",
        },
        {
            "metric": "workplace_assignment_type_na",
            "value": assignment_type_counts.get("NA", 0),
            "note": "Nem foglalkoztatott agentek workplace assignment típusa.",
        },
        {
            "metric": "workplace_country_hu",
            "value": country_counts.get("HU", 0),
            "note": "Magyarországi workplace-es agentek.",
        },
        {
            "metric": "workplace_country_foreign",
            "value": country_counts.get("FOREIGN", 0),
            "note": "Külföldi workplace-es agentek.",
        },
        {
            "metric": "workplace_country_na",
            "value": country_counts.get("NA", 0),
            "note": "Nem foglalkoztatott agentek.",
        },
        {
            "metric": "old_worker_records_total",
            "value": old_worker_pools["old_total_rows"],
            "note": "Régi worker/workplace pool összes rekordja.",
        },
        {
            "metric": "old_worker_records_used",
            "value": write_result["used_old_worker_records"],
            "note": "Felhasznált régi worker rekordok száma.",
        },
        {
            "metric": "old_worker_records_unused",
            "value": write_result["unused_old_worker_records"],
            "note": "Fel nem használt régi worker rekordok száma.",
        },
        {
            "metric": "missing_domestic_assignment",
            "value": write_result["missing_domestic_assignment"],
            "note": "Ideálisan 0.",
        },
        {
            "metric": "empty_workplace_id_domestic",
            "value": write_result["empty_workplace_id_domestic"],
            "note": "Ideálisan 0.",
        },
    ]

    pd.DataFrame(rows).to_csv(
        DOMESTIC_ASSIGNMENT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script a 95-ös kompakt foreign/domestic státuszú agentfájlt használja inputként.",
        },
        {
            "order": 2,
            "note": "Csak az EMP_DOM_PENDING agentek kapnak régi belföldi workplace rekordot.",
        },
        {
            "order": 3,
            "note": "Az EMP_FOREIGN agentek foreign workplace státuszát nem módosítjuk.",
        },
        {
            "order": 4,
            "note": "A NOT_EMP agentek továbbra is NO_WORK / NO_TEAOR jelölésűek maradnak.",
        },
        {
            "order": 5,
            "note": "Elsődleges domestic matching kulcs: county × sex × 5 éves age_group × education.",
        },
        {
            "order": 6,
            "note": "Ha pontos kulcson nincs elég régi worker rekord, fallbackként a régi worker pool többletrekordjaiból rendelünk.",
        },
        {
            "order": 7,
            "note": "A régi worker rekordok száma nagyobb, mint a domestic target, mert explicit foreign workplace réteget vezettünk be.",
        },
        {
            "order": 8,
            "note": "A fel nem használt régi worker rekordok nem hibák, hanem a domestic pool többletének dokumentált maradéka.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        DOMESTIC_ASSIGNMENT_METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 9. FŐ FUTTATÁS
# ============================================================

def run_domestic_workplace_assignment() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Current domestic pending agentek számlálása...")
    current_counts = count_current_domestic_pending_agents(
        current_agents_csv=CURRENT_COMPACT_AGENTS_CSV
    )

    print("Régi worker/workplace pool betöltése...")
    old_worker_pools = build_old_worker_assignment_pools(
        previous_workers_csv=PREVIOUS_WORKER_WORKPLACE_CSV,
        domestic_targets_by_key=current_counts["domestic_counts_by_key"],
    )

    print("Output agentfájl írása domestic workplace assignmentekkel...")
    write_result = write_agents_with_domestic_workplaces(
        current_agents_csv=CURRENT_COMPACT_AGENTS_CSV,
        output_agents_csv=AGENTS_WITH_DOMESTIC_WORKPLACES_CSV,
        old_worker_pools=old_worker_pools,
    )

    print("Validáció írása...")
    write_domestic_assignment_validation(
        current_counts=current_counts,
        old_worker_pools=old_worker_pools,
        write_result=write_result,
    )

    print("Unused old worker summary írása...")
    write_unused_old_worker_records_summary(
        old_worker_pools=old_worker_pools,
        write_result=write_result,
    )

    print("Summary írása...")
    write_summary(
        current_counts=current_counts,
        old_worker_pools=old_worker_pools,
        write_result=write_result,
    )

    print("Method notes írása...")
    write_method_notes()

    print()
    print("Kész.")
    print(f"Output agentek: {write_result['total_written']:,}")
    print(f"Domestic assigned: {write_result['domestic_assigned_total']:,}")
    print(f"Domestic exact: {write_result['domestic_exact_assigned']:,}")
    print(f"Domestic fallback: {write_result['domestic_fallback_assigned']:,}")
    print(f"Old worker used: {write_result['used_old_worker_records']:,}")
    print(f"Old worker unused: {write_result['unused_old_worker_records']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_domestic_workplace_assignment()