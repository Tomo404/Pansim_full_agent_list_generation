from pathlib import Path
import csv
import re
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

AGENTS_WITH_DOMESTIC_WORKPLACES_CSV = Path(
    "../outputs/agent_attribute_generation/domestic_workplace_assignment/"
    "99_agents_with_domestic_workplaces.csv"
)

PREVIOUS_WORKER_WORKPLACE_CSV = Path(
    "../data/raw_reference/agents_with_workplaces.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/employment_workplace_validation"
)

LAYER_SUMMARY_CSV = OUTPUT_FOLDER / "104_employment_workplace_layer_summary.csv"
STATUS_VALIDATION_CSV = OUTPUT_FOLDER / "105_employment_status_validation.csv"
OLD_VS_ASSIGNED_TEAOR_CSV = OUTPUT_FOLDER / "106_old_vs_assigned_workplace_teaor_validation.csv"
OLD_VS_ASSIGNED_SECTOR_CSV = OUTPUT_FOLDER / "107_old_vs_assigned_sector_code_validation.csv"
OLD_VS_ASSIGNED_WORKPLACE_COUNTY_CSV = OUTPUT_FOLDER / "108_old_vs_assigned_workplace_county_validation.csv"
OLD_VS_ASSIGNED_FALLBACK_CSV = OUTPUT_FOLDER / "109_old_vs_assigned_fallback_validation.csv"
UNUSED_OLD_RECORDS_PROFILE_CSV = OUTPUT_FOLDER / "110_unused_old_worker_records_profile.csv"
METHOD_NOTES_CSV = OUTPUT_FOLDER / "111_employment_workplace_validation_method_notes.csv"

MAX_ROWS_TO_PROCESS = None

DOMESTIC_STATUS = "EMP_DOM"
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
    Akkor hasznos, ha a régi pipeline fájljaiban kicsit eltérő oszlopnevek vannak.
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


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


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


def add_counter(counter: dict, key: str, amount: int = 1) -> None:
    counter[key] += amount


# ============================================================
# 3. 99-ES CURRENT AGENTFÁJL PROFILOZÁSA
# ============================================================

def profile_current_employment_layer(agents_csv: Path) -> dict:
    total_rows = 0

    employment_status_counts = defaultdict(int)
    workplace_assignment_type_counts = defaultdict(int)
    workplace_country_counts = defaultdict(int)
    domestic_assignment_method_counts = defaultdict(int)

    domestic_teaor_counts = defaultdict(int)
    domestic_sector_counts = defaultdict(int)
    domestic_workplace_county_counts = defaultdict(int)
    domestic_old_fallback_counts = defaultdict(int)
    domestic_old_teaor_fallback_counts = defaultdict(int)
    domestic_old_fallback_level_counts = defaultdict(int)

    foreign_method_counts = defaultdict(int)

    used_previous_worker_ids = set()
    duplicate_previous_worker_id_count = 0
    duplicate_previous_worker_ids_sample = []

    domestic_count = 0
    foreign_count = 0
    not_employed_count = 0

    missing_workplace_assignment_count = 0
    domestic_empty_workplace_id_count = 0
    foreign_non_foreign_workplace_id_count = 0
    not_employed_non_no_workplace_count = 0

    required_columns = [
        "employment_layer_status",
        "workplace_assignment_type",
        "workplace_country",
        "workplace_id",
        "workplace_teaor_code",
        "previous_worker_record_id",
        "domestic_worker_assignment_method",
        "old_worker_sector_code",
        "workplace_county",
        "old_worker_used_fallback",
        "old_worker_fallback_level",
        "old_worker_used_teaor_fallback",
        "foreign_workplace_assignment_method",
    ]

    with agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for column_name in required_columns:
            if column_name not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop a 99-es agentfájlból: {column_name}"
                )

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            status = normalize_text(row["employment_layer_status"])
            assignment_type = normalize_text(row["workplace_assignment_type"])
            workplace_country = normalize_text(row["workplace_country"])
            workplace_id = normalize_text(row["workplace_id"])
            workplace_teaor = normalize_text(row["workplace_teaor_code"])
            previous_worker_id = normalize_text(row["previous_worker_record_id"])
            domestic_method = normalize_text(row["domestic_worker_assignment_method"])

            employment_status_counts[status] += 1
            workplace_assignment_type_counts[assignment_type] += 1
            workplace_country_counts[workplace_country] += 1
            domestic_assignment_method_counts[domestic_method] += 1

            if status == DOMESTIC_STATUS:
                domestic_count += 1

                if workplace_id == "":
                    domestic_empty_workplace_id_count += 1

                if previous_worker_id in used_previous_worker_ids:
                    duplicate_previous_worker_id_count += 1

                    if len(duplicate_previous_worker_ids_sample) < 20:
                        duplicate_previous_worker_ids_sample.append(previous_worker_id)
                else:
                    used_previous_worker_ids.add(previous_worker_id)

                domestic_teaor_counts[workplace_teaor] += 1
                domestic_sector_counts[normalize_text(row["old_worker_sector_code"])] += 1
                domestic_workplace_county_counts[normalize_county_name(row["workplace_county"])] += 1
                domestic_old_fallback_counts[normalize_text(row["old_worker_used_fallback"])] += 1
                domestic_old_teaor_fallback_counts[normalize_text(row["old_worker_used_teaor_fallback"])] += 1
                domestic_old_fallback_level_counts[normalize_text(row["old_worker_fallback_level"])] += 1

            elif status == FOREIGN_STATUS:
                foreign_count += 1
                foreign_method_counts[normalize_text(row["foreign_workplace_assignment_method"])] += 1

                if workplace_id != "FOREIGN":
                    foreign_non_foreign_workplace_id_count += 1

            elif status == NOT_EMP_STATUS:
                not_employed_count += 1

                if workplace_id != "NO_WORK":
                    not_employed_non_no_workplace_count += 1

            else:
                missing_workplace_assignment_count += 1

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"99-es agentfájl validálva: {total_rows:,}")

    return {
        "total_rows": total_rows,
        "employment_status_counts": dict(employment_status_counts),
        "workplace_assignment_type_counts": dict(workplace_assignment_type_counts),
        "workplace_country_counts": dict(workplace_country_counts),
        "domestic_assignment_method_counts": dict(domestic_assignment_method_counts),
        "foreign_method_counts": dict(foreign_method_counts),
        "domestic_teaor_counts": dict(domestic_teaor_counts),
        "domestic_sector_counts": dict(domestic_sector_counts),
        "domestic_workplace_county_counts": dict(domestic_workplace_county_counts),
        "domestic_old_fallback_counts": dict(domestic_old_fallback_counts),
        "domestic_old_teaor_fallback_counts": dict(domestic_old_teaor_fallback_counts),
        "domestic_old_fallback_level_counts": dict(domestic_old_fallback_level_counts),
        "used_previous_worker_ids": used_previous_worker_ids,
        "duplicate_previous_worker_id_count": duplicate_previous_worker_id_count,
        "duplicate_previous_worker_ids_sample": duplicate_previous_worker_ids_sample,
        "domestic_count": domestic_count,
        "foreign_count": foreign_count,
        "not_employed_count": not_employed_count,
        "missing_workplace_assignment_count": missing_workplace_assignment_count,
        "domestic_empty_workplace_id_count": domestic_empty_workplace_id_count,
        "foreign_non_foreign_workplace_id_count": foreign_non_foreign_workplace_id_count,
        "not_employed_non_no_workplace_count": not_employed_non_no_workplace_count,
    }


# ============================================================
# 4. RÉGI WORKER POOL PROFILOZÁSA
# ============================================================

def profile_previous_worker_pool(previous_workers_csv: Path, used_previous_worker_ids: set) -> dict:
    old_total_rows = 0

    old_teaor_counts = defaultdict(int)
    old_sector_counts = defaultdict(int)
    old_workplace_county_counts = defaultdict(int)
    old_fallback_counts = defaultdict(int)
    old_teaor_fallback_counts = defaultdict(int)
    old_fallback_level_counts = defaultdict(int)

    unused_teaor_counts = defaultdict(int)
    unused_sector_counts = defaultdict(int)
    unused_workplace_county_counts = defaultdict(int)
    unused_fallback_counts = defaultdict(int)
    unused_teaor_fallback_counts = defaultdict(int)
    unused_fallback_level_counts = defaultdict(int)

    used_id_found_count = 0
    old_duplicate_agent_id_count = 0
    old_agent_ids_seen = set()

    old_missing_workplace_id_count = 0

    with previous_workers_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        fieldnames = list(reader.fieldnames)

        agent_id_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["agent_id", "worker_id", "old_agent_id"],
            logical_column_name="agent_id",
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

        used_fallback_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["used_fallback"],
            logical_column_name="used_fallback",
        )

        fallback_level_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["fallback_level"],
            logical_column_name="fallback_level",
        )

        used_teaor_fallback_column = require_first_existing_column(
            fieldnames=fieldnames,
            possible_column_names=["used_teaor_fallback"],
            logical_column_name="used_teaor_fallback",
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

        print("Régi worker fájlban felismert oszlopok:")
        print(f"  agent_id: {agent_id_column}")
        print(f"  sector_code: {sector_code_column}")
        print(f"  workplace_teaor_code: {workplace_teaor_code_column}")

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and old_total_rows >= MAX_ROWS_TO_PROCESS:
                break

            agent_id = normalize_text(row[agent_id_column])
            workplace_id = normalize_text(row[workplace_id_column])
            teaor = normalize_text(row[workplace_teaor_code_column])
            sector_code = normalize_text(row[sector_code_column])
            workplace_county = normalize_county_name(row[workplace_county_column])
            used_fallback = normalize_text(row[used_fallback_column])
            used_teaor_fallback = normalize_text(row[used_teaor_fallback_column])
            fallback_level = normalize_text(row[fallback_level_column])

            if workplace_id == "":
                old_missing_workplace_id_count += 1

            if agent_id in old_agent_ids_seen:
                old_duplicate_agent_id_count += 1
            else:
                old_agent_ids_seen.add(agent_id)

            old_teaor_counts[teaor] += 1
            old_sector_counts[sector_code] += 1
            old_workplace_county_counts[workplace_county] += 1
            old_fallback_counts[used_fallback] += 1
            old_teaor_fallback_counts[used_teaor_fallback] += 1
            old_fallback_level_counts[fallback_level] += 1

            if agent_id in used_previous_worker_ids:
                used_id_found_count += 1
            else:
                unused_teaor_counts[teaor] += 1
                unused_sector_counts[sector_code] += 1
                unused_workplace_county_counts[workplace_county] += 1
                unused_fallback_counts[used_fallback] += 1
                unused_teaor_fallback_counts[used_teaor_fallback] += 1
                unused_fallback_level_counts[fallback_level] += 1

            old_total_rows += 1

            if old_total_rows % 1_000_000 == 0:
                print(f"Régi worker pool validálva: {old_total_rows:,}")

    return {
        "old_total_rows": old_total_rows,
        "old_missing_workplace_id_count": old_missing_workplace_id_count,
        "old_duplicate_agent_id_count": old_duplicate_agent_id_count,
        "used_id_found_count": used_id_found_count,
        "old_teaor_counts": dict(old_teaor_counts),
        "old_sector_counts": dict(old_sector_counts),
        "old_workplace_county_counts": dict(old_workplace_county_counts),
        "old_fallback_counts": dict(old_fallback_counts),
        "old_teaor_fallback_counts": dict(old_teaor_fallback_counts),
        "old_fallback_level_counts": dict(old_fallback_level_counts),
        "unused_teaor_counts": dict(unused_teaor_counts),
        "unused_sector_counts": dict(unused_sector_counts),
        "unused_workplace_county_counts": dict(unused_workplace_county_counts),
        "unused_fallback_counts": dict(unused_fallback_counts),
        "unused_teaor_fallback_counts": dict(unused_teaor_fallback_counts),
        "unused_fallback_level_counts": dict(unused_fallback_level_counts),
    }


# ============================================================
# 5. ÖSSZEHASONLÍTÓ TÁBLÁK
# ============================================================

def compare_old_vs_assigned(
    old_counts: dict,
    assigned_counts: dict,
    category_column_name: str,
    old_count_name: str = "old_worker_pool_count",
    assigned_count_name: str = "assigned_domestic_count",
) -> pd.DataFrame:
    categories = set(old_counts.keys()) | set(assigned_counts.keys())

    rows = []

    old_total = sum(old_counts.values())
    assigned_total = sum(assigned_counts.values())

    for category in categories:
        old_count = old_counts.get(category, 0)
        assigned_count = assigned_counts.get(category, 0)

        rows.append({
            category_column_name: category,
            old_count_name: old_count,
            assigned_count_name: assigned_count,
            "difference_assigned_minus_old": assigned_count - old_count,
            "absolute_difference": abs(assigned_count - old_count),
            "old_share": safe_divide(old_count, old_total),
            "assigned_share": safe_divide(assigned_count, assigned_total),
            "share_difference_assigned_minus_old": (
                safe_divide(assigned_count, assigned_total)
                - safe_divide(old_count, old_total)
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["absolute_difference", old_count_name], ascending=[False, False])
        .reset_index(drop=True)
    )


def create_unused_profile(
    previous_worker_profile: dict,
) -> pd.DataFrame:
    rows = []

    profile_specs = [
        ("unused_workplace_teaor_code", previous_worker_profile["unused_teaor_counts"]),
        ("unused_sector_code", previous_worker_profile["unused_sector_counts"]),
        ("unused_workplace_county", previous_worker_profile["unused_workplace_county_counts"]),
        ("unused_used_fallback", previous_worker_profile["unused_fallback_counts"]),
        ("unused_used_teaor_fallback", previous_worker_profile["unused_teaor_fallback_counts"]),
        ("unused_fallback_level", previous_worker_profile["unused_fallback_level_counts"]),
    ]

    for group_name, counter in profile_specs:
        total = sum(counter.values())

        for category, count in counter.items():
            rows.append({
                "profile_group": group_name,
                "category": category,
                "unused_count": count,
                "unused_share_within_group": safe_divide(count, total),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["profile_group", "unused_count"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ============================================================
# 6. SUMMARY ÉS METHOD NOTES
# ============================================================

def write_status_validation(current_profile: dict) -> None:
    rows = []

    for status, count in current_profile["employment_status_counts"].items():
        rows.append({
            "validation_group": "employment_layer_status",
            "category": status,
            "count": count,
        })

    for assignment_type, count in current_profile["workplace_assignment_type_counts"].items():
        rows.append({
            "validation_group": "workplace_assignment_type",
            "category": assignment_type,
            "count": count,
        })

    for country, count in current_profile["workplace_country_counts"].items():
        rows.append({
            "validation_group": "workplace_country",
            "category": country,
            "count": count,
        })

    for method, count in current_profile["domestic_assignment_method_counts"].items():
        rows.append({
            "validation_group": "domestic_worker_assignment_method",
            "category": method,
            "count": count,
        })

    for method, count in current_profile["foreign_method_counts"].items():
        rows.append({
            "validation_group": "foreign_workplace_assignment_method",
            "category": method,
            "count": count,
        })

    pd.DataFrame(rows).to_csv(
        STATUS_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def write_summary(current_profile: dict, previous_worker_profile: dict) -> None:
    domestic_total = current_profile["domestic_count"]
    foreign_total = current_profile["foreign_count"]
    not_employed_total = current_profile["not_employed_count"]

    employed_total = domestic_total + foreign_total

    rows = [
        {
            "metric": "output_agents_total",
            "value": current_profile["total_rows"],
            "note": "99-es teljes agentfájl sorainak száma.",
        },
        {
            "metric": "employed_total",
            "value": employed_total,
            "note": "EMP_DOM + EMP_FOREIGN.",
        },
        {
            "metric": "domestic_employed_total",
            "value": domestic_total,
            "note": "Belföldi workplace-kel rendelkező foglalkoztatott agentek.",
        },
        {
            "metric": "foreign_employed_total",
            "value": foreign_total,
            "note": "Külföldi workplace-kel rendelkező foglalkoztatott agentek.",
        },
        {
            "metric": "not_employed_total",
            "value": not_employed_total,
            "note": "Nem foglalkoztatott agentek.",
        },
        {
            "metric": "missing_workplace_assignment_status_count",
            "value": current_profile["missing_workplace_assignment_count"],
            "note": "Ismeretlen / nem várt employment_layer_status sorok. Ideálisan 0.",
        },
        {
            "metric": "domestic_empty_workplace_id_count",
            "value": current_profile["domestic_empty_workplace_id_count"],
            "note": "EMP_DOM agentek üres workplace_id-val. Ideálisan 0.",
        },
        {
            "metric": "foreign_non_foreign_workplace_id_count",
            "value": current_profile["foreign_non_foreign_workplace_id_count"],
            "note": "EMP_FOREIGN agentek nem FOREIGN workplace_id-val. Ideálisan 0.",
        },
        {
            "metric": "not_employed_non_no_workplace_count",
            "value": current_profile["not_employed_non_no_workplace_count"],
            "note": "NOT_EMP agentek nem NO_WORK workplace_id-val. Ideálisan 0.",
        },
        {
            "metric": "domestic_previous_worker_ids_used_unique",
            "value": len(current_profile["used_previous_worker_ids"]),
            "note": "Egyedi régi worker rekord ID-k, amelyeket domestic assignmenthez használtunk.",
        },
        {
            "metric": "domestic_duplicate_previous_worker_id_count",
            "value": current_profile["duplicate_previous_worker_id_count"],
            "note": "Duplikált régi worker ID felhasználás domestic assignmentben. Ideálisan 0.",
        },
        {
            "metric": "old_worker_records_total",
            "value": previous_worker_profile["old_total_rows"],
            "note": "Régi agents_with_workplaces.csv sorainak száma.",
        },
        {
            "metric": "old_worker_ids_used_found_in_old_file",
            "value": previous_worker_profile["used_id_found_count"],
            "note": "A domestic assignmentben használt régi worker ID-k megtalálva a régi fájlban.",
        },
        {
            "metric": "old_worker_records_unused",
            "value": previous_worker_profile["old_total_rows"] - previous_worker_profile["used_id_found_count"],
            "note": "Fel nem használt régi worker rekordok.",
        },
        {
            "metric": "old_worker_missing_workplace_id_count",
            "value": previous_worker_profile["old_missing_workplace_id_count"],
            "note": "Régi worker rekordok üres workplace_id-val. Ideálisan 0.",
        },
        {
            "metric": "old_worker_duplicate_agent_id_count",
            "value": previous_worker_profile["old_duplicate_agent_id_count"],
            "note": "Régi worker fájlban duplikált agent_id-k. Ideálisan 0.",
        },
    ]

    pd.DataFrame(rows).to_csv(
        LAYER_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script a 99_agents_with_domestic_workplaces.csv teljes employment/workplace rétegét validálja.",
        },
        {
            "order": 2,
            "note": "A DOMESTIC agenteknél a régi agents_with_workplaces.csv-ből átvett workplace_id, workplace_county, workplace_teaor_code és sector_code eloszlásokat ellenőrzi.",
        },
        {
            "order": 3,
            "note": "A FOREIGN agenteket külön rétegként kezeli; ezek nem vesznek részt a magyarországi TEÁOR validációban.",
        },
        {
            "order": 4,
            "note": "A NOT_EMP agenteknek NO_WORK / NO_TEAOR jelöléssel kell szerepelniük.",
        },
        {
            "order": 5,
            "note": "Az old_vs_assigned táblákban az assigned domestic countot a teljes régi worker poolhoz hasonlítjuk. A total eltérés várható, mert explicit foreign réteget vezettünk be, ezért 72 560 régi worker rekord nem kerül közvetlen felhasználásra.",
        },
        {
            "order": 6,
            "note": "Az NA jelölések tudatos placeholderként szerepelnek. Pandas beolvasásnál keep_default_na=False használata javasolt, hogy ne alakuljanak NaN értékké.",
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

def run_employment_workplace_validation() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("99-es employment/workplace agentfájl profilozása...")
    current_profile = profile_current_employment_layer(
        agents_csv=AGENTS_WITH_DOMESTIC_WORKPLACES_CSV
    )

    print("Régi worker pool profilozása...")
    previous_worker_profile = profile_previous_worker_pool(
        previous_workers_csv=PREVIOUS_WORKER_WORKPLACE_CSV,
        used_previous_worker_ids=current_profile["used_previous_worker_ids"],
    )

    print("Status validáció írása...")
    write_status_validation(
        current_profile=current_profile
    )

    print("TEÁOR old vs assigned validáció...")
    compare_old_vs_assigned(
        old_counts=previous_worker_profile["old_teaor_counts"],
        assigned_counts=current_profile["domestic_teaor_counts"],
        category_column_name="workplace_teaor_code",
    ).to_csv(
        OLD_VS_ASSIGNED_TEAOR_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Sector code old vs assigned validáció...")
    compare_old_vs_assigned(
        old_counts=previous_worker_profile["old_sector_counts"],
        assigned_counts=current_profile["domestic_sector_counts"],
        category_column_name="sector_code",
    ).to_csv(
        OLD_VS_ASSIGNED_SECTOR_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Workplace county old vs assigned validáció...")
    compare_old_vs_assigned(
        old_counts=previous_worker_profile["old_workplace_county_counts"],
        assigned_counts=current_profile["domestic_workplace_county_counts"],
        category_column_name="workplace_county",
    ).to_csv(
        OLD_VS_ASSIGNED_WORKPLACE_COUNTY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Fallback old vs assigned validáció...")
    fallback_rows = []

    fallback_comparisons = [
        (
            "used_fallback",
            previous_worker_profile["old_fallback_counts"],
            current_profile["domestic_old_fallback_counts"],
        ),
        (
            "used_teaor_fallback",
            previous_worker_profile["old_teaor_fallback_counts"],
            current_profile["domestic_old_teaor_fallback_counts"],
        ),
        (
            "fallback_level",
            previous_worker_profile["old_fallback_level_counts"],
            current_profile["domestic_old_fallback_level_counts"],
        ),
    ]

    for group_name, old_counts, assigned_counts in fallback_comparisons:
        comparison = compare_old_vs_assigned(
            old_counts=old_counts,
            assigned_counts=assigned_counts,
            category_column_name="category",
        )

        comparison.insert(0, "fallback_validation_group", group_name)

        fallback_rows.append(comparison)

    pd.concat(fallback_rows, ignore_index=True).to_csv(
        OLD_VS_ASSIGNED_FALLBACK_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Unused old worker rekordok profilja...")
    create_unused_profile(
        previous_worker_profile=previous_worker_profile
    ).to_csv(
        UNUSED_OLD_RECORDS_PROFILE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Summary írása...")
    write_summary(
        current_profile=current_profile,
        previous_worker_profile=previous_worker_profile,
    )

    print("Method notes írása...")
    write_method_notes()

    print()
    print("Kész.")
    print(f"Output agentek: {current_profile['total_rows']:,}")
    print(f"Domestic: {current_profile['domestic_count']:,}")
    print(f"Foreign: {current_profile['foreign_count']:,}")
    print(f"Not employed: {current_profile['not_employed_count']:,}")
    print(f"Old worker records: {previous_worker_profile['old_total_rows']:,}")
    print(f"Old worker used found: {previous_worker_profile['used_id_found_count']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_employment_workplace_validation()