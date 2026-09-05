from pathlib import Path
import csv
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

INPUT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/foreign_workplace_assignment/"
    "90_agents_with_foreign_domestic_workplace_status.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/foreign_workplace_assignment"
)

COMPACT_AGENTS_CSV = OUTPUT_FOLDER / "95_agents_with_compact_foreign_domestic_workplace_status.csv"
CODEBOOK_CSV = OUTPUT_FOLDER / "96_foreign_domestic_status_codebook.csv"
COMPACT_VALIDATION_CSV = OUTPUT_FOLDER / "97_compact_foreign_domestic_status_validation.csv"
COMPACT_SUMMARY_CSV = OUTPUT_FOLDER / "98_compact_foreign_domestic_status_summary.csv"

MAX_ROWS_TO_PROCESS = None


# ============================================================
# 2. KÓDTÁRAK
# ============================================================

VALUE_REPLACEMENTS = {
    "employment_layer_status": {
        "employed_with_foreign_workplace": "EMP_FOREIGN",
        "employed_pending_domestic_workplace": "EMP_DOM_PENDING",
        "not_employed": "NOT_EMP",
        "missing_workplace_assignment": "MISSING_WP",
    },

    "workplace_assignment_type": {
        "foreign_workplace": "FOREIGN",
        "domestic_workplace_pending": "DOM_PENDING",
        "not_applicable": "NA",
    },

    "workplace_assignment_source": {
        "hier_kulfold_foglalkoztatott": "HIER_FOREIGN",
        "pending_previous_worker_record_matching": "PENDING_OLD_WORKER",
        "not_employed_agent": "NOT_EMP",
    },

    "foreign_workplace_assignment_method": {
        "exact_key": "EXACT",
        "fallback": "FALLBACK",
        "not_foreign": "NOT_FOREIGN",
        "not_applicable": "NA",
    },

    "workplace_id": {
        "FOREIGN_WORKPLACE": "FOREIGN",
        "PENDING_DOMESTIC_WORKPLACE": "PENDING_DOM",
        "NO_WORKPLACE": "NO_WORK",
    },

    "workplace_country": {
        "FOREIGN": "FOREIGN",
        "HUNGARY": "HU",
        "NOT_APPLICABLE": "NA",
    },

    "workplace_county": {
        "FOREIGN": "FOREIGN",
        "PENDING_DOMESTIC_WORKPLACE": "PENDING_DOM",
        "NO_WORKPLACE": "NO_WORK",
    },

    "workplace_settlement": {
        "FOREIGN": "FOREIGN",
        "PENDING_DOMESTIC_WORKPLACE": "PENDING_DOM",
        "NO_WORKPLACE": "NO_WORK",
    },

    "workplace_settlement_type": {
        "FOREIGN": "FOREIGN",
        "PENDING_DOMESTIC_WORKPLACE": "PENDING_DOM",
        "NO_WORKPLACE": "NO_WORK",
    },

    "workplace_teaor_code": {
        "FOREIGN_OR_UNKNOWN_TEAOR": "FOREIGN_UNK_TEAOR",
        "PENDING_DOMESTIC_TEAOR": "PENDING_DOM_TEAOR",
        "NO_TEAOR": "NO_TEAOR",
    },

    "previous_worker_record_id": {
        "NO_PREVIOUS_WORKER_RECORD": "NO_PREV_WORKER",
        "PENDING_PREVIOUS_WORKER_RECORD": "PENDING_PREV_WORKER",
    },
}


CODE_MEANINGS = {
    "EMP_FOREIGN": "Foglalkoztatott agent, akinek külföldi munkavégzési helyet jelöltünk ki.",
    "EMP_DOM_PENDING": "Foglalkoztatott agent, aki belföldi workplace assignmentre vár.",
    "NOT_EMP": "Nem foglalkoztatott agent.",
    "MISSING_WP": "Hibajelző: foglalkoztatott agent, akinek nem sikerült workplace státuszt adni.",

    "FOREIGN": "Külföldi workplace / külföldi munkavégzési hely.",
    "DOM_PENDING": "Belföldi workplace assignment még nincs hozzárendelve.",
    "NA": "Nem alkalmazható.",
    "HIER_FOREIGN": "A foreign státusz forrása a hier_kulfold_foglalkoztatott tábla.",
    "PENDING_OLD_WORKER": "A belföldi workplace assignment később a régi worker/workplace poolból jön.",
    "EXACT": "Pontos county × settlement_type × sex × age_group × education kulcson kijelölve.",
    "FALLBACK": "Fallback hasonlósági szabállyal kijelölve.",
    "NOT_FOREIGN": "Foglalkoztatott, de nem foreign workplace státuszú.",
    "NO_WORK": "Nincs munkahely, mert az agent nem foglalkoztatott.",
    "HU": "Magyarországi workplace országkód.",
    "PENDING_DOM": "Belföldi workplace adat később töltendő.",
    "FOREIGN_UNK_TEAOR": "Külföldi workplace; magyar TEÁOR-kód nem ismert / nem alkalmazható.",
    "PENDING_DOM_TEAOR": "Belföldi TEÁOR-kód később töltendő.",
    "NO_TEAOR": "Nincs TEÁOR, mert az agent nem foglalkoztatott.",
    "NO_PREV_WORKER": "Nincs régi worker rekord, mert nem foglalkoztatott vagy foreign workplace.",
    "PENDING_PREV_WORKER": "Régi worker rekord később hozzárendelendő.",
}


# ============================================================
# 3. SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def compact_value(column_name: str, value: str) -> str:
    if column_name not in VALUE_REPLACEMENTS:
        return value

    return VALUE_REPLACEMENTS[column_name].get(value, value)


def write_codebook() -> None:
    rows = []

    for column_name, replacements in VALUE_REPLACEMENTS.items():
        for original_value, compact_code in replacements.items():
            rows.append({
                "column_name": column_name,
                "original_value": original_value,
                "compact_code": compact_code,
                "meaning": CODE_MEANINGS.get(compact_code, ""),
            })

    pd.DataFrame(rows).to_csv(
        CODEBOOK_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 4. KOMPAKTÁLÁS
# ============================================================

def compact_agent_file() -> dict:
    total_rows = 0

    counts_by_employment_layer_status = defaultdict(int)
    counts_by_workplace_assignment_type = defaultdict(int)
    counts_by_foreign_method = defaultdict(int)

    empty_counts_in_workplace_columns = defaultdict(int)

    columns_to_check_for_empty = [
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
        "previous_worker_record_id",
    ]

    with INPUT_AGENTS_CSV.open("r", encoding="utf-8-sig", newline="") as input_file, \
            COMPACT_AGENTS_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames)

        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            for column_name in VALUE_REPLACEMENTS:
                if column_name in row:
                    row[column_name] = compact_value(column_name, row[column_name])

            employment_status = row.get("employment_layer_status", "")
            assignment_type = row.get("workplace_assignment_type", "")
            foreign_method = row.get("foreign_workplace_assignment_method", "")

            counts_by_employment_layer_status[employment_status] += 1
            counts_by_workplace_assignment_type[assignment_type] += 1
            counts_by_foreign_method[foreign_method] += 1

            for column_name in columns_to_check_for_empty:
                if row.get(column_name, "") == "":
                    empty_counts_in_workplace_columns[column_name] += 1

            writer.writerow(row)

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"Kompaktált agentek: {total_rows:,}")

    return {
        "total_rows": total_rows,
        "counts_by_employment_layer_status": dict(counts_by_employment_layer_status),
        "counts_by_workplace_assignment_type": dict(counts_by_workplace_assignment_type),
        "counts_by_foreign_method": dict(counts_by_foreign_method),
        "empty_counts_in_workplace_columns": dict(empty_counts_in_workplace_columns),
    }


# ============================================================
# 5. VALIDÁCIÓ ÉS SUMMARY
# ============================================================

def write_validation(result: dict) -> None:
    rows = []

    for status, count in result["counts_by_employment_layer_status"].items():
        rows.append({
            "validation_group": "employment_layer_status",
            "category": status,
            "count": count,
        })

    for assignment_type, count in result["counts_by_workplace_assignment_type"].items():
        rows.append({
            "validation_group": "workplace_assignment_type",
            "category": assignment_type,
            "count": count,
        })

    for method, count in result["counts_by_foreign_method"].items():
        rows.append({
            "validation_group": "foreign_workplace_assignment_method",
            "category": method,
            "count": count,
        })

    for column_name, empty_count in result["empty_counts_in_workplace_columns"].items():
        rows.append({
            "validation_group": "empty_value_check",
            "category": column_name,
            "count": empty_count,
        })

    pd.DataFrame(rows).to_csv(
        COMPACT_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def write_summary(result: dict) -> None:
    status_counts = result["counts_by_employment_layer_status"]
    type_counts = result["counts_by_workplace_assignment_type"]
    method_counts = result["counts_by_foreign_method"]
    empty_counts = result["empty_counts_in_workplace_columns"]

    rows = [
        {
            "metric": "output_agents_total",
            "value": result["total_rows"],
            "note": "Kompaktált output agentfájl sorainak száma.",
        },
        {
            "metric": "employment_status_emp_foreign",
            "value": status_counts.get("EMP_FOREIGN", 0),
            "note": "Külföldi workplace státuszú foglalkoztatott agentek.",
        },
        {
            "metric": "employment_status_emp_dom_pending",
            "value": status_counts.get("EMP_DOM_PENDING", 0),
            "note": "Belföldi workplace assignmentre váró foglalkoztatott agentek.",
        },
        {
            "metric": "employment_status_not_emp",
            "value": status_counts.get("NOT_EMP", 0),
            "note": "Nem foglalkoztatott agentek.",
        },
        {
            "metric": "employment_status_missing_wp",
            "value": status_counts.get("MISSING_WP", 0),
            "note": "Hibajelző státusz. Ideálisan 0.",
        },
        {
            "metric": "assignment_type_foreign",
            "value": type_counts.get("FOREIGN", 0),
            "note": "Foreign workplace assignment típusú agentek.",
        },
        {
            "metric": "assignment_type_dom_pending",
            "value": type_counts.get("DOM_PENDING", 0),
            "note": "Belföldi workplace assignmentre váró agentek.",
        },
        {
            "metric": "assignment_type_na",
            "value": type_counts.get("NA", 0),
            "note": "Nem alkalmazható workplace assignment típus.",
        },
        {
            "metric": "foreign_method_exact",
            "value": method_counts.get("EXACT", 0),
            "note": "Pontos kulcson foreignnek kijelölt agentek.",
        },
        {
            "metric": "foreign_method_fallback",
            "value": method_counts.get("FALLBACK", 0),
            "note": "Fallbackkel foreignnek kijelölt agentek.",
        },
        {
            "metric": "empty_values_in_workplace_columns_total",
            "value": sum(empty_counts.values()),
            "note": "Üres értékek száma a fontos workplace oszlopokban. Ideálisan 0.",
        },
    ]

    pd.DataFrame(rows).to_csv(
        COMPACT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 6. FŐ FUTTATÁS
# ============================================================

def run_compaction() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Codebook írása...")
    write_codebook()

    print("Agentfájl kompaktálása...")
    result = compact_agent_file()

    print("Validáció írása...")
    write_validation(result)

    print("Summary írása...")
    write_summary(result)

    print()
    print("Kész.")
    print(f"Kompaktált agentek: {result['total_rows']:,}")
    print(f"Output fájl: {COMPACT_AGENTS_CSV.resolve()}")
    print(f"Codebook: {CODEBOOK_CSV.resolve()}")


if __name__ == "__main__":
    run_compaction()