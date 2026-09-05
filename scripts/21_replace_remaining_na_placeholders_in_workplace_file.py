from pathlib import Path
import csv


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

INPUT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/domestic_workplace_assignment/"
    "99_agents_with_domestic_workplaces.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/domestic_workplace_assignment"
)

OUTPUT_AGENTS_CSV = OUTPUT_FOLDER / "99_agents_with_domestic_workplaces_no_na_placeholders.csv"

NA_REPLACEMENT_VALIDATION_CSV = OUTPUT_FOLDER / "112_na_placeholder_replacement_validation.csv"
NA_REPLACEMENT_SUMMARY_CSV = OUTPUT_FOLDER / "113_na_placeholder_replacement_summary.csv"

MAX_ROWS_TO_PROCESS = None

# Csak ezekben az oszlopokban cserélünk NA/Na/na/nA -> NAPP.
# Nem globálisan cserélünk az egész fájlban.
PLACEHOLDER_COLUMNS_TO_CLEAN = [
    "workplace_assignment_type",
    "workplace_country",
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


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def is_na_placeholder(value: str) -> bool:
    """
    Csak akkor True, ha az érték pontosan NA valamilyen casinggel:
    NA, Na, na, nA.

    Nem cserél például ilyeneket:
    - NAPP
    - NO_WORK
    - FOREIGN_UNK_TEAOR
    - valamilyen hosszabb szöveg, amelyben szerepel az 'na'
    """
    return str(value).strip().lower() == "na"


def replace_na_placeholder(value: str) -> str:
    if is_na_placeholder(value):
        return "NAPP"

    return value


# ============================================================
# 3. CSERE ÉS VALIDÁCIÓ
# ============================================================

def replace_na_placeholders() -> dict:
    total_rows = 0
    total_replacements = 0

    replacements_by_column = {
        column_name: 0
        for column_name in PLACEHOLDER_COLUMNS_TO_CLEAN
    }

    remaining_na_by_column = {
        column_name: 0
        for column_name in PLACEHOLDER_COLUMNS_TO_CLEAN
    }

    value_counts_after_cleaning = {}

    with INPUT_AGENTS_CSV.open("r", encoding="utf-8-sig", newline="") as input_file, \
            OUTPUT_AGENTS_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames)

        missing_columns = [
            column_name
            for column_name in PLACEHOLDER_COLUMNS_TO_CLEAN
            if column_name not in fieldnames
        ]

        if missing_columns:
            raise ValueError(
                f"Ezek a tisztítandó oszlopok hiányoznak az input fájlból: {missing_columns}"
            )

        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            for column_name in PLACEHOLDER_COLUMNS_TO_CLEAN:
                original_value = row[column_name]

                if is_na_placeholder(original_value):
                    row[column_name] = "NAPP"
                    replacements_by_column[column_name] += 1
                    total_replacements += 1

            for column_name in PLACEHOLDER_COLUMNS_TO_CLEAN:
                cleaned_value = row[column_name]

                if is_na_placeholder(cleaned_value):
                    remaining_na_by_column[column_name] += 1

                key = (column_name, cleaned_value)
                value_counts_after_cleaning[key] = value_counts_after_cleaning.get(key, 0) + 1

            writer.writerow(row)

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"NA placeholder tisztítás: {total_rows:,} sor")

    return {
        "total_rows": total_rows,
        "total_replacements": total_replacements,
        "replacements_by_column": replacements_by_column,
        "remaining_na_by_column": remaining_na_by_column,
        "value_counts_after_cleaning": value_counts_after_cleaning,
    }


def write_validation(result: dict) -> None:
    rows = []

    for column_name, replacement_count in result["replacements_by_column"].items():
        rows.append({
            "validation_group": "replacements_by_column",
            "column_name": column_name,
            "category": "NA_TO_NAPP",
            "count": replacement_count,
        })

    for column_name, remaining_count in result["remaining_na_by_column"].items():
        rows.append({
            "validation_group": "remaining_na_by_column",
            "column_name": column_name,
            "category": "REMAINING_NA_CASE_INSENSITIVE",
            "count": remaining_count,
        })

    for (column_name, value), count in result["value_counts_after_cleaning"].items():
        rows.append({
            "validation_group": "value_counts_after_cleaning",
            "column_name": column_name,
            "category": value,
            "count": count,
        })

    with NA_REPLACEMENT_VALIDATION_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:
        fieldnames = [
            "validation_group",
            "column_name",
            "category",
            "count",
        ]

        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(result: dict) -> None:
    remaining_na_total = sum(result["remaining_na_by_column"].values())

    rows = [
        {
            "metric": "output_agents_total",
            "value": result["total_rows"],
            "note": "Output agentfájl sorainak száma.",
        },
        {
            "metric": "total_na_to_napp_replacements",
            "value": result["total_replacements"],
            "note": "Összes NA/Na/na/nA -> NAPP csere a kijelölt placeholder oszlopokban.",
        },
        {
            "metric": "remaining_na_placeholders_total",
            "value": remaining_na_total,
            "note": "Maradék NA placeholder a kijelölt oszlopokban. Ideálisan 0.",
        },
    ]

    with NA_REPLACEMENT_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:
        fieldnames = [
            "metric",
            "value",
            "note",
        ]

        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 4. FŐ FUTTATÁS
# ============================================================

def run_na_placeholder_replacement() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("NA placeholder értékek cseréje NAPP-ra...")
    result = replace_na_placeholders()

    print("Validáció írása...")
    write_validation(result)

    print("Summary írása...")
    write_summary(result)

    print()
    print("Kész.")
    print(f"Output agentek: {result['total_rows']:,}")
    print(f"Összes csere: {result['total_replacements']:,}")
    print(f"Output fájl: {OUTPUT_AGENTS_CSV.resolve()}")


if __name__ == "__main__":
    run_na_placeholder_replacement()