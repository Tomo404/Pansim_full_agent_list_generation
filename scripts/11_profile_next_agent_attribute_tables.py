from pathlib import Path
import re

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

RAW_HIER_TABLE_FOLDER = Path("../data/raw_hier_tables")

OUTPUT_FOLDER = Path("../outputs/agent_attribute_generation/table_profiling")

TARGET_FILES = [
    "hier_gazdasagi_aktivitas_varmegyenkent_telepulestipusonkent.xlsx",
    "hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg.xlsx",
    "hier_foglalkoztatott_nepesseg_foglalkozasi_focsoport.xlsx",
    "hier_foglalkoztatott_nemzetgazdasagi_agazat_1.xlsx",
    "hier_foglalkoztatott_ingazas_1.xlsx",
    "hier_Egeszseg_allapot_varmegyenkent_telepulestipusonkent.xlsx",
    "hier_iskolaba_jaro_nepesseg.xlsx",
]

FILE_INVENTORY_CSV = OUTPUT_FOLDER / "49_next_attribute_file_inventory.csv"
STRUCTURE_PROFILE_CSV = OUTPUT_FOLDER / "50_next_attribute_table_structure_profile.csv"
CATEGORY_VALUES_CSV = OUTPUT_FOLDER / "51_next_attribute_category_values.csv"
AGE_LABELS_CSV = OUTPUT_FOLDER / "52_next_attribute_detected_age_labels.csv"
NUMERIC_TOTALS_CSV = OUTPUT_FOLDER / "53_next_attribute_numeric_totals.csv"
LONG_PREVIEW_CSV = OUTPUT_FOLDER / "54_next_attribute_long_preview.csv"


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def normalize_text(raw_value) -> str:
    if pd.isna(raw_value):
        return ""

    text = str(raw_value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def convert_to_number(raw_value):
    if pd.isna(raw_value):
        return None

    text = str(raw_value).strip()

    if text == "":
        return None

    # Magyar ezrespont kezelése, pl. 1.896
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return None


def detect_first_numeric_value_column(raw_table: pd.DataFrame) -> int:
    """
    Megkeresi, hol kezdődnek a számos értékoszlopok.

    A KSH hier táblákban jellemzően bal oldalt kategóriaoszlopok vannak,
    jobbra pedig sok numerikus értékoszlop.
    """
    numeric_table = raw_table.apply(pd.to_numeric, errors="coerce")
    numeric_counts_by_column = numeric_table.notna().sum(axis=0)

    for column_index, numeric_count in numeric_counts_by_column.items():
        if numeric_count > 0:
            return int(column_index)

    raise ValueError("Nem találtam numerikus értékoszlopot.")


def detect_first_category_data_row(raw_table: pd.DataFrame, first_value_column_index: int) -> int:
    """
    Megkeresi az első olyan sort, ahol a bal oldali kategóriarész már nem fejléc,
    hanem tényleges kategóriaadat.
    """
    for row_index in range(len(raw_table)):
        left_cells = raw_table.iloc[row_index, :first_value_column_index].tolist()
        right_cells = raw_table.iloc[row_index, first_value_column_index:].tolist()

        has_left_category_text = any(
            normalize_text(cell) != ""
            for cell in left_cells
        )

        has_numeric_value = any(
            convert_to_number(cell) is not None
            for cell in right_cells
        )

        if has_left_category_text and has_numeric_value:
            return row_index

    raise ValueError("Nem találtam első kategória-adatsort.")


def detect_possible_header_rows(
    raw_table: pd.DataFrame,
    first_data_row_index: int,
) -> list[int]:
    """
    Az adatsor előtti sorokat fejlécként kezeli.
    """
    return list(range(0, first_data_row_index))


def looks_like_age_label(text: str) -> bool:
    normalized = text.lower().strip()

    patterns = [
        r"\d+\s*[–-]\s*\d+\s*éves",
        r"\d+\s*évesnél fiatalabb",
        r"\d+\s*éves és idősebb",
        r"\d+\s*éves",
    ]

    return any(re.search(pattern, normalized) for pattern in patterns)


def create_compact_column_label(raw_table: pd.DataFrame, header_rows: list[int], column_index: int) -> str:
    """
    Egy értékoszlophoz összeállít egy kompakt fejlécnevet a felső sorokból.
    """
    parts = []

    for row_index in header_rows:
        value = normalize_text(raw_table.iloc[row_index, column_index])

        if value != "":
            parts.append(value)

    if len(parts) == 0:
        return f"value_column_{column_index + 1}"

    return " | ".join(parts)


# ============================================================
# 3. EGY FÁJL PROFILOZÁSA
# ============================================================

def profile_single_excel_file(file_path: Path) -> dict:
    """
    Egy Excel fájl összes sheetjét profilozza.
    """
    excel_file = pd.ExcelFile(file_path, engine="openpyxl")

    file_inventory_rows = []
    structure_rows = []
    category_value_rows = []
    age_label_rows = []
    numeric_total_rows = []
    long_preview_rows = []

    for sheet_name in excel_file.sheet_names:
        raw_table = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            engine="openpyxl",
        )

        file_inventory_rows.append({
            "file_name": file_path.name,
            "sheet_name": sheet_name,
            "row_count": raw_table.shape[0],
            "column_count": raw_table.shape[1],
        })

        try:
            first_value_column_index = detect_first_numeric_value_column(raw_table)
            first_data_row_index = detect_first_category_data_row(
                raw_table=raw_table,
                first_value_column_index=first_value_column_index,
            )
        except Exception as error:
            structure_rows.append({
                "file_name": file_path.name,
                "sheet_name": sheet_name,
                "status": "failed",
                "error_message": f"{type(error).__name__}: {error}",
                "row_count": raw_table.shape[0],
                "column_count": raw_table.shape[1],
                "first_value_column_index": "",
                "category_column_count": "",
                "value_column_count": "",
                "first_data_excel_row": "",
                "numeric_total": "",
            })
            continue

        header_rows = detect_possible_header_rows(
            raw_table=raw_table,
            first_data_row_index=first_data_row_index,
        )

        category_column_count = first_value_column_index
        value_column_count = raw_table.shape[1] - first_value_column_index

        numeric_values = (
            raw_table
            .iloc[first_data_row_index:, first_value_column_index:]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        numeric_total = numeric_values.sum().sum()

        structure_rows.append({
            "file_name": file_path.name,
            "sheet_name": sheet_name,
            "status": "ok",
            "error_message": "",
            "row_count": raw_table.shape[0],
            "column_count": raw_table.shape[1],
            "first_value_column_index": first_value_column_index,
            "category_column_count": category_column_count,
            "value_column_count": value_column_count,
            "first_data_excel_row": first_data_row_index + 1,
            "numeric_total": numeric_total,
        })

        numeric_total_rows.append({
            "file_name": file_path.name,
            "sheet_name": sheet_name,
            "numeric_total_all_value_cells": numeric_total,
        })

        category_part = raw_table.iloc[first_data_row_index:, :first_value_column_index].copy()
        category_part = category_part.ffill()

        value_part = raw_table.iloc[first_data_row_index:, first_value_column_index:].copy()
        value_part_numeric = value_part.apply(pd.to_numeric, errors="coerce").fillna(0)

        for category_column_index in range(first_value_column_index):
            category_values = (
                category_part.iloc[:, category_column_index]
                .dropna()
                .apply(normalize_text)
            )

            category_values = [
                value
                for value in category_values.unique().tolist()
                if value != ""
            ]

            for category_value in category_values[:200]:
                category_value_rows.append({
                    "file_name": file_path.name,
                    "sheet_name": sheet_name,
                    "category_column_index": category_column_index,
                    "category_column_excel": category_column_index + 1,
                    "category_value": category_value,
                    "looks_like_age_label": looks_like_age_label(category_value),
                })

                if looks_like_age_label(category_value):
                    age_label_rows.append({
                        "file_name": file_path.name,
                        "sheet_name": sheet_name,
                        "category_column_index": category_column_index,
                        "category_column_excel": category_column_index + 1,
                        "age_label": category_value,
                    })

        preview_row_counter = 0

        for local_row_position, original_row_index in enumerate(value_part_numeric.index):
            if preview_row_counter >= 2000:
                break

            category_values = []

            for category_column_index in range(first_value_column_index):
                category_values.append(
                    normalize_text(category_part.iloc[local_row_position, category_column_index])
                )

            for value_column_index in value_part_numeric.columns:
                value = value_part_numeric.loc[original_row_index, value_column_index]

                if value == 0:
                    continue

                long_preview_rows.append({
                    "file_name": file_path.name,
                    "sheet_name": sheet_name,
                    "excel_row_number": original_row_index + 1,
                    "excel_value_column_number": value_column_index + 1,
                    "value_column_label": create_compact_column_label(
                        raw_table=raw_table,
                        header_rows=header_rows,
                        column_index=value_column_index,
                    ),
                    "category_values_joined": " || ".join(category_values),
                    "value": value,
                })

                preview_row_counter += 1

                if preview_row_counter >= 2000:
                    break

    return {
        "file_inventory_rows": file_inventory_rows,
        "structure_rows": structure_rows,
        "category_value_rows": category_value_rows,
        "age_label_rows": age_label_rows,
        "numeric_total_rows": numeric_total_rows,
        "long_preview_rows": long_preview_rows,
    }


# ============================================================
# 4. FŐ FUTTATÁS
# ============================================================

def run_table_profiling() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    all_file_inventory_rows = []
    all_structure_rows = []
    all_category_value_rows = []
    all_age_label_rows = []
    all_numeric_total_rows = []
    all_long_preview_rows = []

    for file_name in TARGET_FILES:
        file_path = RAW_HIER_TABLE_FOLDER / file_name

        if not file_path.exists():
            all_structure_rows.append({
                "file_name": file_name,
                "sheet_name": "",
                "status": "missing_file",
                "error_message": f"Nem található: {file_path}",
                "row_count": "",
                "column_count": "",
                "first_value_column_index": "",
                "category_column_count": "",
                "value_column_count": "",
                "first_data_excel_row": "",
                "numeric_total": "",
            })
            print(f"Hiányzó fájl: {file_path}")
            continue

        print(f"Profilozás: {file_path.name}")

        profile_result = profile_single_excel_file(file_path)

        all_file_inventory_rows.extend(profile_result["file_inventory_rows"])
        all_structure_rows.extend(profile_result["structure_rows"])
        all_category_value_rows.extend(profile_result["category_value_rows"])
        all_age_label_rows.extend(profile_result["age_label_rows"])
        all_numeric_total_rows.extend(profile_result["numeric_total_rows"])
        all_long_preview_rows.extend(profile_result["long_preview_rows"])

    pd.DataFrame(all_file_inventory_rows).to_csv(
        FILE_INVENTORY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_structure_rows).to_csv(
        STRUCTURE_PROFILE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_category_value_rows).to_csv(
        CATEGORY_VALUES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_age_label_rows).to_csv(
        AGE_LABELS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_numeric_total_rows).to_csv(
        NUMERIC_TOTALS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_long_preview_rows).to_csv(
        LONG_PREVIEW_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_table_profiling()