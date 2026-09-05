from pathlib import Path
import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# Itt módosíts, ha máshol vannak a flat táblák.
# ============================================================

RAW_FLAT_DATA_FOLDER = Path("../data/raw_flat_tables")
OUTPUT_FOLDER = Path("../outputs/flat_validation")

EXCEL_FILE_PATTERN = "*.xlsx"
DATA_SHEET_NAME = "Adattábla"


# ============================================================
# 2. FÁJLKEZELÉS
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    """
    Létrehozza az output mappát, ha még nem létezik.
    """
    output_folder.mkdir(parents=True, exist_ok=True)


def collect_excel_files(raw_data_folder: Path, file_pattern: str) -> list[Path]:
    """
    Összegyűjti az Excel fájlokat a megadott flat adat mappából.
    """
    excel_files = sorted(raw_data_folder.glob(file_pattern))

    if not excel_files:
        raise FileNotFoundError(
            f"Nem találtam Excel fájlokat ebben a mappában: {raw_data_folder.resolve()}"
        )

    return excel_files


def read_flat_excel_table(excel_file_path: Path) -> pd.DataFrame:
    """
    Beolvassa a flat Excel táblát.

    A flat táblák szerkezete:
    - első sor: településnevek
    - első oszlop: kategóriák
    - többi cella: számérték
    """
    raw_table = pd.read_excel(
        excel_file_path,
        sheet_name=DATA_SHEET_NAME,
        header=None,
        engine="openpyxl",
    )

    return raw_table


# ============================================================
# 3. FLAT TÁBLÁK ÁTALAKÍTÁSA HOSSZÚ FORMÁTUMRA
# ============================================================

def convert_flat_table_to_long_format(
    raw_table: pd.DataFrame,
    source_file_name: str,
) -> pd.DataFrame:
    """
    A flat táblát hosszú formátumra alakítja.

    Eredeti forma:
    category | settlement_1 | settlement_2 | settlement_3 | ...

    Új forma:
    source_file | category | settlement | value
    """

    # Első sorban vannak a településnevek, B oszloptól jobbra.
    settlement_names = raw_table.iloc[0, 1:].copy()

    # A kategóriák A oszlopban vannak, a 2. Excel-sortól lefelé.
    category_names = raw_table.iloc[1:, 0].copy()

    # A számos adatok B2-től indulnak.
    value_table = raw_table.iloc[1:, 1:].copy()
    value_table = value_table.apply(pd.to_numeric, errors="coerce")

    long_rows = []

    for row_position, category_name in enumerate(category_names):
        # Üres kategóriasorokat kihagyjuk.
        if pd.isna(category_name):
            continue

        original_excel_row_number = row_position + 2

        for column_position, settlement_name in enumerate(settlement_names):
            # Üres településoszlopokat kihagyjuk.
            if pd.isna(settlement_name):
                continue

            original_excel_column_number = column_position + 2

            raw_value = value_table.iloc[row_position, column_position]
            value = 0 if pd.isna(raw_value) else raw_value

            long_rows.append({
                "source_file": source_file_name,
                "excel_row_number": original_excel_row_number,
                "excel_column_number": original_excel_column_number,
                "category": str(category_name).strip(),
                "settlement": str(settlement_name).strip(),
                "value": value,
            })

    long_table = pd.DataFrame(long_rows)

    return long_table


# ============================================================
# 4. ÖSSZEFOGLALÓK ÉS AGGREGÁLÁSOK
# ============================================================

def summarize_flat_file_structure(
    raw_table: pd.DataFrame,
    source_file_name: str,
) -> dict:
    """
    Szerkezeti összefoglalót készít egy flat tábláról.
    """
    settlement_count = raw_table.iloc[0, 1:].notna().sum()
    category_count = raw_table.iloc[1:, 0].notna().sum()

    return {
        "source_file": source_file_name,
        "row_count_total": raw_table.shape[0],
        "column_count_total": raw_table.shape[1],
        "category_count": int(category_count),
        "settlement_count": int(settlement_count),
    }


def calculate_flat_row_totals(long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Kategóriánként országos összeget számol.

    Példa:
    Férfi -> országos férfi népesség
    Nő -> országos női népesség
    Egyszemélyes háztartás -> országos egyszemélyes háztartások száma
    """
    row_totals = (
        long_table
        .groupby(["source_file", "category"], dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "national_total"})
        .sort_values(["source_file", "category"])
    )

    return row_totals


def calculate_flat_settlement_totals(long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Településenként összesít.

    FIGYELEM:
    Ez nem mindig jelent értelmes teljes összeget,
    mert egy fájlban több alternatív kategóriabontás is lehet.
    Ezért ezt főleg technikai sanity checkként használjuk.
    """
    settlement_totals = (
        long_table
        .groupby(["source_file", "settlement"], dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "raw_settlement_total_all_categories"})
        .sort_values(["source_file", "settlement"])
    )

    return settlement_totals


def get_national_total_for_categories(
    row_totals: pd.DataFrame,
    source_file_contains: str,
    categories: list[str],
) -> int:
    """
    Megadott kategóriák országos összegét adja vissza egy adott fájlból.
    """
    filtered_rows = row_totals[
        row_totals["source_file"].str.contains(source_file_contains, case=False, regex=False)
        & row_totals["category"].isin(categories)
    ]

    return int(filtered_rows["national_total"].sum())


def create_validation_check_row(
    check_name: str,
    source_file_contains: str,
    categories: list[str],
    row_totals: pd.DataFrame,
    expected_total: int | None = None,
) -> dict:
    """
    Egy validációs sort készít.

    Ha van expected_total, akkor különbséget is számol.
    Ha nincs, akkor csak kiírja a kategóriacsoport összegét.
    """
    observed_total = get_national_total_for_categories(
        row_totals=row_totals,
        source_file_contains=source_file_contains,
        categories=categories,
    )

    difference = None
    ratio = None

    if expected_total is not None:
        difference = observed_total - expected_total
        ratio = observed_total / expected_total if expected_total != 0 else None

    return {
        "check_name": check_name,
        "source_file_contains": source_file_contains,
        "categories": " | ".join(categories),
        "observed_total": observed_total,
        "expected_total": expected_total,
        "difference": difference,
        "ratio": ratio,
    }


def create_flat_validation_checks(row_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Előre definiált sanity checkeket készít a flat táblákhoz.

    Ezek nem végleges validációk, hanem első körös ellenőrzések.
    """
    validation_rows = []

    # --------------------------------------------------------
    # Népességi tábla
    # --------------------------------------------------------

    validation_rows.append(
        create_validation_check_row(
            check_name="Népesség nem szerint: Férfi + Nő",
            source_file_contains="nepesseg",
            categories=["Férfi", "Nő"],
            row_totals=row_totals,
            expected_total=9_603_634,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Népesség 10 éves korcsoportok szerint",
            source_file_contains="nepesseg",
            categories=[
                "10 évesnél fiatalabb",
                "10–19 éves",
                "20–29 éves",
                "30–39 éves",
                "40–49 éves",
                "50–59 éves",
                "60–69 éves",
                "70–79 éves",
                "80–89 éves",
                "90 éves és idősebb",
            ],
            row_totals=row_totals,
            expected_total=9_603_634,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Gazdasági aktivitás flat tábla alapján",
            source_file_contains="nepesseg",
            categories=[
                "Foglalkoztatott",
                "Munkanélküli",
                "Ellátásban részesülő inaktív",
                "Eltartott",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Gazdasági aktivitás + 15 év alatti személy",
            source_file_contains="nepesseg",
            categories=[
                "15 évesnél fiatalabb személy",
                "Foglalkoztatott",
                "Munkanélküli",
                "Ellátásban részesülő inaktív",
                "Eltartott",
            ],
            row_totals=row_totals,
            expected_total=9_603_634,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="15-64 éves férfi + nő",
            source_file_contains="nepesseg",
            categories=[
                "15–64 éves férfi",
                "15–64 éves nő",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="65 éves és idősebb férfi + nő",
            source_file_contains="nepesseg",
            categories=[
                "65 éves és idősebb férfi",
                "65 éves és idősebb nő",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Családi állapot 15+ népesség",
            source_file_contains="nepesseg",
            categories=[
                "Nőtlen, hajadon",
                "Házas",
                "Elvált",
                "Özvegy",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Iskolai végzettség kategóriák",
            source_file_contains="nepesseg",
            categories=[
                "Általános iskola 8. évfolyamnál alacsonyabb",
                "Általános iskola 8. évfolyam",
                "Középfokú iskola érettségi nélkül, szakmai oklevéllel",
                "Érettségi",
                "Egyetem, főiskola stb. oklevéllel",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    # --------------------------------------------------------
    # Háztartási tábla
    # --------------------------------------------------------

    validation_rows.append(
        create_validation_check_row(
            check_name="Háztartások háztartásméret szerint",
            source_file_contains="haztartasok",
            categories=[
                "Egyszemélyes háztartás",
                "Kétszemélyes háztartás",
                "Háromszemélyes háztartás",
                "Négyszemélyes háztartás",
                "Ötszemélyes háztartás",
                "Hat vagy többszemélyes háztartás",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Háztartások: van/nincs foglalkoztatott",
            source_file_contains="haztartasok",
            categories=[
                "Van foglalkoztatott a háztartásban",
                "Nincs foglalkoztatott a háztartásban",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Háztartások: van/nincs 15 év alatti személy",
            source_file_contains="haztartasok",
            categories=[
                "Nincs 15 évesnél fiatalabb személy a háztartásban",
                "1 személy 15 évesnél fiatalabb a háztartásban",
                "2 személy 15 évesnél fiatalabb a háztartásban",
                "3 vagy több személy 15 évesnél fiatalabb a háztartásban",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Háztartások: van/nincs 65+ személy",
            source_file_contains="haztartasok",
            categories=[
                "Nincs 65 éves és idősebb személy a háztartásban",
                "1 személy 65 éves és idősebb a háztartásban",
                "2 személy 65 éves és idősebb a háztartásban",
                "3 vagy több személy 65 éves és idősebb a háztartásban",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    # --------------------------------------------------------
    # Lakásos tábla
    # --------------------------------------------------------

    validation_rows.append(
        create_validation_check_row(
            check_name="Lakott lakás",
            source_file_contains="lakasok",
            categories=[
                "Lakott lakás",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_rows.append(
        create_validation_check_row(
            check_name="Lakások szobaszám szerint",
            source_file_contains="lakasok",
            categories=[
                "1 szoba",
                "2 szoba",
                "3 szoba",
                "4 vagy több szoba",
            ],
            row_totals=row_totals,
            expected_total=None,
        )
    )

    validation_checks = pd.DataFrame(validation_rows)

    return validation_checks


# ============================================================
# 5. QUERY SEGÉDFÜGGVÉNYEK
# Ezeket később interaktív elemzéshez is használhatod.
# ============================================================

def query_flat_database_by_category_text(
    long_table: pd.DataFrame,
    category_text: str,
) -> pd.DataFrame:
    """
    Visszaadja azokat a sorokat, ahol a kategória tartalmazza a megadott szöveget.

    Példa:
    category_text = "Foglalkoztatott"
    """
    result = long_table[
        long_table["category"].str.contains(category_text, case=False, regex=False, na=False)
    ].copy()

    return result


def query_flat_database_by_settlement(
    long_table: pd.DataFrame,
    settlement_name: str,
) -> pd.DataFrame:
    """
    Visszaadja egy adott település összes flat adatát.

    Példa:
    settlement_name = "Pécs"
    """
    result = long_table[
        long_table["settlement"].str.contains(settlement_name, case=False, regex=False, na=False)
    ].copy()

    return result


# ============================================================
# 6. FŐ FUTTATÁSI FÜGGVÉNY
# ============================================================

def run_flat_table_validation() -> None:
    """
    Ez a fő függvény.

    Lépései:
    1. Flat Excel fájlok összegyűjtése
    2. Beolvasás
    3. Hosszú formátum készítése
    4. Országos kategóriaösszegek számítása
    5. Előre definiált sanity checkek számítása
    6. CSV outputok mentése
    """
    create_output_folder_if_missing(OUTPUT_FOLDER)

    excel_files = collect_excel_files(
        raw_data_folder=RAW_FLAT_DATA_FOLDER,
        file_pattern=EXCEL_FILE_PATTERN,
    )

    file_structure_summaries = []
    all_long_tables = []

    for excel_file_path in excel_files:
        print(f"Feldolgozás alatt: {excel_file_path.name}")

        raw_table = read_flat_excel_table(excel_file_path)

        file_structure_summary = summarize_flat_file_structure(
            raw_table=raw_table,
            source_file_name=excel_file_path.name,
        )
        file_structure_summaries.append(file_structure_summary)

        long_table = convert_flat_table_to_long_format(
            raw_table=raw_table,
            source_file_name=excel_file_path.name,
        )
        all_long_tables.append(long_table)

    combined_long_table = pd.concat(all_long_tables, ignore_index=True)

    file_structure_summary_table = pd.DataFrame(file_structure_summaries)
    row_totals = calculate_flat_row_totals(combined_long_table)
    settlement_totals = calculate_flat_settlement_totals(combined_long_table)
    validation_checks = create_flat_validation_checks(row_totals)

    # --------------------------------------------------------
    # Outputok mentése
    # --------------------------------------------------------

    file_structure_summary_table.to_csv(
        OUTPUT_FOLDER / "01_flat_file_structure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row_totals.to_csv(
        OUTPUT_FOLDER / "02_flat_row_totals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    settlement_totals.to_csv(
        OUTPUT_FOLDER / "03_flat_settlement_totals_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_checks.to_csv(
        OUTPUT_FOLDER / "04_flat_validation_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_long_table.to_csv(
        OUTPUT_FOLDER / "05_flat_combined_long_format_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Riportok helye: {OUTPUT_FOLDER.resolve()}")


# ============================================================
# 7. PROGRAM INDÍTÁSA
# Ezt ne módosítsd.
# ============================================================

if __name__ == "__main__":
    run_flat_table_validation()