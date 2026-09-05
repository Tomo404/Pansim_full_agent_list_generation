from pathlib import Path
import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# Itt kell módosítanod, ha máshol vannak a táblázatok.
# ============================================================

RAW_DATA_FOLDER = Path("../data/raw_hier_tables")
OUTPUT_FOLDER = Path("../outputs/validation")

EXCEL_FILE_PATTERN = "*.xlsx"

PREFERRED_DATA_SHEET_NAMES = [
    "Adattábla",
    "Sheet1",
]


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK FÁJLKEZELÉSHEZ
# Ezeket elsőre nem kell módosítanod.
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    """
    Létrehozza az output mappát, ha még nem létezik.
    """
    output_folder.mkdir(parents=True, exist_ok=True)


def collect_excel_files(raw_data_folder: Path, file_pattern: str) -> list[Path]:
    """
    Összegyűjti az Excel fájlokat a megadott mappából.
    """
    excel_files = sorted(raw_data_folder.glob(file_pattern))

    if not excel_files:
        raise FileNotFoundError(
            f"Nem találtam Excel fájlokat ebben a mappában: {raw_data_folder.resolve()}"
        )

    return excel_files


def choose_data_sheet_name(excel_file_path: Path) -> str:
    """
    Kiválasztja, hogy melyik munkalapot olvassuk be.

    A KSH-táblák nagy részében az adatlap neve 'Adattábla'.
    Néhány fájlban viszont lehet például 'Sheet1'.
    """
    excel_file = pd.ExcelFile(excel_file_path)
    available_sheet_names = excel_file.sheet_names

    for preferred_sheet_name in PREFERRED_DATA_SHEET_NAMES:
        if preferred_sheet_name in available_sheet_names:
            return preferred_sheet_name

    return available_sheet_names[0]


def read_excel_sheet_without_header(excel_file_path: Path) -> pd.DataFrame:
    """
    Beolvassa az Excel táblát úgy, hogy nem feltételez fix fejlécet.

    Azért header=None, mert ezekben a KSH táblákban:
    - az első 1-2 sor földrajzi fejléc,
    - a bal oldali oszlopok kategóriák,
    - a számoszlopok csak később kezdődnek.
    """
    sheet_name = choose_data_sheet_name(excel_file_path)

    raw_table = pd.read_excel(
        excel_file_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )

    return raw_table


# ============================================================
# 3. TÁBLASZERKEZET FELISMERÉSE
# Ezeket sem kell elsőre módosítanod.
# ============================================================

def convert_table_to_numeric(raw_table: pd.DataFrame) -> pd.DataFrame:
    """
    Megpróbálja a cellákat számmá alakítani.

    Ami nem szám, abból NaN lesz.
    Ez segít felismerni, hol kezdődnek a tényleges adatértékek.
    """
    return raw_table.apply(pd.to_numeric, errors="coerce")


def detect_first_numeric_data_row(raw_table: pd.DataFrame) -> int:
    """
    Megkeresi az első olyan sort, ahol már vannak számértékek.

    A legtöbb táblánál ez a 3. Excel-sor, Python indexben 2.
    """
    numeric_table = convert_table_to_numeric(raw_table)

    rows_with_numbers = numeric_table.notna().sum(axis=1)

    for row_index, number_count in rows_with_numbers.items():
        if number_count > 0:
            return row_index

    raise ValueError("Nem találtam számos adatot a táblában.")

def detect_first_category_data_row(
    raw_table: pd.DataFrame,
    first_value_column_index: int,
) -> int:
    """
    Megkeresi az első tényleges adatsort a bal oldali kategóriaoszlopok alapján.

    Erre azért van szükség, mert néhány KSH táblában az első kategóriasorok
    számos értékei üresek lehetnek, de ettől még a kategóriaszerkezet már
    ott kezdődik.

    Példa:
    15 évesnél fiatalabb foglalkoztatott = üres / 0,
    de a 'Férfi' és '15 évesnél fiatalabb' kategóriát nem szabad elveszíteni.
    """
    category_area = raw_table.iloc[2:, :first_value_column_index]

    rows_with_category_text = category_area.notna().sum(axis=1)

    for row_index, category_count in rows_with_category_text.items():
        if category_count > 0:
            return row_index

    raise ValueError("Nem találtam kategóriaadatot a táblában.")

def detect_first_numeric_data_column(raw_table: pd.DataFrame) -> int:
    """
    Megkeresi az első olyan oszlopot, ahol már számos adatok vannak.

    Ez általában C, D, E vagy F környéke.
    Python indexben:
    C = 2
    D = 3
    E = 4
    F = 5
    """
    numeric_table = convert_table_to_numeric(raw_table)

    columns_with_numbers = numeric_table.notna().sum(axis=0)

    for column_index, number_count in columns_with_numbers.items():
        if number_count > 0:
            return column_index

    raise ValueError("Nem találtam számos adatot tartalmazó oszlopot.")


def split_category_and_value_columns(raw_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Szétválasztja a táblát két részre:

    1. category_part:
       bal oldali kategóriaoszlopok
       például nem, korcsoport, végzettség, aktivitás

    2. value_part:
       jobb oldali számoszlopok
       például Budapest, Pest vármegye, Baranya vármegye stb.
    """
    first_value_column_index = detect_first_numeric_data_column(raw_table)
    first_data_row_index = detect_first_category_data_row(
        raw_table=raw_table,
        first_value_column_index=first_value_column_index,
    )

    category_part = raw_table.iloc[first_data_row_index:, :first_value_column_index].copy()
    value_part = raw_table.iloc[first_data_row_index:, first_value_column_index:].copy()

    return category_part, value_part


def build_geography_headers(raw_table: pd.DataFrame) -> pd.DataFrame:
    """
    A felső két sorból földrajzi fejléceket épít.

    1. fejlécsor: vármegye / Budapest
    2. fejlécsor: településtípus
       például Főváros, Megyei jogú város(ok), Egyéb város(ok), Község(ek)
    """
    first_value_column_index = detect_first_numeric_data_column(raw_table)

    geography_header_rows = raw_table.iloc[0:2, first_value_column_index:].copy()

    # A vármegye neve csak az első oszlopban szerepel, utána üres cellák vannak.
    # Ezért jobbra kitöltjük.
    counties = geography_header_rows.iloc[0].ffill()

    # A településtípus néha üres lehet, ezt később "Nincs megadva" értékre cseréljük.
    settlement_types = geography_header_rows.iloc[1].fillna("Nincs megadva")

    geography_headers = pd.DataFrame({
        "value_column_index": geography_header_rows.columns,
        "county": counties.values,
        "settlement_type": settlement_types.values,
    })

    return geography_headers


def clean_category_columns(category_part: pd.DataFrame) -> pd.DataFrame:
    """
    Kitölti lefelé a kategóriaoszlopokat.

    Erre azért van szükség, mert az Excelben sok cella vizuálisan összevonva van,
    pandasban viszont ezek NaN-ként jelennek meg.

    Példa:
    Férfi | 15–19 éves | Foglalkoztatott | ...
    NaN   | NaN        | NaN             | ...
    NaN   | NaN        | Munkanélküli    | ...

    Ebből ezt csináljuk:
    Férfi | 15–19 éves | Foglalkoztatott | ...
    Férfi | 15–19 éves | Foglalkoztatott | ...
    Férfi | 15–19 éves | Munkanélküli    | ...
    """
    cleaned_categories = category_part.ffill()

    cleaned_categories = cleaned_categories.rename(
        columns={
            column_index: f"category_{position + 1}"
            for position, column_index in enumerate(cleaned_categories.columns)
        }
    )

    return cleaned_categories


# ============================================================
# 4. HOSSZÚ FORMÁTUMÚ TÁBLA ÉPÍTÉSE
# Ez lesz a későbbi aggregálás alapja.
# ============================================================

def convert_hierarchical_table_to_long_format(
    raw_table: pd.DataFrame,
    source_file_name: str,
) -> pd.DataFrame:
    """
    A KSH hierarchikus táblát hosszú formátumra alakítja.

    Eredeti forma:
    kategóriák bal oldalt, földrajzi helyek oszlopokban.

    Új forma:
    minden sor egy kategória + földrajzi hely + érték kombináció.

    Példa:
    source_file | category_1 | category_2 | county | settlement_type | value
    """
    first_value_column_index = detect_first_numeric_data_column(raw_table)
    first_data_row_index = detect_first_category_data_row(
        raw_table=raw_table,
        first_value_column_index=first_value_column_index,
    )

    category_part, value_part = split_category_and_value_columns(raw_table)
    cleaned_categories = clean_category_columns(category_part)
    geography_headers = build_geography_headers(raw_table)

    value_part_numeric = value_part.apply(pd.to_numeric, errors="coerce")

    long_rows = []

    for local_row_position, original_row_index in enumerate(value_part_numeric.index):
        category_values = cleaned_categories.iloc[local_row_position].to_dict()

        for value_column_index in value_part_numeric.columns:
            raw_value = value_part_numeric.loc[original_row_index, value_column_index]

            # KSH táblákban az üres cella sokszor nulla / nem fordul elő.
            # Aggregálásnál ezt 0-nak vesszük.
            value = 0 if pd.isna(raw_value) else raw_value

            geography_row = geography_headers[
                geography_headers["value_column_index"] == value_column_index
            ].iloc[0]

            long_row = {
                "source_file": source_file_name,
                "excel_row_number": original_row_index + 1,
                "excel_value_column_number": value_column_index + 1,
                "county": geography_row["county"],
                "settlement_type": geography_row["settlement_type"],
                "value": value,
            }

            long_row.update(category_values)

            long_rows.append(long_row)

    long_table = pd.DataFrame(long_rows)

    return long_table


# ============================================================
# 5. AGGREGÁLÁSOK ÉS VALIDÁCIÓS TÁBLÁK
# Ezekből lesznek az output CSV-k.
# ============================================================

def summarize_one_excel_file(excel_file_path: Path) -> dict:
    """
    Egy fájlról készít szerkezeti összefoglalót.
    """
    raw_table = read_excel_sheet_without_header(excel_file_path)

    first_value_column_index = detect_first_numeric_data_column(raw_table)
    first_data_row_index = detect_first_category_data_row(
        raw_table=raw_table,
        first_value_column_index=first_value_column_index,
    )

    category_column_count = first_value_column_index
    value_column_count = raw_table.shape[1] - first_value_column_index
    data_row_count = raw_table.shape[0] - first_data_row_index

    return {
        "source_file": excel_file_path.name,
        "row_count_total": raw_table.shape[0],
        "column_count_total": raw_table.shape[1],
        "first_data_row_excel_number": first_data_row_index + 1,
        "first_value_column_excel_number": first_value_column_index + 1,
        "category_column_count": category_column_count,
        "value_column_count": value_column_count,
        "data_row_count": data_row_count,
    }


def aggregate_values_by_file_and_geography(long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Megnézi, hogy fájlonként, vármegyénként és településtípusonként
    mennyi az összesített érték.

    FIGYELEM:
    Ez nem mindig jelent valódi népességösszeget, mert néhány táblában
    lehetnek átfedő / aggregált kategóriák.
    Ezért ez első körben sanity check, nem végső validáció.
    """
    grouped = (
        long_table
        .groupby(["source_file", "county", "settlement_type"], dropna=False)["value"]
        .sum()
        .reset_index()
        .sort_values(["source_file", "county", "settlement_type"])
    )

    return grouped


def aggregate_values_by_file(long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Fájlonkénti teljes nyers összeg.

    FIGYELEM:
    Ez több táblánál túl nagy lehet, ha a sorok között aggregált kategóriák
    vagy alternatív bontások vannak.
    """
    grouped = (
        long_table
        .groupby("source_file", dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "raw_total_value_all_rows_all_columns"})
        .sort_values("source_file")
    )

    return grouped


def calculate_row_totals(long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Eredeti Excel-soronként kiszámolja a sorösszeget.

    Ez segíthet megtalálni, mely sorok adnak nagy aggregált értéket.
    """
    category_columns = [
        column_name
        for column_name in long_table.columns
        if column_name.startswith("category_")
    ]

    row_totals = (
        long_table
        .groupby(["source_file", "excel_row_number"] + category_columns, dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "row_total"})
        .sort_values(["source_file", "excel_row_number"])
    )

    return row_totals


def find_possible_aggregate_rows(row_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Megpróbálja jelölni a gyanús aggregált sorokat.

    Ez csak segédellenőrzés.
    Nem dönt automatikusan arról, hogy egy sort ki kell-e hagyni.
    """
    category_columns = [
        column_name
        for column_name in row_totals.columns
        if column_name.startswith("category_")
    ]

    aggregate_keywords = [
        "összes",
        "összesen",
        "együtt",
        "mind",
        "teljes",
    ]

    possible_aggregate_mask = pd.Series(False, index=row_totals.index)

    for category_column in category_columns:
        category_text = row_totals[category_column].astype(str).str.lower()

        for keyword in aggregate_keywords:
            possible_aggregate_mask = possible_aggregate_mask | category_text.str.contains(
                keyword,
                na=False,
                regex=False,
            )

    possible_aggregate_rows = row_totals[possible_aggregate_mask].copy()

    return possible_aggregate_rows


def check_expected_national_population(
    summary_by_file: pd.DataFrame,
    expected_population: int = 9_603_634,
) -> pd.DataFrame:
    """
    Összeveti a fájlonkénti nyers összeget a 2022-es országos népességgel.

    Fontos:
    Nem minden hierarchikus tábla fog pontosan 9 603 634-et adni.
    Például:
    - csak 15 éves és idősebb népességre vonatkozó tábla,
    - csak foglalkoztatottakra vonatkozó tábla,
    - egészségi állapotnál többféle bontás egymás alatt.

    Ezért ez csak figyelmeztető összehasonlítás.
    """
    checked = summary_by_file.copy()

    checked["expected_population_reference"] = expected_population
    checked["difference_from_expected_population"] = (
        checked["raw_total_value_all_rows_all_columns"] - expected_population
    )
    checked["ratio_to_expected_population"] = (
        checked["raw_total_value_all_rows_all_columns"] / expected_population
    )

    return checked


# ============================================================
# 6. FŐ FUTTATÁSI FÜGGVÉNY
# Ha később módosítasz, elsőre főleg itt érdemes.
# ============================================================

def run_hierarchical_table_validation() -> None:
    """
    Ez a fő függvény.

    Lépései:
    1. Excel fájlok összegyűjtése
    2. Táblaszerkezet felismerése
    3. Hosszú formátum létrehozása
    4. Aggregálások elkészítése
    5. CSV riportok mentése
    """
    create_output_folder_if_missing(OUTPUT_FOLDER)

    excel_files = collect_excel_files(
        raw_data_folder=RAW_DATA_FOLDER,
        file_pattern=EXCEL_FILE_PATTERN,
    )

    file_structure_summaries = []
    all_long_tables = []

    for excel_file_path in excel_files:
        print(f"Feldolgozás alatt: {excel_file_path.name}")

        raw_table = read_excel_sheet_without_header(excel_file_path)

        file_summary = summarize_one_excel_file(excel_file_path)
        file_structure_summaries.append(file_summary)

        long_table = convert_hierarchical_table_to_long_format(
            raw_table=raw_table,
            source_file_name=excel_file_path.name,
        )

        all_long_tables.append(long_table)

    combined_long_table = pd.concat(all_long_tables, ignore_index=True)

    file_structure_summary_table = pd.DataFrame(file_structure_summaries)

    geography_totals = aggregate_values_by_file_and_geography(combined_long_table)
    file_totals = aggregate_values_by_file(combined_long_table)
    file_totals_checked = check_expected_national_population(file_totals)

    row_totals = calculate_row_totals(combined_long_table)
    possible_aggregate_rows = find_possible_aggregate_rows(row_totals)

    # Output fájlok mentése
    file_structure_summary_table.to_csv(
        OUTPUT_FOLDER / "01_file_structure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    geography_totals.to_csv(
        OUTPUT_FOLDER / "02_geography_totals_by_file.csv",
        index=False,
        encoding="utf-8-sig",
    )

    file_totals_checked.to_csv(
        OUTPUT_FOLDER / "03_file_totals_checked_against_population.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row_totals.to_csv(
        OUTPUT_FOLDER / "04_row_totals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    possible_aggregate_rows.to_csv(
        OUTPUT_FOLDER / "05_possible_aggregate_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_long_table.to_csv(
        OUTPUT_FOLDER / "06_combined_long_format_table.csv",
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
    run_hierarchical_table_validation()