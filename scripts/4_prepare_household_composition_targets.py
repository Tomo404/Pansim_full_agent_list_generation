from pathlib import Path
import math

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# Itt módosíts, ha máshol vannak a fájljaid.
# ============================================================

HIER_HOMESIZE_FILE = Path("../data/raw_hier_tables/hier_homesize_agegroup_employment.xlsx")

HOUSEHOLD_GENERATION_TARGETS_CSV = Path(
    "../outputs/household_generation/01_household_generation_targets.csv"
)

OUTPUT_FOLDER = Path("../outputs/household_generation")

HIER_DATA_SHEET_NAME = "Adattábla"


# Ezeket használjuk a flat household target oszlopok és a hier household size kategóriák
# egységesítésére.
HOUSEHOLD_SIZE_CATEGORY_TO_SIZE_LABEL = {
    "Egyszemélyes háztartás": "1_person",
    "Kétszemélyes háztartás": "2_person",
    "Háromszemélyes háztartás": "3_person",
    "Négyszemélyes háztartás": "4_person",
    "Ötszemélyes háztartás": "5_person",
    "Hat vagy többszemélyes háztartás": "6plus_person",
}

HOUSEHOLD_TARGET_SIZE_COLUMNS = {
    "households_1_person": "1_person",
    "households_2_person": "2_person",
    "households_3_person": "3_person",
    "households_4_person": "4_person",
    "households_5_person": "5_person",
    "households_6plus_person": "6plus_person",
}


# ============================================================
# 2. ÁLTALÁNOS SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    """
    Létrehozza az output mappát, ha még nem létezik.
    """
    output_folder.mkdir(parents=True, exist_ok=True)


def convert_count_to_integer(raw_value) -> int:
    """
    Biztonságosan egész számmá alakít egy darabszámot.
    """
    if pd.isna(raw_value):
        return 0

    return int(round(float(raw_value)))


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Nullával osztás ellen védett osztás.
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator

def normalize_county_name(raw_county_name: str) -> str:
    """
    Vármegye-nevek egységesítése a hier és generated táblák összekötéséhez.

    Példák:
    'Baranya vármegye' -> 'Baranya'
    'Gyor-Moson-Sopron' -> 'Győr-Moson-Sopron'
    'fováros' -> 'Budapest'
    """
    if pd.isna(raw_county_name):
        return ""

    county_name = str(raw_county_name).strip()

    if county_name.endswith(" vármegye"):
        county_name = county_name.replace(" vármegye", "").strip()

    manual_county_corrections = {
        "Gyor-Moson-Sopron": "Győr-Moson-Sopron",
        "fováros": "Budapest",
        "Fováros": "Budapest",
        "főváros": "Budapest",
        "Főváros": "Budapest",
    }

    return manual_county_corrections.get(county_name, county_name)


def normalize_settlement_type(raw_settlement_type: str) -> str:
    """
    Településtípus-nevek egységesítése.

    Példa:
    'Fováros' -> 'Főváros'
    """
    if pd.isna(raw_settlement_type):
        return ""

    settlement_type = str(raw_settlement_type).strip()

    manual_type_corrections = {
        "Fováros": "Főváros",
        "fováros": "Főváros",
    }

    return manual_type_corrections.get(settlement_type, settlement_type)

# ============================================================
# 3. HIER HOMESIZE TÁBLA BEOLVASÁSA ÉS LONG FORMÁTUM
# ============================================================

def read_hier_homesize_table_without_header(hier_homesize_file: Path) -> pd.DataFrame:
    """
    Beolvassa a hier_homesize_agegroup_employment Excel fájlt fejléc nélkül.

    A táblában:
    - felső 2 sor: vármegye + településtípus fejléc
    - A-C oszlop: kategóriák
      A: háztartásméret
      B: háztartás korösszetétele
      C: háztartás foglalkoztatottsági összetétele
    - D oszloptól jobbra: értékek
    """
    raw_table = pd.read_excel(
        hier_homesize_file,
        sheet_name=HIER_DATA_SHEET_NAME,
        header=None,
        engine="openpyxl",
    )

    return raw_table


def detect_first_numeric_value_column(raw_table: pd.DataFrame) -> int:
    """
    Megkeresi, hol kezdődnek a számoszlopok.

    Ennél a táblánál várhatóan D oszlop, Python indexben 3.
    De nem hardcode-oljuk, hanem felismerjük.
    """
    numeric_table = raw_table.apply(pd.to_numeric, errors="coerce")
    numeric_counts_by_column = numeric_table.notna().sum(axis=0)

    for column_index, numeric_count in numeric_counts_by_column.items():
        if numeric_count > 0:
            return column_index

    raise ValueError("Nem találtam számos értékoszlopot a homesize táblában.")


def build_geography_header_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    """
    A felső 2 sorból földrajzi fejlécet épít.

    Output:
    value_column_index
    county_name
    settlement_type
    """
    first_value_column_index = detect_first_numeric_value_column(raw_table)

    geography_header_rows = raw_table.iloc[0:2, first_value_column_index:].copy()

    county_names = geography_header_rows.iloc[0].ffill()
    settlement_types = geography_header_rows.iloc[1].fillna("Nincs megadva")

    geography_header_table = pd.DataFrame({
        "value_column_index": geography_header_rows.columns,
        "county_name": county_names.values,
        "settlement_type": settlement_types.values,
    })

    return geography_header_table


def convert_hier_homesize_table_to_long_format(raw_table: pd.DataFrame) -> pd.DataFrame:
    """
    A hier_homesize táblát hosszú formátumra alakítja.

    Output:
    household_size_category
    household_age_composition
    household_employment_composition
    county_name
    settlement_type
    value
    """
    first_value_column_index = detect_first_numeric_value_column(raw_table)

    # A tényleges kategóriák a 3. Excel-sortól indulnak, Python indexben 2.
    first_data_row_index = 2

    category_part = raw_table.iloc[first_data_row_index:, :first_value_column_index].copy()
    value_part = raw_table.iloc[first_data_row_index:, first_value_column_index:].copy()

    # Az Excelben vizuálisan összevont cellák pandasban üresek,
    # ezért lefelé kitöltjük a kategóriákat.
    category_part = category_part.ffill()

    category_part = category_part.rename(
        columns={
            0: "household_size_category",
            1: "household_age_composition",
            2: "household_employment_composition",
        }
    )

    value_part_numeric = value_part.apply(pd.to_numeric, errors="coerce").fillna(0)

    geography_header_table = build_geography_header_table(raw_table)

    long_rows = []

    for local_row_position, original_row_index in enumerate(value_part_numeric.index):
        category_row = category_part.iloc[local_row_position]

        household_size_category = category_row["household_size_category"]
        household_age_composition = category_row["household_age_composition"]
        household_employment_composition = category_row["household_employment_composition"]

        # Üres vagy nem várt háztartásméret kategóriák kihagyása.
        if household_size_category not in HOUSEHOLD_SIZE_CATEGORY_TO_SIZE_LABEL:
            continue

        for value_column_index in value_part_numeric.columns:
            raw_value = value_part_numeric.loc[original_row_index, value_column_index]
            value = convert_count_to_integer(raw_value)

            geography_row = geography_header_table[
                geography_header_table["value_column_index"] == value_column_index
            ].iloc[0]

            long_rows.append({
                "excel_row_number": original_row_index + 1,
                "excel_value_column_number": value_column_index + 1,
                "household_size_category": household_size_category,
                "household_size_label": HOUSEHOLD_SIZE_CATEGORY_TO_SIZE_LABEL[
                    household_size_category
                ],
                "household_age_composition": household_age_composition,
                "household_employment_composition": household_employment_composition,
                "county_name": normalize_county_name(geography_row["county_name"]),
                "settlement_type": normalize_settlement_type(geography_row["settlement_type"]),
                "value": value,
            })

    long_table = pd.DataFrame(long_rows)

    return long_table


# ============================================================
# 4. HIER KOMPOZÍCIÓS ARÁNYOK SZÁMÍTÁSA
# ============================================================

def calculate_hier_composition_totals(long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Összesíti a hier táblát:
    vármegye × településtípus × háztartásméret szinten.
    """
    totals = (
        long_table
        .groupby(
            ["county_name", "settlement_type", "household_size_label"],
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "hier_total_households"})
        .sort_values(["county_name", "settlement_type", "household_size_label"])
    )

    return totals


def calculate_hier_composition_probabilities(long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Kiszámolja, hogy adott vármegye + településtípus + háztartásméret mellett
    milyen arányban fordulnak elő a korösszetétel × foglalkoztatottsági típusok.

    Ez lesz a későbbi rávetítés alapja.
    """
    composition_counts = (
        long_table
        .groupby(
            [
                "county_name",
                "settlement_type",
                "household_size_label",
                "household_age_composition",
                "household_employment_composition",
            ],
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "composition_count"})
    )

    group_totals = (
        composition_counts
        .groupby(
            ["county_name", "settlement_type", "household_size_label"],
            dropna=False,
        )["composition_count"]
        .sum()
        .reset_index()
        .rename(columns={"composition_count": "hier_total_households_in_group"})
    )

    probabilities = composition_counts.merge(
        group_totals,
        on=["county_name", "settlement_type", "household_size_label"],
        how="left",
    )

    probabilities["composition_probability"] = probabilities.apply(
        lambda row: safe_divide(
            row["composition_count"],
            row["hier_total_households_in_group"],
        ),
        axis=1,
    )

    probabilities = probabilities.sort_values(
        [
            "county_name",
            "settlement_type",
            "household_size_label",
            "composition_probability",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    return probabilities


def calculate_national_fallback_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    """
    Országos fallback arányokat készít household_size szerint.

    Erre akkor lehet szükség, ha valamelyik vármegye + településtípus + méret kombinációban
    nincs hier adat, de a generated householdok között mégis van ilyen típus.
    """
    national_counts = (
        probabilities
        .groupby(
            [
                "household_size_label",
                "household_age_composition",
                "household_employment_composition",
            ],
            dropna=False,
        )["composition_count"]
        .sum()
        .reset_index()
    )

    national_totals = (
        national_counts
        .groupby("household_size_label", dropna=False)["composition_count"]
        .sum()
        .reset_index()
        .rename(columns={"composition_count": "national_total_for_size"})
    )

    fallback = national_counts.merge(
        national_totals,
        on="household_size_label",
        how="left",
    )

    fallback["composition_probability"] = fallback.apply(
        lambda row: safe_divide(
            row["composition_count"],
            row["national_total_for_size"],
        ),
        axis=1,
    )

    fallback = fallback.rename(
        columns={
            "composition_count": "fallback_composition_count",
            "national_total_for_size": "fallback_total_for_size",
        }
    )

    return fallback


# ============================================================
# 5. GENERATED HOUSEHOLD TARGETEK ELŐKÉSZÍTÉSE
# ============================================================

def load_household_generation_targets(targets_csv: Path) -> pd.DataFrame:
    """
    Beolvassa a korábbi household generation target táblát.
    """
    targets = pd.read_csv(targets_csv)

    return targets


def convert_household_generation_targets_to_long_format(targets: pd.DataFrame) -> pd.DataFrame:
    """
    A településszintű generated household target táblát hosszú formátumra alakítja.

    Eredeti:
    settlement + households_1_person + households_2_person + ...

    Új:
    settlement + household_size_label + generated_household_count
    """
    long_rows = []

    for _, row in targets.iterrows():
        for source_column, household_size_label in HOUSEHOLD_TARGET_SIZE_COLUMNS.items():
            generated_household_count = convert_count_to_integer(row[source_column])

            long_rows.append({
                "settlement_key": row["settlement_key"],
                "settlement_name": row["settlement_name_flat"],
                "settlement_ksh_code": row["settlement_ksh_code"],
                "county_name": normalize_county_name(row["county_name"]),
                "settlement_type": normalize_settlement_type(row["settlement_type"]),
                "household_size_label": household_size_label,
                "generated_household_count": generated_household_count,
            })

    generated_targets_long = pd.DataFrame(long_rows)

    return generated_targets_long


def aggregate_generated_household_targets_by_area(
    generated_targets_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    A generated household targeteket összesíti:
    vármegye × településtípus × háztartásméret szinten.
    """
    area_targets = (
        generated_targets_long
        .groupby(
            ["county_name", "settlement_type", "household_size_label"],
            dropna=False,
        )["generated_household_count"]
        .sum()
        .reset_index()
        .sort_values(["county_name", "settlement_type", "household_size_label"])
    )

    return area_targets


def compare_generated_targets_with_hier_totals(
    generated_area_targets: pd.DataFrame,
    hier_totals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Összeveti a flat/generated household targeteket a hier homesize táblából kijövő totalokkal.

    Fontos:
    Nem kell 100%-ban egyezniük.
    A flat tábla a településszintű darabszámforrás,
    a hier tábla pedig kompozíciós arányforrás.
    """
    comparison = generated_area_targets.merge(
        hier_totals,
        on=["county_name", "settlement_type", "household_size_label"],
        how="outer",
    )

    comparison["generated_household_count"] = pd.to_numeric(
        comparison["generated_household_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["hier_total_households"] = pd.to_numeric(
        comparison["hier_total_households"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["difference_generated_minus_hier"] = (
        comparison["generated_household_count"]
        - comparison["hier_total_households"]
    )

    comparison["ratio_generated_to_hier"] = comparison.apply(
        lambda row: safe_divide(
            row["generated_household_count"],
            row["hier_total_households"],
        ),
        axis=1,
    )

    comparison = comparison.sort_values(
        ["county_name", "settlement_type", "household_size_label"]
    ).reset_index(drop=True)

    return comparison


# ============================================================
# 6. INTEGER ALLOKÁCIÓ: ARÁNYOKBÓL DARABSZÁMOK
# ============================================================

def allocate_integer_counts_by_largest_remainder(
    probabilities_for_group: pd.DataFrame,
    total_count_to_allocate: int,
) -> pd.DataFrame:
    """
    Arányok alapján egész darabszámokat oszt ki úgy,
    hogy a kiosztott darabszámok összege pontosan total_count_to_allocate legyen.

    Módszer:
    1. expected_count = probability * total
    2. floor_count = floor(expected_count)
    3. maradékot a legnagyobb törtrészek kapják
    """
    allocation = probabilities_for_group.copy()

    if total_count_to_allocate <= 0 or allocation.empty:
        allocation["allocated_household_count"] = 0
        return allocation

    allocation["expected_household_count"] = (
        allocation["composition_probability"] * total_count_to_allocate
    )

    allocation["allocated_household_count"] = allocation["expected_household_count"].apply(
        math.floor
    )

    already_allocated = int(allocation["allocated_household_count"].sum())
    remaining_to_allocate = total_count_to_allocate - already_allocated

    allocation["allocation_remainder"] = (
        allocation["expected_household_count"]
        - allocation["allocated_household_count"]
    )

    allocation = allocation.sort_values(
        ["allocation_remainder", "expected_household_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if remaining_to_allocate > 0:
        allocation.loc[
            allocation.index < remaining_to_allocate,
            "allocated_household_count"
        ] += 1

    allocation = allocation.sort_values(
        [
            "allocated_household_count",
            "expected_household_count",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return allocation


def allocate_area_composition_targets(
    generated_area_targets: pd.DataFrame,
    probabilities: pd.DataFrame,
    national_fallback_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vármegye + településtípus + háztartásméret szinten osztja szét a generated targeteket
    korösszetétel × foglalkoztatottsági összetétel kategóriákra.
    """
    allocated_tables = []

    for _, target_row in generated_area_targets.iterrows():
        county_name = target_row["county_name"]
        settlement_type = target_row["settlement_type"]
        household_size_label = target_row["household_size_label"]
        total_count_to_allocate = convert_count_to_integer(
            target_row["generated_household_count"]
        )

        probabilities_for_group = probabilities[
            (probabilities["county_name"] == county_name)
            & (probabilities["settlement_type"] == settlement_type)
            & (probabilities["household_size_label"] == household_size_label)
        ].copy()

        used_fallback = False

        if probabilities_for_group.empty:
            probabilities_for_group = national_fallback_probabilities[
                national_fallback_probabilities["household_size_label"] == household_size_label
            ].copy()

            probabilities_for_group["county_name"] = county_name
            probabilities_for_group["settlement_type"] = settlement_type
            probabilities_for_group["hier_total_households_in_group"] = 0
            probabilities_for_group["composition_count"] = probabilities_for_group[
                "fallback_composition_count"
            ]

            used_fallback = True

        allocated_group = allocate_integer_counts_by_largest_remainder(
            probabilities_for_group=probabilities_for_group,
            total_count_to_allocate=total_count_to_allocate,
        )

        allocated_group["generated_group_total"] = total_count_to_allocate
        allocated_group["used_national_fallback_probability"] = used_fallback

        allocated_tables.append(allocated_group)

    if not allocated_tables:
        return pd.DataFrame()

    allocated_area_targets = pd.concat(allocated_tables, ignore_index=True)

    allocated_area_targets = allocated_area_targets[
        allocated_area_targets["allocated_household_count"] > 0
    ].copy()

    return allocated_area_targets


def allocate_settlement_composition_targets(
    generated_targets_long: pd.DataFrame,
    probabilities: pd.DataFrame,
    national_fallback_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Településszinten osztja szét a generated household targeteket
    korösszetétel × foglalkoztatottsági összetétel kategóriákra.

    Ez fontos, mert később ebből lehet majd konkrét householdokra címkét osztani.
    """
    allocated_tables = []

    for _, target_row in generated_targets_long.iterrows():
        settlement_key = target_row["settlement_key"]
        settlement_name = target_row["settlement_name"]
        settlement_ksh_code = target_row["settlement_ksh_code"]
        county_name = target_row["county_name"]
        settlement_type = target_row["settlement_type"]
        household_size_label = target_row["household_size_label"]
        total_count_to_allocate = convert_count_to_integer(
            target_row["generated_household_count"]
        )

        if total_count_to_allocate <= 0:
            continue

        probabilities_for_group = probabilities[
            (probabilities["county_name"] == county_name)
            & (probabilities["settlement_type"] == settlement_type)
            & (probabilities["household_size_label"] == household_size_label)
        ].copy()

        used_fallback = False

        if probabilities_for_group.empty:
            probabilities_for_group = national_fallback_probabilities[
                national_fallback_probabilities["household_size_label"] == household_size_label
            ].copy()

            probabilities_for_group["county_name"] = county_name
            probabilities_for_group["settlement_type"] = settlement_type
            probabilities_for_group["hier_total_households_in_group"] = 0
            probabilities_for_group["composition_count"] = probabilities_for_group[
                "fallback_composition_count"
            ]

            used_fallback = True

        allocated_group = allocate_integer_counts_by_largest_remainder(
            probabilities_for_group=probabilities_for_group,
            total_count_to_allocate=total_count_to_allocate,
        )

        allocated_group["settlement_key"] = settlement_key
        allocated_group["settlement_name"] = settlement_name
        allocated_group["settlement_ksh_code"] = settlement_ksh_code
        allocated_group["generated_settlement_size_total"] = total_count_to_allocate
        allocated_group["used_national_fallback_probability"] = used_fallback

        allocated_group = allocated_group[
            allocated_group["allocated_household_count"] > 0
        ].copy()

        allocated_tables.append(allocated_group)

    if not allocated_tables:
        return pd.DataFrame()

    allocated_settlement_targets = pd.concat(allocated_tables, ignore_index=True)

    output_columns = [
        "settlement_key",
        "settlement_name",
        "settlement_ksh_code",
        "county_name",
        "settlement_type",
        "household_size_label",
        "household_age_composition",
        "household_employment_composition",
        "allocated_household_count",
        "generated_settlement_size_total",
        "composition_probability",
        "expected_household_count",
        "used_national_fallback_probability",
    ]

    allocated_settlement_targets = allocated_settlement_targets[output_columns].copy()

    allocated_settlement_targets = allocated_settlement_targets.sort_values(
        [
            "county_name",
            "settlement_type",
            "settlement_name",
            "household_size_label",
            "allocated_household_count",
        ],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    return allocated_settlement_targets


# ============================================================
# 7. VALIDÁCIÓK
# ============================================================

def validate_area_allocation(allocated_area_targets: pd.DataFrame) -> pd.DataFrame:
    """
    Ellenőrzi, hogy area szinten az allokált darabszámok visszaadják-e
    a generated targeteket.
    """
    validation = (
        allocated_area_targets
        .groupby(
            ["county_name", "settlement_type", "household_size_label"],
            dropna=False,
        )
        .agg(
            allocated_total=("allocated_household_count", "sum"),
            generated_group_total=("generated_group_total", "max"),
        )
        .reset_index()
    )

    validation["difference_allocated_vs_generated"] = (
        validation["allocated_total"]
        - validation["generated_group_total"]
    )

    return validation


def validate_settlement_allocation(allocated_settlement_targets: pd.DataFrame) -> pd.DataFrame:
    """
    Ellenőrzi, hogy település + háztartásméret szinten az allokált darabszámok
    visszaadják-e az eredeti generated targeteket.
    """
    validation = (
        allocated_settlement_targets
        .groupby(
            ["settlement_key", "settlement_name", "household_size_label"],
            dropna=False,
        )
        .agg(
            allocated_total=("allocated_household_count", "sum"),
            generated_settlement_size_total=("generated_settlement_size_total", "max"),
        )
        .reset_index()
    )

    validation["difference_allocated_vs_generated"] = (
        validation["allocated_total"]
        - validation["generated_settlement_size_total"]
    )

    return validation


def create_overall_composition_processing_summary(
    long_table: pd.DataFrame,
    generated_targets_long: pd.DataFrame,
    allocated_settlement_targets: pd.DataFrame,
    settlement_allocation_validation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rövid országos összefoglaló a feldolgozásról.
    """
    summary_rows = []

    summary_rows.append({
        "metric": "Hier homesize long table total",
        "value": long_table["value"].sum(),
        "note": "A hier homesize_agegroup_employment tábla nyers, releváns kategóriáinak összege.",
    })

    summary_rows.append({
        "metric": "Generated household targets total",
        "value": generated_targets_long["generated_household_count"].sum(),
        "note": "A korábban generált flat household target összes háztartása.",
    })

    summary_rows.append({
        "metric": "Allocated settlement composition total",
        "value": allocated_settlement_targets["allocated_household_count"].sum(),
        "note": "A településszintre szétosztott household-kompozíciós célok összege.",
    })

    summary_rows.append({
        "metric": "Settlement allocation max absolute difference",
        "value": settlement_allocation_validation[
            "difference_allocated_vs_generated"
        ].abs().max(),
        "note": "Ennek 0-nak kell lennie, ha minden település + méret pontosan visszaadódik.",
    })

    summary_rows.append({
        "metric": "Settlement allocation nonzero difference rows",
        "value": (
            settlement_allocation_validation["difference_allocated_vs_generated"] != 0
        ).sum(),
        "note": "Ennek 0-nak kell lennie.",
    })

    summary = pd.DataFrame(summary_rows)

    return summary


# ============================================================
# 8. FŐ FUTTATÁSI FÜGGVÉNY
# ============================================================

def run_household_composition_target_preparation() -> None:
    """
    Fő folyamat:

    1. Beolvassa a hier_homesize_agegroup_employment táblát.
    2. Long formátumra alakítja.
    3. Kiszámolja a kompozíciós arányokat.
    4. Beolvassa a korábbi household generation targeteket.
    5. Összeveti a flat/generated darabszámokat a hier totalokkal.
    6. Kompozíciós célokat allokál area és settlement szinten.
    7. Validációs fájlokat ment.
    """
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Hier homesize_agegroup_employment tábla beolvasása...")
    raw_homesize_table = read_hier_homesize_table_without_header(HIER_HOMESIZE_FILE)

    print("Hier homesize tábla hosszú formátumra alakítása...")
    homesize_long_table = convert_hier_homesize_table_to_long_format(
        raw_table=raw_homesize_table
    )

    print("Hier totalok számítása...")
    hier_totals = calculate_hier_composition_totals(homesize_long_table)

    print("Hier kompozíciós arányok számítása...")
    hier_probabilities = calculate_hier_composition_probabilities(homesize_long_table)

    print("Országos fallback arányok számítása...")
    national_fallback_probabilities = calculate_national_fallback_probabilities(
        hier_probabilities
    )

    print("Korábbi household generation targetek beolvasása...")
    household_generation_targets = load_household_generation_targets(
        HOUSEHOLD_GENERATION_TARGETS_CSV
    )

    print("Generated household targetek hosszú formátumra alakítása...")
    generated_targets_long = convert_household_generation_targets_to_long_format(
        household_generation_targets
    )

    print("Generated household targetek aggregálása area szinten...")
    generated_area_targets = aggregate_generated_household_targets_by_area(
        generated_targets_long
    )

    print("Generated vs hier totalok összevetése...")
    generated_vs_hier_comparison = compare_generated_targets_with_hier_totals(
        generated_area_targets=generated_area_targets,
        hier_totals=hier_totals,
    )

    print("Area szintű kompozíciós targetek allokálása...")
    allocated_area_targets = allocate_area_composition_targets(
        generated_area_targets=generated_area_targets,
        probabilities=hier_probabilities,
        national_fallback_probabilities=national_fallback_probabilities,
    )

    print("Településszintű kompozíciós targetek allokálása...")
    allocated_settlement_targets = allocate_settlement_composition_targets(
        generated_targets_long=generated_targets_long,
        probabilities=hier_probabilities,
        national_fallback_probabilities=national_fallback_probabilities,
    )

    print("Allokációk validálása...")
    area_allocation_validation = validate_area_allocation(allocated_area_targets)
    settlement_allocation_validation = validate_settlement_allocation(
        allocated_settlement_targets
    )

    print("Összefoglaló készítése...")
    overall_summary = create_overall_composition_processing_summary(
        long_table=homesize_long_table,
        generated_targets_long=generated_targets_long,
        allocated_settlement_targets=allocated_settlement_targets,
        settlement_allocation_validation=settlement_allocation_validation,
    )

    # --------------------------------------------------------
    # Outputok mentése
    # --------------------------------------------------------

    homesize_long_table.to_csv(
        OUTPUT_FOLDER / "06_homesize_agegroup_employment_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hier_totals.to_csv(
        OUTPUT_FOLDER / "07_homesize_hier_totals_by_area_size.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hier_probabilities.to_csv(
        OUTPUT_FOLDER / "08_homesize_composition_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    generated_area_targets.to_csv(
        OUTPUT_FOLDER / "09_generated_household_targets_by_area_size.csv",
        index=False,
        encoding="utf-8-sig",
    )

    generated_vs_hier_comparison.to_csv(
        OUTPUT_FOLDER / "10_compare_generated_vs_hier_homesize_totals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    allocated_area_targets.to_csv(
        OUTPUT_FOLDER / "11_allocated_area_composition_targets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    allocated_settlement_targets.to_csv(
        OUTPUT_FOLDER / "12_allocated_settlement_composition_targets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    area_allocation_validation.to_csv(
        OUTPUT_FOLDER / "13_area_composition_allocation_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    settlement_allocation_validation.to_csv(
        OUTPUT_FOLDER / "14_settlement_composition_allocation_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    national_fallback_probabilities.to_csv(
        OUTPUT_FOLDER / "15_national_fallback_composition_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_summary.to_csv(
        OUTPUT_FOLDER / "16_household_composition_processing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


# ============================================================
# 9. PROGRAM INDÍTÁSA
# Ezt ne módosítsd.
# ============================================================

if __name__ == "__main__":
    run_household_composition_target_preparation()