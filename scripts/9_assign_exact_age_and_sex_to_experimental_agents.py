from pathlib import Path
import csv
import math
import random
import re
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

AGE_DISTRIBUTION_XLSX = Path(
    "../data/raw_hier_tables/age_distribution.xlsx"
)

EXPERIMENTAL_AGENTS_CSV = Path(
    "../outputs/household_generation/18_generated_agents_from_households_experimental.csv"
)

FLAT_LONG_TABLE_CSV = Path(
    "../outputs/flat_validation/05_flat_combined_long_format_table.csv"
)

OUTPUT_FOLDER = Path("../outputs/household_generation")

AGENTS_WITH_EXACT_AGE_SEX_CSV = OUTPUT_FOLDER / "29_generated_agents_with_exact_age_sex_experimental.csv"

AGE_SEX_DISTRIBUTION_LONG_CSV = OUTPUT_FOLDER / "30_age_sex_distribution_long.csv"

AGE_SEX_PROBABILITIES_CSV = OUTPUT_FOLDER / "31_age_sex_probabilities_by_area_broad_age_group.csv"

AGENT_COUNTS_BY_AREA_BROAD_AGE_CSV = OUTPUT_FOLDER / "32_agent_counts_by_area_broad_age_group.csv"

AGE_SEX_ASSIGNMENT_VALIDATION_CSV = OUTPUT_FOLDER / "33_age_sex_assignment_validation_by_area_age_sex.csv"

FLAT_SETTLEMENT_POPULATION_COMPARISON_CSV = OUTPUT_FOLDER / "34_flat_settlement_population_vs_generated_agents.csv"

FLAT_SETTLEMENT_GENDER_COMPARISON_CSV = OUTPUT_FOLDER / "35_flat_settlement_gender_vs_generated_agents.csv"

FLAT_SETTLEMENT_AGE_GROUP_COMPARISON_CSV = OUTPUT_FOLDER / "36_flat_settlement_age_group_vs_generated_agents.csv"

AGE_SEX_ASSIGNMENT_SUMMARY_CSV = OUTPUT_FOLDER / "37_age_sex_assignment_summary.csv"


AGE_DISTRIBUTION_SHEET_NAME = "Adattábla"

RANDOM_SEED = 42

# Teszthez állíthatod pl. 100_000-re.
# Teljes futtatáshoz legyen None.
MAX_AGENTS_TO_PROCESS = None


# ============================================================
# 2. ÁLTALÁNOS SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def convert_count_to_integer(raw_value) -> int:
    if raw_value is None:
        return 0

    if pd.isna(raw_value):
        return 0

    text_value = str(raw_value).strip()

    if text_value == "":
        return 0

    return int(round(float(text_value)))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def normalize_county_name(raw_county_name: str) -> str:
    """
    Vármegye-nevek egységesítése.

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
        "Győr-Moson-Sopron": "Győr-Moson-Sopron",
        "fováros": "Budapest",
        "Fováros": "Budapest",
        "főváros": "Budapest",
        "Főváros": "Budapest",
        "Budapest": "Budapest",
    }

    return manual_county_corrections.get(county_name, county_name)


def normalize_settlement_type(raw_settlement_type: str) -> str:
    if pd.isna(raw_settlement_type):
        return ""

    settlement_type = str(raw_settlement_type).strip()

    manual_type_corrections = {
        "Fováros": "Főváros",
        "fováros": "Főváros",
        "főváros": "Főváros",
        "Főváros": "Főváros",
    }

    return manual_type_corrections.get(settlement_type, settlement_type)


def normalize_sex_label(raw_sex_label: str) -> str:
    """
    KSH nem címke → egyszerű angol kód.
    """
    if pd.isna(raw_sex_label):
        return ""

    sex_label = str(raw_sex_label).strip().lower()

    if sex_label == "férfi":
        return "male"

    if sex_label == "nő":
        return "female"

    return sex_label


def map_exact_age_to_broad_age_group(exact_age: int) -> str:
    """
    A korábbi agentgenerálás broad_age_group logikájához illesztjük.
    """
    if exact_age < 30:
        return "under_30"

    if exact_age <= 64:
        return "age_30_64"

    return "age_65_plus"


def parse_exact_age(raw_age_label: str) -> int | None:
    """
    Példák:
    '0 éves' -> 0
    '25 éves' -> 25
    '99 éves' -> 99
    """
    if pd.isna(raw_age_label):
        return None

    text = str(raw_age_label).strip().lower()

    match = re.search(r"(\d+)", text)

    if match is None:
        return None

    return int(match.group(1))


def normalize_settlement_name(raw_name: str) -> str:
    """
    Ugyanaz a logika, mint a korábbi flat feldolgozásnál.
    Itt csak a flat long összevetéshez használjuk.
    """
    if pd.isna(raw_name):
        return ""

    normalized = str(raw_name).strip().lower()

    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[\.,;:()\[\]/\\]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


# ============================================================
# 3. AGE DISTRIBUTION TÁBLA BEOLVASÁSA
# ============================================================

def read_age_distribution_raw_table(age_distribution_xlsx: Path) -> pd.DataFrame:
    raw_table = pd.read_excel(
        age_distribution_xlsx,
        sheet_name=AGE_DISTRIBUTION_SHEET_NAME,
        header=None,
        engine="openpyxl",
    )

    return raw_table


def detect_first_numeric_value_column(raw_table: pd.DataFrame) -> int:
    """
    Megkeresi, hol kezdődnek a számos értékoszlopok.
    """
    numeric_table = raw_table.apply(pd.to_numeric, errors="coerce")
    numeric_counts_by_column = numeric_table.notna().sum(axis=0)

    for column_index, numeric_count in numeric_counts_by_column.items():
        if numeric_count > 0:
            return column_index

    raise ValueError("Nem találtam számos értékoszlopot az age distribution táblában.")


def detect_geography_header_start_row(raw_table: pd.DataFrame) -> int:
    """
    Megkeresi, melyik sorban kezdődik a földrajzi fejléc.

    Az age_distribution táblában ez jellemzően az a sor,
    ahol Budapest / Pest vármegye / Baranya vármegye stb. szerepel.
    """
    for row_index in range(min(10, len(raw_table))):
        row_values = []

        for cell_value in raw_table.iloc[row_index].tolist():
            if pd.isna(cell_value):
                continue

            row_values.append(str(cell_value).strip().lower())

        row_text = " ".join(row_values)

        if "budapest" in row_text and "vármegye" in row_text:
            return row_index

    raise ValueError(
        "Nem találtam meg a földrajzi fejléc kezdősorát az age_distribution táblában."
    )


def detect_first_age_data_row(raw_table: pd.DataFrame) -> int:
    """
    Megkeresi az első tényleges adatsort.

    Várhatóan ott kezdődik, ahol:
    - az első oszlopban Férfi vagy Nő van
    - a második oszlopban valamilyen '0 éves' jellegű korcímke van
    """
    for row_index in range(len(raw_table)):
        first_cell = str(raw_table.iloc[row_index, 0]).strip().lower()
        second_cell = str(raw_table.iloc[row_index, 1]).strip().lower()

        if first_cell in ["férfi", "nő"] and "éves" in second_cell:
            return row_index

    raise ValueError("Nem találtam meg az első életkoros adatsort az age_distribution táblában.")


def build_geography_header_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    """
    Földrajzi fejléc építése az age_distribution táblából.

    A fájl szerkezete:
    - cím / év sorok felül
    - egyik sor: vármegye / Budapest
    - következő sor: településtípus
    """
    first_value_column_index = detect_first_numeric_value_column(raw_table)
    geography_header_start_row = detect_geography_header_start_row(raw_table)

    county_row_index = geography_header_start_row
    settlement_type_row_index = geography_header_start_row + 1

    geography_header_rows = raw_table.iloc[
        [county_row_index, settlement_type_row_index],
        first_value_column_index:
    ].copy()

    county_names = geography_header_rows.iloc[0].ffill()
    settlement_types = geography_header_rows.iloc[1].fillna("Nincs megadva")

    geography_header_table = pd.DataFrame({
        "value_column_index": geography_header_rows.columns,
        "county_name": county_names.values,
        "settlement_type": settlement_types.values,
    })

    geography_header_table["county_name"] = geography_header_table["county_name"].apply(
        normalize_county_name
    )

    geography_header_table["settlement_type"] = geography_header_table["settlement_type"].apply(
        normalize_settlement_type
    )

    print("Age distribution geography header ellenőrzés:")
    print(geography_header_table.head(10))
    return geography_header_table


def convert_age_distribution_to_long_format(raw_table: pd.DataFrame) -> pd.DataFrame:
    """
    age_distribution.xlsx → long format.

    Output:
    county_name
    settlement_type
    sex
    exact_age
    broad_age_group
    value
    """
    first_value_column_index = detect_first_numeric_value_column(raw_table)

    first_data_row_index = detect_first_age_data_row(raw_table)

    category_part = raw_table.iloc[first_data_row_index:, :first_value_column_index].copy()
    value_part = raw_table.iloc[first_data_row_index:, first_value_column_index:].copy()

    # Az első kategória: nem. Excelben gyakran merge-elt, ezért ffill.
    category_part = category_part.ffill()

    # A várható első két kategóriaoszlop:
    # 0: nem
    # 1: életkor
    category_part = category_part.rename(
        columns={
            0: "sex_label",
            1: "age_label",
        }
    )

    value_part_numeric = value_part.apply(pd.to_numeric, errors="coerce").fillna(0)

    geography_header_table = build_geography_header_table(raw_table)

    long_rows = []

    for local_row_position, original_row_index in enumerate(value_part_numeric.index):
        category_row = category_part.iloc[local_row_position]

        sex = normalize_sex_label(category_row["sex_label"])
        exact_age = parse_exact_age(category_row["age_label"])

        if sex not in ["male", "female"]:
            continue

        if exact_age is None:
            continue

        broad_age_group = map_exact_age_to_broad_age_group(exact_age)

        for value_column_index in value_part_numeric.columns:
            raw_value = value_part_numeric.loc[original_row_index, value_column_index]
            value = convert_count_to_integer(raw_value)

            geography_row = geography_header_table[
                geography_header_table["value_column_index"] == value_column_index
            ].iloc[0]

            long_rows.append({
                "excel_row_number": original_row_index + 1,
                "excel_value_column_number": value_column_index + 1,
                "county_name": geography_row["county_name"],
                "settlement_type": geography_row["settlement_type"],
                "sex": sex,
                "exact_age": exact_age,
                "broad_age_group": broad_age_group,
                "value": value,
            })

    long_table = pd.DataFrame(long_rows)

    return long_table


def calculate_age_sex_probabilities(
    age_distribution_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vármegye × településtípus × broad_age_group szinten kiszámolja:
    sex × exact_age valószínűség.
    """
    counts = (
        age_distribution_long
        .groupby(
            [
                "county_name",
                "settlement_type",
                "broad_age_group",
                "sex",
                "exact_age",
            ],
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "age_sex_count"})
    )

    group_totals = (
        counts
        .groupby(
            ["county_name", "settlement_type", "broad_age_group"],
            dropna=False,
        )["age_sex_count"]
        .sum()
        .reset_index()
        .rename(columns={"age_sex_count": "age_sex_group_total"})
    )

    probabilities = counts.merge(
        group_totals,
        on=["county_name", "settlement_type", "broad_age_group"],
        how="left",
    )

    probabilities["age_sex_probability"] = probabilities.apply(
        lambda row: safe_divide(row["age_sex_count"], row["age_sex_group_total"]),
        axis=1,
    )

    probabilities = probabilities.sort_values(
        [
            "county_name",
            "settlement_type",
            "broad_age_group",
            "exact_age",
            "sex",
        ]
    ).reset_index(drop=True)

    return probabilities


# ============================================================
# 4. AGENTEK ELSŐ PASS: CSOPORTDARABSZÁMOK
# ============================================================

def count_agents_by_area_and_broad_age_group(
    experimental_agents_csv: Path,
) -> pd.DataFrame:
    """
    Első pass a nagy agent fájlon.

    Megszámolja, hogy hány agent van:
    county_name × settlement_type × broad_age_group szinten.
    """
    counts = defaultdict(int)

    total_agents_seen = 0

    with experimental_agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents_seen >= MAX_AGENTS_TO_PROCESS:
                break

            county_name = normalize_county_name(row["county_name"])
            settlement_type = normalize_settlement_type(row["settlement_type"])
            broad_age_group = row["broad_age_group"]

            key = (county_name, settlement_type, broad_age_group)
            counts[key] += 1

            total_agents_seen += 1

            if total_agents_seen % 1_000_000 == 0:
                print(f"Első pass agentek: {total_agents_seen:,}")

    rows = []

    for key, count in counts.items():
        county_name, settlement_type, broad_age_group = key

        rows.append({
            "county_name": county_name,
            "settlement_type": settlement_type,
            "broad_age_group": broad_age_group,
            "generated_agent_count": count,
        })

    count_table = pd.DataFrame(rows)

    count_table = count_table.sort_values(
        ["county_name", "settlement_type", "broad_age_group"]
    ).reset_index(drop=True)

    return count_table


# ============================================================
# 5. INTEGER ALLOKÁCIÓ EXACT AGE + SEX SZINTRE
# ============================================================
def encode_age_sex_for_pool(sex: str, exact_age: int) -> int:
    """
    Sex + exact_age kompakt kódolása a poolhoz.

    male   -> even code
    female -> odd code

    Példa:
    male, 40   -> 80
    female, 40 -> 81
    """
    sex_normalized = normalize_sex_label(sex)

    if sex_normalized == "male":
        sex_code = 0
    elif sex_normalized == "female":
        sex_code = 1
    else:
        raise ValueError(f"Ismeretlen sex érték a pool kódolásnál: {sex}")

    return int(exact_age) * 2 + sex_code


def decode_age_sex_from_pool_code(pool_code: int) -> dict:
    """
    Kompakt pool kód visszaalakítása sex + exact_age értékké.
    """
    exact_age = int(pool_code) // 2
    sex_code = int(pool_code) % 2

    if sex_code == 0:
        sex = "male"
    else:
        sex = "female"

    return {
        "sex": sex,
        "exact_age": exact_age,
    }

def allocate_integer_counts_by_largest_remainder(
    probabilities_for_group: pd.DataFrame,
    total_count_to_allocate: int,
) -> pd.DataFrame:
    allocation = probabilities_for_group.copy()

    if total_count_to_allocate <= 0 or allocation.empty:
        allocation["allocated_agent_count"] = 0
        allocation["expected_agent_count"] = 0.0
        allocation["allocation_remainder"] = 0.0
        return allocation

    allocation["expected_agent_count"] = (
        allocation["age_sex_probability"] * total_count_to_allocate
    )

    allocation["allocated_agent_count"] = allocation["expected_agent_count"].apply(
        math.floor
    )

    already_allocated = int(allocation["allocated_agent_count"].sum())
    remaining_to_allocate = total_count_to_allocate - already_allocated

    allocation["allocation_remainder"] = (
        allocation["expected_agent_count"]
        - allocation["allocated_agent_count"]
    )

    allocation = allocation.sort_values(
        ["allocation_remainder", "expected_agent_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if remaining_to_allocate > 0:
        allocation.loc[
            allocation.index < remaining_to_allocate,
            "allocated_agent_count"
        ] += 1

    allocation = allocation.sort_values(
        ["county_name", "settlement_type", "broad_age_group", "exact_age", "sex"]
    ).reset_index(drop=True)

    return allocation


def create_age_sex_allocation_table(
    agent_counts_by_group: pd.DataFrame,
    age_sex_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agent darabszámokat osztja sex × exact_age kategóriákra,
    adott county × settlement_type × broad_age_group csoporton belül.
    """
    allocated_tables = []

    for _, count_row in agent_counts_by_group.iterrows():
        county_name = count_row["county_name"]
        settlement_type = count_row["settlement_type"]
        broad_age_group = count_row["broad_age_group"]
        generated_agent_count = convert_count_to_integer(count_row["generated_agent_count"])

        probabilities_for_group = age_sex_probabilities[
            (age_sex_probabilities["county_name"] == county_name)
            & (age_sex_probabilities["settlement_type"] == settlement_type)
            & (age_sex_probabilities["broad_age_group"] == broad_age_group)
        ].copy()

        if probabilities_for_group.empty:
            raise ValueError(
                f"Nincs age/sex probability ehhez a csoporthoz: "
                f"{county_name}, {settlement_type}, {broad_age_group}"
            )

        allocated_group = allocate_integer_counts_by_largest_remainder(
            probabilities_for_group=probabilities_for_group,
            total_count_to_allocate=generated_agent_count,
        )

        allocated_group["generated_agent_group_total"] = generated_agent_count

        allocated_tables.append(allocated_group)

    allocation_table = pd.concat(allocated_tables, ignore_index=True)

    allocation_table = allocation_table[
        allocation_table["allocated_agent_count"] > 0
    ].copy()

    allocation_table = allocation_table.sort_values(
        ["county_name", "settlement_type", "broad_age_group", "exact_age", "sex"]
    ).reset_index(drop=True)

    return allocation_table


def build_remaining_age_sex_pool(
    age_sex_allocation_table: pd.DataFrame,
) -> dict:
    """
    A második passhoz elkészíti az age/sex poolt.

    Fontos javítás:
    Nem kategóriánként tartjuk a remaining_countot, mert az sorban
    elfogyasztaná az age/sex kategóriákat. Ehelyett minden allokált
    agenthez létrehozunk egy kompakt age/sex kódot, majd a csoporton
    belül megkeverjük a listát.

    Így ugyanúgy pontosan tartjuk:
    county × settlement_type × broad_age_group × sex × exact_age darabszámokat,

    de nem település-sorrend szerint csoportosulnak az életkorok.
    """
    pool = defaultdict(list)

    for _, row in age_sex_allocation_table.iterrows():
        key = (
            row["county_name"],
            row["settlement_type"],
            row["broad_age_group"],
        )

        allocated_count = convert_count_to_integer(row["allocated_agent_count"])

        pool_code = encode_age_sex_for_pool(
            sex=row["sex"],
            exact_age=convert_count_to_integer(row["exact_age"]),
        )

        pool[key].extend([pool_code] * allocated_count)

    random_generator = random.Random(RANDOM_SEED)

    print("Age/sex pool keverése csoportonként...")

    for key in pool:
        random_generator.shuffle(pool[key])

    return pool


def pop_next_age_sex_from_pool(
    pool: dict,
    county_name: str,
    settlement_type: str,
    broad_age_group: str,
) -> dict:
    key = (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        broad_age_group,
    )

    if key not in pool:
        raise ValueError(f"Nincs age/sex pool ehhez a csoporthoz: {key}")

    if len(pool[key]) == 0:
        raise ValueError(f"Elfogyott az age/sex pool ehhez a csoporthoz: {key}")

    pool_code = pool[key].pop()

    return decode_age_sex_from_pool_code(pool_code)


# ============================================================
# 6. AGENTEK MÁSODIK PASS: EXACT AGE + SEX HOZZÁÍRÁSA
# ============================================================

def assign_exact_age_and_sex_to_agents(
    experimental_agents_csv: Path,
    output_agents_csv: Path,
    age_sex_pool: dict,
) -> dict:
    """
    Második pass a nagy agent fájlon.

    Minden agenthez hozzárendel:
    - sex
    - exact_age
    """
    generated_counts_by_area_age_sex = defaultdict(int)
    generated_counts_by_settlement = defaultdict(int)
    generated_counts_by_settlement_sex = defaultdict(int)
    generated_counts_by_settlement_age_group = defaultdict(int)

    total_agents_written = 0

    with experimental_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)

        output_columns = list(reader.fieldnames) + [
            "sex",
            "exact_age",
        ]

        writer = csv.DictWriter(output_file, fieldnames=output_columns)
        writer.writeheader()

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents_written >= MAX_AGENTS_TO_PROCESS:
                break

            county_name = normalize_county_name(row["county_name"])
            settlement_type = normalize_settlement_type(row["settlement_type"])
            broad_age_group = row["broad_age_group"]

            age_sex = pop_next_age_sex_from_pool(
                pool=age_sex_pool,
                county_name=county_name,
                settlement_type=settlement_type,
                broad_age_group=broad_age_group,
            )

            row["county_name"] = county_name
            row["settlement_type"] = settlement_type
            row["sex"] = age_sex["sex"]
            row["exact_age"] = age_sex["exact_age"]

            writer.writerow(row)

            area_age_sex_key = (
                county_name,
                settlement_type,
                broad_age_group,
                age_sex["sex"],
                age_sex["exact_age"],
            )

            generated_counts_by_area_age_sex[area_age_sex_key] += 1

            settlement_key = row["settlement_key"]
            settlement_name = row["settlement_name"]

            generated_counts_by_settlement[
                (settlement_key, settlement_name)
            ] += 1

            generated_counts_by_settlement_sex[
                (settlement_key, settlement_name, age_sex["sex"])
            ] += 1

            flat_age_group_label = map_exact_age_to_flat_10_year_age_group(
                age_sex["exact_age"]
            )

            generated_counts_by_settlement_age_group[
                (settlement_key, settlement_name, flat_age_group_label)
            ] += 1

            total_agents_written += 1

            if total_agents_written % 1_000_000 == 0:
                print(f"Második pass agentek: {total_agents_written:,}")

    return {
        "total_agents_written": total_agents_written,
        "generated_counts_by_area_age_sex": generated_counts_by_area_age_sex,
        "generated_counts_by_settlement": generated_counts_by_settlement,
        "generated_counts_by_settlement_sex": generated_counts_by_settlement_sex,
        "generated_counts_by_settlement_age_group": generated_counts_by_settlement_age_group,
    }


# ============================================================
# 7. FLAT ÖSSZEVETÉSEK
# ============================================================

def load_flat_long_table(flat_long_table_csv: Path) -> pd.DataFrame:
    flat_long_table = pd.read_csv(flat_long_table_csv)

    flat_long_table["settlement_key"] = flat_long_table["settlement"].apply(
        normalize_settlement_name
    )

    flat_long_table["value"] = pd.to_numeric(
        flat_long_table["value"],
        errors="coerce",
    ).fillna(0)

    return flat_long_table


def extract_flat_settlement_population_targets(
    flat_long_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flat településszintű teljes népesség:
    Férfi + Nő.
    """
    rows = flat_long_table[
        flat_long_table["source_file"].str.contains("nepesseg", case=False, regex=False)
        & flat_long_table["category"].isin(["Férfi", "Nő"])
    ].copy()

    targets = (
        rows
        .groupby(["settlement_key", "settlement"], dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "settlement": "settlement_name_flat",
                "value": "flat_population_total",
            }
        )
    )

    targets["flat_population_total"] = targets["flat_population_total"].apply(
        convert_count_to_integer
    )

    return targets


def extract_flat_settlement_gender_targets(
    flat_long_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flat településszintű nem szerinti target:
    Férfi / Nő.
    """
    rows = flat_long_table[
        flat_long_table["source_file"].str.contains("nepesseg", case=False, regex=False)
        & flat_long_table["category"].isin(["Férfi", "Nő"])
    ].copy()

    rows["sex"] = rows["category"].apply(normalize_sex_label)

    targets = (
        rows
        .groupby(["settlement_key", "settlement", "sex"], dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "settlement": "settlement_name_flat",
                "value": "flat_gender_count",
            }
        )
    )

    targets["flat_gender_count"] = targets["flat_gender_count"].apply(
        convert_count_to_integer
    )

    return targets


def parse_flat_age_group_category(category: str) -> tuple[int, int, str] | None:
    """
    Flat népességi korcsoport kategóriák felismerése.

    Példák:
    '0–9 éves' -> (0, 9, '0_9')
    '10–19 éves' -> (10, 19, '10_19')
    '90 éves és idősebb' -> (90, 200, '90_plus')

    Nem fogja felismerni:
    '15 évesnél fiatalabb személy'
    mert az más logikai kategória.
    """
    text = str(category).strip().lower()

    range_match = re.match(r"^(\d+)\s*[–-]\s*(\d+)\s*éves", text)

    if range_match:
        min_age = int(range_match.group(1))
        max_age = int(range_match.group(2))
        label = f"{min_age}_{max_age}"
        return min_age, max_age, label

    plus_match = re.match(r"^(\d+)\s*éves és idősebb", text)

    if plus_match:
        min_age = int(plus_match.group(1))
        max_age = 200
        label = f"{min_age}_plus"
        return min_age, max_age, label

    return None


def map_exact_age_to_flat_10_year_age_group(exact_age: int) -> str:
    """
    Generated agent exact_age → flat korcsoport label.

    Ez a 10 éves KSH kategóriákhoz igazodik.
    Ha a flat fájlban eltérő felső kategória van, a comparison fájl majd mutatja.
    """
    age = int(exact_age)

    if age < 10:
        return "0_9"

    if age < 20:
        return "10_19"

    if age < 30:
        return "20_29"

    if age < 40:
        return "30_39"

    if age < 50:
        return "40_49"

    if age < 60:
        return "50_59"

    if age < 70:
        return "60_69"

    if age < 80:
        return "70_79"

    if age < 90:
        return "80_89"

    return "90_plus"


def extract_flat_settlement_age_group_targets(
    flat_long_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flat településszintű korcsoport targetek.

    Csak azokat a kategóriákat veszi, amelyek tiszta korintervallumnak tűnnek.
    """
    candidate_rows = flat_long_table[
        flat_long_table["source_file"].str.contains("nepesseg", case=False, regex=False)
    ].copy()

    parsed_rows = []

    for _, row in candidate_rows.iterrows():
        parsed = parse_flat_age_group_category(row["category"])

        if parsed is None:
            continue

        min_age, max_age, age_group_label = parsed

        parsed_rows.append({
            "settlement_key": row["settlement_key"],
            "settlement_name_flat": row["settlement"],
            "flat_age_group_label": age_group_label,
            "flat_age_group_min_age": min_age,
            "flat_age_group_max_age": max_age,
            "flat_age_group_count": convert_count_to_integer(row["value"]),
            "source_category": row["category"],
        })

    targets = pd.DataFrame(parsed_rows)

    return targets


def convert_generated_settlement_counter_to_dataframe(
    counter: dict,
    value_column_name: str,
    key_column_names: list[str],
) -> pd.DataFrame:
    rows = []

    for key_tuple, count in counter.items():
        row = {}

        for index, column_name in enumerate(key_column_names):
            row[column_name] = key_tuple[index]

        row[value_column_name] = count
        rows.append(row)

    return pd.DataFrame(rows)


def create_flat_settlement_population_comparison(
    flat_population_targets: pd.DataFrame,
    generated_counts_by_settlement: dict,
) -> pd.DataFrame:
    generated = convert_generated_settlement_counter_to_dataframe(
        counter=generated_counts_by_settlement,
        value_column_name="generated_agent_count",
        key_column_names=["settlement_key", "settlement_name_generated"],
    )

    comparison = flat_population_targets.merge(
        generated,
        on="settlement_key",
        how="outer",
    )

    comparison["flat_population_total"] = pd.to_numeric(
        comparison["flat_population_total"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["generated_agent_count"] = pd.to_numeric(
        comparison["generated_agent_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["difference_generated_minus_flat"] = (
        comparison["generated_agent_count"]
        - comparison["flat_population_total"]
    )

    comparison["absolute_difference"] = comparison[
        "difference_generated_minus_flat"
    ].abs()

    comparison["relative_abs_error"] = comparison.apply(
        lambda row: safe_divide(
            abs(row["difference_generated_minus_flat"]),
            row["flat_population_total"],
        ),
        axis=1,
    )

    comparison["ratio_generated_to_flat"] = comparison.apply(
        lambda row: safe_divide(
            row["generated_agent_count"],
            row["flat_population_total"],
        ),
        axis=1,
    )

    comparison = comparison.sort_values(
        ["absolute_difference", "flat_population_total"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return comparison


def create_flat_settlement_gender_comparison(
    flat_gender_targets: pd.DataFrame,
    generated_counts_by_settlement_sex: dict,
) -> pd.DataFrame:
    generated = convert_generated_settlement_counter_to_dataframe(
        counter=generated_counts_by_settlement_sex,
        value_column_name="generated_sex_count",
        key_column_names=["settlement_key", "settlement_name_generated", "sex"],
    )

    comparison = flat_gender_targets.merge(
        generated,
        on=["settlement_key", "sex"],
        how="outer",
    )

    comparison["flat_gender_count"] = pd.to_numeric(
        comparison["flat_gender_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["generated_sex_count"] = pd.to_numeric(
        comparison["generated_sex_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["difference_generated_minus_flat"] = (
        comparison["generated_sex_count"]
        - comparison["flat_gender_count"]
    )

    comparison["absolute_difference"] = comparison[
        "difference_generated_minus_flat"
    ].abs()

    comparison["relative_abs_error"] = comparison.apply(
        lambda row: safe_divide(
            abs(row["difference_generated_minus_flat"]),
            row["flat_gender_count"],
        ),
        axis=1,
    )

    comparison["ratio_generated_to_flat"] = comparison.apply(
        lambda row: safe_divide(
            row["generated_sex_count"],
            row["flat_gender_count"],
        ),
        axis=1,
    )

    comparison = comparison.sort_values(
        ["absolute_difference", "flat_gender_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return comparison


def create_flat_settlement_age_group_comparison(
    flat_age_group_targets: pd.DataFrame,
    generated_counts_by_settlement_age_group: dict,
) -> pd.DataFrame:
    generated = convert_generated_settlement_counter_to_dataframe(
        counter=generated_counts_by_settlement_age_group,
        value_column_name="generated_age_group_count",
        key_column_names=[
            "settlement_key",
            "settlement_name_generated",
            "flat_age_group_label",
        ],
    )

    comparison = flat_age_group_targets.merge(
        generated,
        on=["settlement_key", "flat_age_group_label"],
        how="outer",
    )

    comparison["flat_age_group_count"] = pd.to_numeric(
        comparison["flat_age_group_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["generated_age_group_count"] = pd.to_numeric(
        comparison["generated_age_group_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["difference_generated_minus_flat"] = (
        comparison["generated_age_group_count"]
        - comparison["flat_age_group_count"]
    )

    comparison["absolute_difference"] = comparison[
        "difference_generated_minus_flat"
    ].abs()

    comparison["relative_abs_error"] = comparison.apply(
        lambda row: safe_divide(
            abs(row["difference_generated_minus_flat"]),
            row["flat_age_group_count"],
        ),
        axis=1,
    )

    comparison["ratio_generated_to_flat"] = comparison.apply(
        lambda row: safe_divide(
            row["generated_age_group_count"],
            row["flat_age_group_count"],
        ),
        axis=1,
    )

    comparison = comparison.sort_values(
        ["absolute_difference", "flat_age_group_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return comparison


# ============================================================
# 8. AGE DISTRIBUTION VALIDÁCIÓ
# ============================================================

def create_age_sex_assignment_validation(
    age_sex_allocation_table: pd.DataFrame,
    generated_counts_by_area_age_sex: dict,
) -> pd.DataFrame:
    generated = convert_generated_settlement_counter_to_dataframe(
        counter=generated_counts_by_area_age_sex,
        value_column_name="generated_agent_count",
        key_column_names=[
            "county_name",
            "settlement_type",
            "broad_age_group",
            "sex",
            "exact_age",
        ],
    )

    validation = age_sex_allocation_table.merge(
        generated,
        on=[
            "county_name",
            "settlement_type",
            "broad_age_group",
            "sex",
            "exact_age",
        ],
        how="outer",
    )

    validation["allocated_agent_count"] = pd.to_numeric(
        validation["allocated_agent_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["generated_agent_count"] = pd.to_numeric(
        validation["generated_agent_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_generated_minus_allocated"] = (
        validation["generated_agent_count"]
        - validation["allocated_agent_count"]
    )

    validation["absolute_difference"] = validation[
        "difference_generated_minus_allocated"
    ].abs()

    validation = validation.sort_values(
        ["absolute_difference", "allocated_agent_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return validation


def create_summary_table(
    age_distribution_long: pd.DataFrame,
    agent_counts_by_area_broad_age: pd.DataFrame,
    age_sex_allocation_table: pd.DataFrame,
    total_agents_written: int,
    age_sex_assignment_validation: pd.DataFrame,
    flat_population_comparison: pd.DataFrame,
    flat_gender_comparison: pd.DataFrame,
    flat_age_group_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.append({
        "metric": "age_distribution_total",
        "value": age_distribution_long["value"].sum(),
        "note": "Az age_distribution.xlsx long formátumú összesített népessége.",
    })

    rows.append({
        "metric": "experimental_agents_total_before_age_sex_assignment",
        "value": agent_counts_by_area_broad_age["generated_agent_count"].sum(),
        "note": "A korábbi experimental agent fájl agent darabszáma.",
    })

    rows.append({
        "metric": "allocated_age_sex_total",
        "value": age_sex_allocation_table["allocated_agent_count"].sum(),
        "note": "Age/sex szinten allokált agentek száma.",
    })

    rows.append({
        "metric": "agents_written_with_exact_age_and_sex",
        "value": total_agents_written,
        "note": "Ténylegesen kiírt agentek száma exact_age és sex mezőkkel.",
    })

    rows.append({
        "metric": "age_sex_assignment_max_absolute_difference",
        "value": age_sex_assignment_validation["absolute_difference"].max(),
        "note": "Ennek 0-nak kell lennie: a kiírt age/sex agentek visszaadják-e az allocation táblát.",
    })

    rows.append({
        "metric": "age_sex_assignment_nonzero_difference_rows",
        "value": (
            age_sex_assignment_validation["difference_generated_minus_allocated"] != 0
        ).sum(),
        "note": "Ennek 0-nak kell lennie.",
    })

    rows.append({
        "metric": "flat_population_total",
        "value": flat_population_comparison["flat_population_total"].sum(),
        "note": "Flat Férfi+Nő településszintű népesség összesen.",
    })

    rows.append({
        "metric": "generated_population_total_for_flat_comparison",
        "value": flat_population_comparison["generated_agent_count"].sum(),
        "note": "Generált agentek száma settlement összevetésben.",
    })

    rows.append({
        "metric": "flat_population_difference_generated_minus_flat",
        "value": (
            flat_population_comparison["generated_agent_count"].sum()
            - flat_population_comparison["flat_population_total"].sum()
        ),
        "note": "Ez várhatóan kb. -6057, a korábbi residual miatt.",
    })

    rows.append({
        "metric": "flat_population_max_settlement_absolute_difference",
        "value": flat_population_comparison["absolute_difference"].max(),
        "note": "Legnagyobb településszintű népességeltérés.",
    })

    rows.append({
        "metric": "flat_gender_total_difference_generated_minus_flat",
        "value": (
            flat_gender_comparison["generated_sex_count"].sum()
            - flat_gender_comparison["flat_gender_count"].sum()
        ),
        "note": "Generált nem szerinti összeg mínusz flat nem szerinti összeg.",
    })

    rows.append({
        "metric": "flat_age_group_total_difference_generated_minus_flat",
        "value": (
            flat_age_group_comparison["generated_age_group_count"].sum()
            - flat_age_group_comparison["flat_age_group_count"].sum()
        ),
        "note": "Generált 10 éves korcsoport összeg mínusz flat korcsoport összeg.",
    })

    summary = pd.DataFrame(rows)

    return summary


# ============================================================
# 9. FŐ FUTTATÁSI FÜGGVÉNY
# ============================================================

def run_exact_age_and_sex_assignment() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Age distribution Excel beolvasása...")
    raw_age_distribution = read_age_distribution_raw_table(AGE_DISTRIBUTION_XLSX)

    print("Age distribution long formátum készítése...")
    age_distribution_long = convert_age_distribution_to_long_format(
        raw_table=raw_age_distribution
    )

    print("Age/sex valószínűségek számítása...")
    age_sex_probabilities = calculate_age_sex_probabilities(
        age_distribution_long=age_distribution_long
    )

    print("Experimental agentek első pass: area + broad age group darabszámok...")
    agent_counts_by_area_broad_age = count_agents_by_area_and_broad_age_group(
        experimental_agents_csv=EXPERIMENTAL_AGENTS_CSV
    )

    print("Exact age + sex allokáció készítése...")
    age_sex_allocation_table = create_age_sex_allocation_table(
        agent_counts_by_group=agent_counts_by_area_broad_age,
        age_sex_probabilities=age_sex_probabilities
    )

    print("Age/sex remaining pool készítése...")
    age_sex_pool = build_remaining_age_sex_pool(
        age_sex_allocation_table=age_sex_allocation_table
    )

    print("Experimental agentek második pass: exact_age + sex mezők kiírása...")
    assignment_outputs = assign_exact_age_and_sex_to_agents(
        experimental_agents_csv=EXPERIMENTAL_AGENTS_CSV,
        output_agents_csv=AGENTS_WITH_EXACT_AGE_SEX_CSV,
        age_sex_pool=age_sex_pool,
    )

    print("Age/sex assignment validáció készítése...")
    age_sex_assignment_validation = create_age_sex_assignment_validation(
        age_sex_allocation_table=age_sex_allocation_table,
        generated_counts_by_area_age_sex=assignment_outputs[
            "generated_counts_by_area_age_sex"
        ],
    )

    print("Flat long tábla beolvasása...")
    flat_long_table = load_flat_long_table(FLAT_LONG_TABLE_CSV)

    print("Flat settlement population targetek kinyerése...")
    flat_population_targets = extract_flat_settlement_population_targets(
        flat_long_table=flat_long_table
    )

    print("Flat settlement gender targetek kinyerése...")
    flat_gender_targets = extract_flat_settlement_gender_targets(
        flat_long_table=flat_long_table
    )

    print("Flat settlement age group targetek kinyerése...")
    flat_age_group_targets = extract_flat_settlement_age_group_targets(
        flat_long_table=flat_long_table
    )

    print("Flat settlement population összevetés...")
    flat_population_comparison = create_flat_settlement_population_comparison(
        flat_population_targets=flat_population_targets,
        generated_counts_by_settlement=assignment_outputs[
            "generated_counts_by_settlement"
        ],
    )

    print("Flat settlement gender összevetés...")
    flat_gender_comparison = create_flat_settlement_gender_comparison(
        flat_gender_targets=flat_gender_targets,
        generated_counts_by_settlement_sex=assignment_outputs[
            "generated_counts_by_settlement_sex"
        ],
    )

    print("Flat settlement age group összevetés...")
    flat_age_group_comparison = create_flat_settlement_age_group_comparison(
        flat_age_group_targets=flat_age_group_targets,
        generated_counts_by_settlement_age_group=assignment_outputs[
            "generated_counts_by_settlement_age_group"
        ],
    )

    print("Összefoglaló készítése...")
    summary = create_summary_table(
        age_distribution_long=age_distribution_long,
        agent_counts_by_area_broad_age=agent_counts_by_area_broad_age,
        age_sex_allocation_table=age_sex_allocation_table,
        total_agents_written=assignment_outputs["total_agents_written"],
        age_sex_assignment_validation=age_sex_assignment_validation,
        flat_population_comparison=flat_population_comparison,
        flat_gender_comparison=flat_gender_comparison,
        flat_age_group_comparison=flat_age_group_comparison,
    )

    print("Outputok mentése...")

    age_distribution_long.to_csv(
        AGE_SEX_DISTRIBUTION_LONG_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    age_sex_probabilities.to_csv(
        AGE_SEX_PROBABILITIES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    agent_counts_by_area_broad_age.to_csv(
        AGENT_COUNTS_BY_AREA_BROAD_AGE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    age_sex_assignment_validation.to_csv(
        AGE_SEX_ASSIGNMENT_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    flat_population_comparison.to_csv(
        FLAT_SETTLEMENT_POPULATION_COMPARISON_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    flat_gender_comparison.to_csv(
        FLAT_SETTLEMENT_GENDER_COMPARISON_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    flat_age_group_comparison.to_csv(
        FLAT_SETTLEMENT_AGE_GROUP_COMPARISON_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        AGE_SEX_ASSIGNMENT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Kiírt agentek exact_age + sex mezőkkel: {assignment_outputs['total_agents_written']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


# ============================================================
# 10. PROGRAM INDÍTÁSA
# ==========================================f==================

if __name__ == "__main__":
    run_exact_age_and_sex_assignment()