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

AGE_CALIBRATED_AGENTS_CSV = Path(
    "../outputs/household_generation/43_generated_agents_with_exact_age_sex_age_calibrated.csv"
)

ACTIVITY_EDUCATION_HIER_XLSX = Path(
    "../data/raw_hier_tables/hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg.xlsx"
)

FLAT_LONG_TABLE_CSV = Path(
    "../outputs/flat_validation/05_flat_combined_long_format_table.csv"
)

OUTPUT_FOLDER = Path("../outputs/agent_attribute_generation")

AGENTS_WITH_ACTIVITY_EDUCATION_CSV = OUTPUT_FOLDER / "55_agents_with_activity_education.csv"

ACTIVITY_EDUCATION_HIER_LONG_CSV = OUTPUT_FOLDER / "56_activity_education_hier_long.csv"

ACTIVITY_EDUCATION_PROBABILITIES_CSV = OUTPUT_FOLDER / "57_activity_education_probabilities.csv"

ACTIVITY_EDUCATION_AGENT_COUNTS_CSV = OUTPUT_FOLDER / "58_agent_counts_by_activity_education_group.csv"

ACTIVITY_EDUCATION_ASSIGNMENT_VALIDATION_CSV = OUTPUT_FOLDER / "59_activity_education_assignment_validation.csv"

FLAT_ACTIVITY_VALIDATION_CSV = OUTPUT_FOLDER / "60_flat_activity_validation.csv"

ACTIVITY_EDUCATION_SUMMARY_CSV = OUTPUT_FOLDER / "61_activity_education_assignment_summary.csv"


SHEET_NAME = "Adattábla"

RANDOM_SEED = 42

# Teszthez pl. 100_000.
# Teljes futtatáshoz None.
MAX_AGENTS_TO_PROCESS = None


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


def convert_count_to_integer(raw_value) -> int:
    if raw_value is None:
        return 0

    if pd.isna(raw_value):
        return 0

    text_value = str(raw_value).strip()

    if text_value == "":
        return 0

    if re.fullmatch(r"\d{1,3}(\.\d{3})+", text_value):
        text_value = text_value.replace(".", "")

    return int(round(float(text_value)))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def normalize_county_name(raw_county_name: str) -> str:
    if pd.isna(raw_county_name):
        return ""

    county_name = str(raw_county_name).strip()

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
    if pd.isna(raw_settlement_type):
        return ""

    settlement_type = str(raw_settlement_type).strip()

    manual = {
        "Fováros": "Főváros",
        "fováros": "Főváros",
        "főváros": "Főváros",
        "Főváros": "Főváros",
    }

    return manual.get(settlement_type, settlement_type)


def normalize_sex_label(raw_sex_label: str) -> str:
    if pd.isna(raw_sex_label):
        return ""

    sex = str(raw_sex_label).strip().lower()

    if sex in ["férfi", "ferfi", "male"]:
        return "male"

    if sex in ["nő", "no", "female"]:
        return "female"

    return sex


def normalize_settlement_name(raw_name: str) -> str:
    if pd.isna(raw_name):
        return ""

    normalized = str(raw_name).strip().lower()

    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[\.,;:()\[\]/\\]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def normalize_activity_label(raw_label: str) -> str:
    label = normalize_text(raw_label)

    manual = {
        "Foglalkoztatott": "Foglalkoztatott",
        "Munkanélküli": "Munkanélküli",
        "Ellátásban részesülő inaktív": "Ellátásban részesülő inaktív",
        "Eltartott": "Eltartott",
        "15 évesnél fiatalabb": "15 évesnél fiatalabb",
        "15 évesnél fiatalabb személy": "15 évesnél fiatalabb",
    }

    return manual.get(label, label)


def normalize_education_label(raw_label: str) -> str:
    return normalize_text(raw_label)


def map_exact_age_to_activity_education_age_group(exact_age: int) -> str:
    """
    A hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg tábla korcsoportjaihoz igazít.

    A tábla 15+ népességet tartalmaz:
    15–19, 20–24, ..., 65–69, 70 éves és idősebb.
    """
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


# ============================================================
# 3. HIER TÁBLA BEOLVASÁSA LONG FORMÁTUMBA
# ============================================================

def detect_first_numeric_value_column(raw_table: pd.DataFrame) -> int:
    numeric_table = raw_table.apply(pd.to_numeric, errors="coerce")
    numeric_counts_by_column = numeric_table.notna().sum(axis=0)

    for column_index, numeric_count in numeric_counts_by_column.items():
        if numeric_count > 0:
            return int(column_index)

    raise ValueError("Nem találtam numerikus értékoszlopot.")


def detect_first_category_data_row(
    raw_table: pd.DataFrame,
    first_value_column_index: int,
) -> int:
    for row_index in range(len(raw_table)):
        left_cells = raw_table.iloc[row_index, :first_value_column_index].tolist()
        right_cells = raw_table.iloc[row_index, first_value_column_index:].tolist()

        has_left_category = any(normalize_text(cell) != "" for cell in left_cells)

        has_numeric_value = any(
            pd.to_numeric(pd.Series([cell]), errors="coerce").notna().iloc[0]
            for cell in right_cells
        )

        if has_left_category and has_numeric_value:
            return row_index

    raise ValueError("Nem találtam első kategória-adatsort.")


def build_geography_header_table(
    raw_table: pd.DataFrame,
    first_value_column_index: int,
    first_data_row_index: int,
) -> pd.DataFrame:
    """
    A KSH hier táblákban a data row előtti sorokból építjük a földrajzi fejlécet.

    Ennél a táblánál:
    - 0. sor: vármegye / Budapest
    - 1. sor: településtípus
    """
    header_rows = raw_table.iloc[:first_data_row_index, first_value_column_index:].copy()

    county_names = header_rows.iloc[0].ffill()
    settlement_types = header_rows.iloc[1].fillna("Nincs megadva")

    geography = pd.DataFrame({
        "value_column_index": header_rows.columns,
        "county_name": county_names.values,
        "settlement_type": settlement_types.values,
    })

    geography["county_name"] = geography["county_name"].apply(normalize_county_name)
    geography["settlement_type"] = geography["settlement_type"].apply(normalize_settlement_type)

    return geography


def read_activity_education_hier_long(input_xlsx: Path) -> pd.DataFrame:
    raw_table = pd.read_excel(
        input_xlsx,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl",
    )

    first_value_column_index = detect_first_numeric_value_column(raw_table)

    first_data_row_index = detect_first_category_data_row(
        raw_table=raw_table,
        first_value_column_index=first_value_column_index,
    )

    geography = build_geography_header_table(
        raw_table=raw_table,
        first_value_column_index=first_value_column_index,
        first_data_row_index=first_data_row_index,
    )

    category_part = raw_table.iloc[first_data_row_index:, :first_value_column_index].copy()
    category_part = category_part.ffill()

    category_part = category_part.rename(
        columns={
            0: "sex_raw",
            1: "age_group",
            2: "economic_activity_status",
            3: "education_level",
        }
    )

    value_part = raw_table.iloc[first_data_row_index:, first_value_column_index:].copy()
    value_part_numeric = value_part.apply(pd.to_numeric, errors="coerce").fillna(0)

    long_rows = []

    for local_row_position, original_row_index in enumerate(value_part_numeric.index):
        category_row = category_part.iloc[local_row_position]

        sex = normalize_sex_label(category_row["sex_raw"])
        age_group = normalize_text(category_row["age_group"])
        activity = normalize_activity_label(category_row["economic_activity_status"])
        education = normalize_education_label(category_row["education_level"])

        if sex not in ["male", "female"]:
            continue

        for value_column_index in value_part_numeric.columns:
            value = convert_count_to_integer(
                value_part_numeric.loc[original_row_index, value_column_index]
            )

            geo_row = geography[
                geography["value_column_index"] == value_column_index
            ].iloc[0]

            long_rows.append({
                "county_name": geo_row["county_name"],
                "settlement_type": geo_row["settlement_type"],
                "sex": sex,
                "age_group": age_group,
                "economic_activity_status": activity,
                "education_level": education,
                "value": value,
            })

    long_table = pd.DataFrame(long_rows)

    return long_table


def calculate_activity_education_probabilities(
    hier_long: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        hier_long
        .groupby(
            [
                "county_name",
                "settlement_type",
                "sex",
                "age_group",
                "economic_activity_status",
                "education_level",
            ],
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "hier_count"})
    )

    group_totals = (
        counts
        .groupby(
            ["county_name", "settlement_type", "sex", "age_group"],
            dropna=False,
        )["hier_count"]
        .sum()
        .reset_index()
        .rename(columns={"hier_count": "hier_group_total"})
    )

    probabilities = counts.merge(
        group_totals,
        on=["county_name", "settlement_type", "sex", "age_group"],
        how="left",
    )

    probabilities["probability"] = probabilities.apply(
        lambda row: safe_divide(row["hier_count"], row["hier_group_total"]),
        axis=1,
    )

    probabilities = probabilities.sort_values(
        [
            "county_name",
            "settlement_type",
            "sex",
            "age_group",
            "economic_activity_status",
            "education_level",
        ]
    ).reset_index(drop=True)

    return probabilities


# ============================================================
# 4. AGENTEK CSOPORTSZÁMLÁLÁSA
# ============================================================

def count_agents_by_activity_education_group(
    agents_csv: Path,
) -> dict:
    counts_by_group = defaultdict(int)
    total_agents_seen = 0
    under_15_agents = 0
    age_15plus_agents = 0

    with agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents_seen >= MAX_AGENTS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])

            if exact_age < 15:
                under_15_agents += 1
            else:
                county_name = normalize_county_name(row["county_name"])
                settlement_type = normalize_settlement_type(row["settlement_type"])
                sex = normalize_sex_label(row["sex"])
                age_group = map_exact_age_to_activity_education_age_group(exact_age)

                key = (
                    county_name,
                    settlement_type,
                    sex,
                    age_group,
                )

                counts_by_group[key] += 1
                age_15plus_agents += 1

            total_agents_seen += 1

            if total_agents_seen % 1_000_000 == 0:
                print(f"Első pass agentek: {total_agents_seen:,}")

    rows = []

    for key, count in counts_by_group.items():
        county_name, settlement_type, sex, age_group = key

        rows.append({
            "county_name": county_name,
            "settlement_type": settlement_type,
            "sex": sex,
            "age_group": age_group,
            "generated_agent_count": count,
        })

    count_table = pd.DataFrame(rows)

    return {
        "total_agents_seen": total_agents_seen,
        "under_15_agents": under_15_agents,
        "age_15plus_agents": age_15plus_agents,
        "count_table": count_table,
    }


# ============================================================
# 5. INTEGER ALLOKÁCIÓ
# ============================================================

def allocate_integer_counts_by_largest_remainder(
    probability_rows: pd.DataFrame,
    total_count_to_allocate: int,
) -> pd.DataFrame:
    allocation = probability_rows.copy()

    if total_count_to_allocate <= 0:
        allocation["expected_count"] = 0.0
        allocation["allocated_agent_count"] = 0
        allocation["allocation_remainder"] = 0.0
        return allocation

    allocation["expected_count"] = allocation["probability"] * total_count_to_allocate
    allocation["allocated_agent_count"] = allocation["expected_count"].apply(math.floor)

    already_allocated = int(allocation["allocated_agent_count"].sum())
    remaining = total_count_to_allocate - already_allocated

    allocation["allocation_remainder"] = (
        allocation["expected_count"] - allocation["allocated_agent_count"]
    )

    allocation = allocation.sort_values(
        ["allocation_remainder", "expected_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if remaining > 0:
        allocation.loc[allocation.index < remaining, "allocated_agent_count"] += 1

    allocation = allocation.sort_values(
        [
            "county_name",
            "settlement_type",
            "sex",
            "age_group",
            "economic_activity_status",
            "education_level",
        ]
    ).reset_index(drop=True)

    return allocation


def create_activity_education_allocation_table(
    agent_count_table: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    allocated_groups = []

    for _, count_row in agent_count_table.iterrows():
        county_name = count_row["county_name"]
        settlement_type = count_row["settlement_type"]
        sex = count_row["sex"]
        age_group = count_row["age_group"]
        generated_agent_count = convert_count_to_integer(count_row["generated_agent_count"])

        probability_rows = probabilities[
            (probabilities["county_name"] == county_name)
            & (probabilities["settlement_type"] == settlement_type)
            & (probabilities["sex"] == sex)
            & (probabilities["age_group"] == age_group)
        ].copy()

        if probability_rows.empty:
            raise ValueError(
                f"Nincs activity/education probability ehhez a csoporthoz: "
                f"{county_name}, {settlement_type}, {sex}, {age_group}"
            )

        allocated = allocate_integer_counts_by_largest_remainder(
            probability_rows=probability_rows,
            total_count_to_allocate=generated_agent_count,
        )

        allocated["generated_agent_group_total"] = generated_agent_count

        allocated_groups.append(allocated)

    allocation_table = pd.concat(allocated_groups, ignore_index=True)

    allocation_table = allocation_table[
        allocation_table["allocated_agent_count"] > 0
    ].copy()

    return allocation_table


# ============================================================
# 6. POOL ÉPÍTÉSE
# ============================================================

def build_option_id_lookup(allocation_table: pd.DataFrame) -> tuple[dict, dict]:
    option_to_id = {}
    id_to_option = {}

    next_id = 0

    for _, row in allocation_table.iterrows():
        option = (
            row["economic_activity_status"],
            row["education_level"],
        )

        if option not in option_to_id:
            option_to_id[option] = next_id
            id_to_option[next_id] = {
                "economic_activity_status": option[0],
                "education_level": option[1],
            }
            next_id += 1

    return option_to_id, id_to_option


def build_activity_education_pool(
    allocation_table: pd.DataFrame,
) -> tuple[dict, dict]:
    option_to_id, id_to_option = build_option_id_lookup(allocation_table)

    pool = defaultdict(list)

    for _, row in allocation_table.iterrows():
        key = (
            row["county_name"],
            row["settlement_type"],
            row["sex"],
            row["age_group"],
        )

        option = (
            row["economic_activity_status"],
            row["education_level"],
        )

        option_id = option_to_id[option]

        allocated_count = convert_count_to_integer(row["allocated_agent_count"])

        pool[key].extend([option_id] * allocated_count)

    random_generator = random.Random(RANDOM_SEED)

    print("Activity/education pool keverése csoportonként...")

    for key in pool:
        random_generator.shuffle(pool[key])

    return pool, id_to_option


def pop_activity_education_from_pool(
    pool: dict,
    id_to_option: dict,
    county_name: str,
    settlement_type: str,
    sex: str,
    age_group: str,
) -> dict:
    key = (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_sex_label(sex),
        age_group,
    )

    if key not in pool:
        raise ValueError(f"Nincs activity/education pool ehhez a csoporthoz: {key}")

    if len(pool[key]) == 0:
        raise ValueError(f"Elfogyott az activity/education pool ehhez a csoporthoz: {key}")

    option_id = pool[key].pop()

    return id_to_option[option_id]


# ============================================================
# 7. MÁSODIK PASS: ATTRIBÚTUMOK KIÍRÁSA
# ============================================================

def assign_activity_and_education_to_agents(
    input_agents_csv: Path,
    output_agents_csv: Path,
    pool: dict,
    id_to_option: dict,
) -> dict:
    generated_counts_by_hier_group = defaultdict(int)
    generated_counts_by_activity = defaultdict(int)

    total_agents_written = 0
    under_15_written = 0
    age_15plus_written = 0
    employed_written = 0

    with input_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)

        input_fieldnames = list(reader.fieldnames)

        new_columns = [
            "activity_education_age_group",
            "economic_activity_status_calibrated",
            "education_level_calibrated",
            "activity_education_assignment_source",
        ]

        output_fieldnames = input_fieldnames + [
            column for column in new_columns if column not in input_fieldnames
        ]

        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents_written >= MAX_AGENTS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])

            county_name = normalize_county_name(row["county_name"])
            settlement_type = normalize_settlement_type(row["settlement_type"])
            sex = normalize_sex_label(row["sex"])
            age_group = map_exact_age_to_activity_education_age_group(exact_age)

            row["county_name"] = county_name
            row["settlement_type"] = settlement_type
            row["sex"] = sex
            row["activity_education_age_group"] = age_group

            if exact_age < 15:
                activity = "15 évesnél fiatalabb"
                education = "15 évesnél fiatalabb személy"
                assignment_source = "under_15_rule"
                under_15_written += 1
            else:
                assigned = pop_activity_education_from_pool(
                    pool=pool,
                    id_to_option=id_to_option,
                    county_name=county_name,
                    settlement_type=settlement_type,
                    sex=sex,
                    age_group=age_group,
                )

                activity = assigned["economic_activity_status"]
                education = assigned["education_level"]
                assignment_source = "hier_15plus_activity_education"
                age_15plus_written += 1

                generated_counts_by_hier_group[
                    (
                        county_name,
                        settlement_type,
                        sex,
                        age_group,
                        activity,
                        education,
                    )
                ] += 1

            if activity == "Foglalkoztatott":
                employed_written += 1

            generated_counts_by_activity[
                activity
            ] += 1

            row["economic_activity_status_calibrated"] = activity
            row["education_level_calibrated"] = education
            row["activity_education_assignment_source"] = assignment_source

            writer.writerow(row)

            total_agents_written += 1

            if total_agents_written % 1_000_000 == 0:
                print(f"Második pass agentek: {total_agents_written:,}")

    remaining_pool_items = sum(len(items) for items in pool.values())

    return {
        "total_agents_written": total_agents_written,
        "under_15_written": under_15_written,
        "age_15plus_written": age_15plus_written,
        "employed_written": employed_written,
        "remaining_pool_items": remaining_pool_items,
        "generated_counts_by_hier_group": generated_counts_by_hier_group,
        "generated_counts_by_activity": generated_counts_by_activity,
    }


# ============================================================
# 8. VALIDÁCIÓK
# ============================================================

def convert_counter_to_dataframe(
    counter: dict,
    key_column_names: list[str],
    value_column_name: str,
) -> pd.DataFrame:
    rows = []

    for key_tuple, count in counter.items():
        row = {}

        for index, column_name in enumerate(key_column_names):
            row[column_name] = key_tuple[index]

        row[value_column_name] = count
        rows.append(row)

    return pd.DataFrame(rows)


def create_assignment_validation(
    allocation_table: pd.DataFrame,
    generated_counts_by_hier_group: dict,
) -> pd.DataFrame:
    generated = convert_counter_to_dataframe(
        counter=generated_counts_by_hier_group,
        key_column_names=[
            "county_name",
            "settlement_type",
            "sex",
            "age_group",
            "economic_activity_status",
            "education_level",
        ],
        value_column_name="generated_agent_count",
    )

    validation = allocation_table.merge(
        generated,
        on=[
            "county_name",
            "settlement_type",
            "sex",
            "age_group",
            "economic_activity_status",
            "education_level",
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


def load_flat_long_table(flat_long_table_csv: Path) -> pd.DataFrame:
    flat_long_table = pd.read_csv(flat_long_table_csv)

    flat_long_table["settlement_key"] = flat_long_table["settlement"].apply(
        normalize_settlement_name
    )

    flat_long_table["value"] = flat_long_table["value"].apply(
        convert_count_to_integer
    )

    return flat_long_table


def map_flat_activity_category(category: str) -> str | None:
    text = normalize_text(category)

    manual = {
        "Foglalkoztatott": "Foglalkoztatott",
        "Munkanélküli": "Munkanélküli",
        "Ellátásban részesülő inaktív": "Ellátásban részesülő inaktív",
        "Eltartott": "Eltartott",
        "15 évesnél fiatalabb": "15 évesnél fiatalabb",
        "15 évesnél fiatalabb személy": "15 évesnél fiatalabb",
        "15 évesnél fiatalabbak": "15 évesnél fiatalabb",
    }

    return manual.get(text, None)


def extract_flat_activity_targets(flat_long_table: pd.DataFrame) -> pd.DataFrame:
    candidate_rows = flat_long_table[
        flat_long_table["source_file"].str.contains("nepesseg", case=False, regex=False)
    ].copy()

    rows = []

    for _, row in candidate_rows.iterrows():
        mapped_activity = map_flat_activity_category(row["category"])

        if mapped_activity is None:
            continue

        rows.append({
            "economic_activity_status_calibrated": mapped_activity,
            "flat_activity_count": convert_count_to_integer(row["value"]),
        })

    targets = (
        pd.DataFrame(rows)
        .groupby("economic_activity_status_calibrated", dropna=False)["flat_activity_count"]
        .sum()
        .reset_index()
    )

    return targets


def create_flat_activity_validation(
    flat_activity_targets: pd.DataFrame,
    generated_counts_by_activity: dict,
) -> pd.DataFrame:
    generated = convert_counter_to_dataframe(
        counter={
            (activity,): count
            for activity, count in generated_counts_by_activity.items()
        },
        key_column_names=["economic_activity_status_calibrated"],
        value_column_name="generated_activity_count",
    )

    validation = flat_activity_targets.merge(
        generated,
        on="economic_activity_status_calibrated",
        how="outer",
    )

    validation["flat_activity_count"] = pd.to_numeric(
        validation["flat_activity_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["generated_activity_count"] = pd.to_numeric(
        validation["generated_activity_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_generated_minus_flat"] = (
        validation["generated_activity_count"]
        - validation["flat_activity_count"]
    )

    validation["absolute_difference"] = validation[
        "difference_generated_minus_flat"
    ].abs()

    validation["relative_abs_error"] = validation.apply(
        lambda row: safe_divide(
            row["absolute_difference"],
            row["flat_activity_count"],
        ),
        axis=1,
    )

    validation = validation.sort_values(
        ["absolute_difference", "flat_activity_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return validation


def create_summary(
    agent_count_result: dict,
    allocation_table: pd.DataFrame,
    assignment_outputs: dict,
    assignment_validation: pd.DataFrame,
    flat_activity_validation: pd.DataFrame,
    hier_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.append({
        "metric": "input_agents_total",
        "value": agent_count_result["total_agents_seen"],
        "note": "Input agentek száma a 43-as age-calibrated fájlban.",
    })

    rows.append({
        "metric": "output_agents_total",
        "value": assignment_outputs["total_agents_written"],
        "note": "Output agentek száma az 55-ös fájlban.",
    })

    rows.append({
        "metric": "under_15_agents_rule_assigned",
        "value": assignment_outputs["under_15_written"],
        "note": "15 év alatti agentek, szabály alapján kiosztva.",
    })

    rows.append({
        "metric": "age_15plus_agents_hier_assigned",
        "value": assignment_outputs["age_15plus_written"],
        "note": "15+ agentek, hier activity/education tábla alapján kiosztva.",
    })

    rows.append({
        "metric": "allocated_15plus_activity_education_total",
        "value": allocation_table["allocated_agent_count"].sum(),
        "note": "Hier probability alapján allokált 15+ agentek száma.",
    })

    rows.append({
        "metric": "remaining_pool_items_after_assignment",
        "value": assignment_outputs["remaining_pool_items"],
        "note": "Ennek 0-nak kell lennie.",
    })

    rows.append({
        "metric": "assignment_validation_max_absolute_difference",
        "value": assignment_validation["absolute_difference"].max(),
        "note": "Ennek 0-nak kell lennie.",
    })

    rows.append({
        "metric": "assignment_validation_nonzero_rows",
        "value": (
            assignment_validation["difference_generated_minus_allocated"] != 0
        ).sum(),
        "note": "Ennek 0-nak kell lennie.",
    })

    rows.append({
        "metric": "generated_employed_count",
        "value": assignment_outputs["employed_written"],
        "note": "Új calibrated Foglalkoztatott agentek száma.",
    })

    rows.append({
        "metric": "hier_15plus_source_total",
        "value": hier_long["value"].sum(),
        "note": "A hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg forrás összesített értéke.",
    })

    rows.append({
        "metric": "flat_activity_total",
        "value": flat_activity_validation["flat_activity_count"].sum(),
        "note": "Flat activity target összesen.",
    })

    rows.append({
        "metric": "generated_activity_total",
        "value": flat_activity_validation["generated_activity_count"].sum(),
        "note": "Generated activity total.",
    })

    rows.append({
        "metric": "flat_activity_generated_minus_flat_total",
        "value": (
            flat_activity_validation["generated_activity_count"].sum()
            - flat_activity_validation["flat_activity_count"].sum()
        ),
        "note": "Várhatóan residual / forráseltérés miatt nem feltétlenül 0.",
    })

    rows.append({
        "metric": "flat_activity_sum_absolute_difference",
        "value": flat_activity_validation["absolute_difference"].sum(),
        "note": "Flat activity kategóriák abszolút eltéréseinek összege.",
    })

    rows.append({
        "metric": "flat_activity_max_absolute_difference",
        "value": flat_activity_validation["absolute_difference"].max(),
        "note": "Legnagyobb activity kategória eltérés.",
    })

    return pd.DataFrame(rows)


# ============================================================
# 9. FŐ FUTTATÁS
# ============================================================

def run_activity_education_assignment() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Hier activity/education tábla long formátumra alakítása...")
    hier_long = read_activity_education_hier_long(
        input_xlsx=ACTIVITY_EDUCATION_HIER_XLSX
    )

    print("Activity/education probability táblázat készítése...")
    probabilities = calculate_activity_education_probabilities(
        hier_long=hier_long
    )

    print("Agentek első pass: county/type/sex/age_group darabszámok...")
    agent_count_result = count_agents_by_activity_education_group(
        agents_csv=AGE_CALIBRATED_AGENTS_CSV
    )

    print("Activity/education allokáció készítése...")
    allocation_table = create_activity_education_allocation_table(
        agent_count_table=agent_count_result["count_table"],
        probabilities=probabilities,
    )

    print("Activity/education pool építése...")
    pool, id_to_option = build_activity_education_pool(
        allocation_table=allocation_table
    )

    print("Agentek második pass: activity + education mezők kiírása...")
    assignment_outputs = assign_activity_and_education_to_agents(
        input_agents_csv=AGE_CALIBRATED_AGENTS_CSV,
        output_agents_csv=AGENTS_WITH_ACTIVITY_EDUCATION_CSV,
        pool=pool,
        id_to_option=id_to_option,
    )

    print("Assignment validáció készítése...")
    assignment_validation = create_assignment_validation(
        allocation_table=allocation_table,
        generated_counts_by_hier_group=assignment_outputs[
            "generated_counts_by_hier_group"
        ],
    )

    print("Flat activity validáció...")
    flat_long_table = load_flat_long_table(
        flat_long_table_csv=FLAT_LONG_TABLE_CSV
    )

    flat_activity_targets = extract_flat_activity_targets(
        flat_long_table=flat_long_table
    )

    flat_activity_validation = create_flat_activity_validation(
        flat_activity_targets=flat_activity_targets,
        generated_counts_by_activity=assignment_outputs[
            "generated_counts_by_activity"
        ],
    )

    print("Összefoglaló készítése...")
    summary = create_summary(
        agent_count_result=agent_count_result,
        allocation_table=allocation_table,
        assignment_outputs=assignment_outputs,
        assignment_validation=assignment_validation,
        flat_activity_validation=flat_activity_validation,
        hier_long=hier_long,
    )

    print("Outputok mentése...")

    hier_long.to_csv(
        ACTIVITY_EDUCATION_HIER_LONG_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    probabilities.to_csv(
        ACTIVITY_EDUCATION_PROBABILITIES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    agent_count_result["count_table"].to_csv(
        ACTIVITY_EDUCATION_AGENT_COUNTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    assignment_validation.to_csv(
        ACTIVITY_EDUCATION_ASSIGNMENT_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    flat_activity_validation.to_csv(
        FLAT_ACTIVITY_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        ACTIVITY_EDUCATION_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Input agentek: {agent_count_result['total_agents_seen']:,}")
    print(f"Output agentek: {assignment_outputs['total_agents_written']:,}")
    print(f"15 év alatti agentek: {assignment_outputs['under_15_written']:,}")
    print(f"15+ agentek: {assignment_outputs['age_15plus_written']:,}")
    print(f"Foglalkoztatott agentek: {assignment_outputs['employed_written']:,}")
    print(f"Maradék pool elem: {assignment_outputs['remaining_pool_items']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_activity_education_assignment()