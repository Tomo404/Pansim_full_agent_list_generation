from pathlib import Path
import csv
import re
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

CURRENT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/64_agents_with_activity_education_activity_calibrated.csv"
)

FLAT_LONG_TABLE_CSV = Path(
    "../outputs/flat_validation/05_flat_combined_long_format_table.csv"
)

ACTIVITY_EDUCATION_HIER_LONG_CSV = Path(
    "../outputs/agent_attribute_generation/56_activity_education_hier_long.csv"
)

OUTPUT_FOLDER = Path("../outputs/agent_attribute_generation")

REFERENCE_TARGETS_CSV = OUTPUT_FOLDER / "68_current_agent_demographic_education_reference_targets.csv"
SEX_VALIDATION_CSV = OUTPUT_FOLDER / "69_current_agent_sex_validation.csv"
AGE_GROUP_VALIDATION_CSV = OUTPUT_FOLDER / "70_current_agent_age_group_validation.csv"
EDUCATION_VALIDATION_CSV = OUTPUT_FOLDER / "71_current_agent_education_validation.csv"
ACTIVITY_EDUCATION_VALIDATION_CSV = OUTPUT_FOLDER / "72_current_agent_activity_education_validation.csv"
FULL_HIER_ACTIVITY_EDUCATION_VALIDATION_CSV = OUTPUT_FOLDER / "73_current_agent_full_hier_activity_education_validation.csv"
SUMMARY_CSV = OUTPUT_FOLDER / "74_current_agent_demographic_education_summary.csv"
METHOD_NOTES_CSV = OUTPUT_FOLDER / "75_current_agent_demographic_education_method_notes.csv"

MAX_AGENTS_TO_PROCESS = None

CLEAN_AGE_GROUP_ORDER = [
    "0_9",
    "10_19",
    "20_29",
    "30_39",
    "40_49",
    "50_59",
    "60_69",
    "70_79",
    "80_89",
    "90_plus",
]


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


def normalize_settlement_name(raw_name: str) -> str:
    if pd.isna(raw_name):
        return ""

    normalized = str(raw_name).strip().lower()
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[\.,;:()\[\]/\\]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


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


def sex_to_flat_label(sex: str) -> str:
    sex = normalize_sex_label(sex)

    if sex == "male":
        return "Férfi"

    if sex == "female":
        return "Nő"

    return sex


def normalize_activity_label(raw_label: str) -> str:
    label = normalize_text(raw_label)

    manual = {
        "Foglalkoztatott": "Foglalkoztatott",
        "Munkanélküli": "Munkanélküli",
        "Ellátásban részesülő inaktív": "Ellátásban részesülő inaktív",
        "Eltartott": "Eltartott",
        "15 évesnél fiatalabb": "15 évesnél fiatalabb",
        "15 évesnél fiatalabb személy": "15 évesnél fiatalabb",
        "15 évesnél fiatalabbak": "15 évesnél fiatalabb",
    }

    return manual.get(label, label)


def normalize_education_label(raw_label: str) -> str:
    return normalize_text(raw_label)


def map_exact_age_to_clean_10_year_age_group(exact_age: int) -> str:
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


def map_exact_age_to_activity_education_age_group(exact_age: int) -> str:
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
# 3. FLAT TARGETEK
# ============================================================

def load_flat_long_table(flat_long_table_csv: Path) -> pd.DataFrame:
    flat_long_table = pd.read_csv(flat_long_table_csv)

    flat_long_table["settlement_key"] = flat_long_table["settlement"].apply(
        normalize_settlement_name
    )

    flat_long_table["value"] = flat_long_table["value"].apply(
        convert_count_to_integer
    )

    return flat_long_table


def parse_clean_flat_10_year_age_group_category(category: str) -> str | None:
    text = str(category).strip().lower()

    if text == "10 évesnél fiatalabb":
        return "0_9"

    accepted_ranges = {
        (10, 19): "10_19",
        (20, 29): "20_29",
        (30, 39): "30_39",
        (40, 49): "40_49",
        (50, 59): "50_59",
        (60, 69): "60_69",
        (70, 79): "70_79",
        (80, 89): "80_89",
    }

    range_match = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)\s*éves", text)

    if range_match:
        min_age = int(range_match.group(1))
        max_age = int(range_match.group(2))

        return accepted_ranges.get((min_age, max_age), None)

    if text == "90 éves és idősebb":
        return "90_plus"

    return None


def extract_flat_sex_targets(flat_long_table: pd.DataFrame) -> pd.DataFrame:
    candidate_rows = flat_long_table[
        flat_long_table["source_file"].str.contains("nepesseg", case=False, regex=False)
    ].copy()

    rows = []

    for _, row in candidate_rows.iterrows():
        category = normalize_text(row["category"])

        if category not in ["Férfi", "Nő"]:
            continue

        rows.append({
            "sex_flat_label": category,
            "flat_sex_count": convert_count_to_integer(row["value"]),
        })

    targets = (
        pd.DataFrame(rows)
        .groupby("sex_flat_label", dropna=False)["flat_sex_count"]
        .sum()
        .reset_index()
    )

    return targets


def extract_flat_age_group_targets(flat_long_table: pd.DataFrame) -> pd.DataFrame:
    candidate_rows = flat_long_table[
        flat_long_table["source_file"].str.contains("nepesseg", case=False, regex=False)
    ].copy()

    rows = []

    for _, row in candidate_rows.iterrows():
        age_group = parse_clean_flat_10_year_age_group_category(row["category"])

        if age_group is None:
            continue

        rows.append({
            "clean_age_group_label": age_group,
            "flat_age_group_count": convert_count_to_integer(row["value"]),
        })

    targets = (
        pd.DataFrame(rows)
        .groupby("clean_age_group_label", dropna=False)["flat_age_group_count"]
        .sum()
        .reset_index()
    )

    return targets


# ============================================================
# 4. HIER TARGETEK
# ============================================================

def load_activity_education_hier_long(input_csv: Path) -> pd.DataFrame:
    hier_long = pd.read_csv(input_csv)

    hier_long["county_name"] = hier_long["county_name"].apply(normalize_county_name)
    hier_long["settlement_type"] = hier_long["settlement_type"].apply(normalize_settlement_type)
    hier_long["sex"] = hier_long["sex"].apply(normalize_sex_label)
    hier_long["age_group"] = hier_long["age_group"].apply(normalize_text)
    hier_long["economic_activity_status"] = hier_long["economic_activity_status"].apply(normalize_activity_label)
    hier_long["education_level"] = hier_long["education_level"].apply(normalize_education_label)
    hier_long["value"] = hier_long["value"].apply(convert_count_to_integer)

    return hier_long


def create_hier_education_targets(hier_long: pd.DataFrame) -> pd.DataFrame:
    targets = (
        hier_long
        .groupby("education_level", dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "hier_education_count"})
    )

    return targets


def create_hier_activity_education_targets(hier_long: pd.DataFrame) -> pd.DataFrame:
    targets = (
        hier_long
        .groupby(
            [
                "economic_activity_status",
                "education_level",
            ],
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "hier_activity_education_count"})
    )

    return targets


def create_full_hier_activity_education_targets(hier_long: pd.DataFrame) -> pd.DataFrame:
    targets = (
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
        .rename(columns={"value": "hier_full_count"})
    )

    return targets


# ============================================================
# 5. CURRENT AGENTEK SZÁMLÁLÁSA
# ============================================================

def count_current_agents(current_agents_csv: Path) -> dict:
    sex_counts = defaultdict(int)
    age_group_counts = defaultdict(int)
    education_counts = defaultdict(int)
    activity_education_counts = defaultdict(int)
    full_activity_education_counts = defaultdict(int)
    activity_counts = defaultdict(int)

    total_agents = 0
    under_15_agents = 0
    age_15plus_agents = 0

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = [
            "sex",
            "exact_age",
            "county_name",
            "settlement_type",
            "economic_activity_status_calibrated",
            "education_level_calibrated",
        ]

        for required_column in required_columns:
            if required_column not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó kötelező oszlop a 64-es agentfájlból: {required_column}"
                )

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents >= MAX_AGENTS_TO_PROCESS:
                break

            sex = normalize_sex_label(row["sex"])
            sex_flat_label = sex_to_flat_label(sex)

            exact_age = convert_count_to_integer(row["exact_age"])
            clean_age_group = map_exact_age_to_clean_10_year_age_group(exact_age)
            activity_education_age_group = map_exact_age_to_activity_education_age_group(exact_age)

            county_name = normalize_county_name(row["county_name"])
            settlement_type = normalize_settlement_type(row["settlement_type"])

            activity = normalize_activity_label(
                row["economic_activity_status_calibrated"]
            )

            education = normalize_education_label(
                row["education_level_calibrated"]
            )

            sex_counts[sex_flat_label] += 1
            age_group_counts[clean_age_group] += 1
            activity_counts[activity] += 1

            if exact_age < 15:
                under_15_agents += 1
            else:
                age_15plus_agents += 1

                education_counts[education] += 1

                activity_education_counts[
                    (
                        activity,
                        education,
                    )
                ] += 1

                full_activity_education_counts[
                    (
                        county_name,
                        settlement_type,
                        sex,
                        activity_education_age_group,
                        activity,
                        education,
                    )
                ] += 1

            total_agents += 1

            if total_agents % 1_000_000 == 0:
                print(f"Beolvasott current agentek: {total_agents:,}")

    return {
        "total_agents": total_agents,
        "under_15_agents": under_15_agents,
        "age_15plus_agents": age_15plus_agents,
        "sex_counts": dict(sex_counts),
        "age_group_counts": dict(age_group_counts),
        "education_counts": dict(education_counts),
        "activity_counts": dict(activity_counts),
        "activity_education_counts": dict(activity_education_counts),
        "full_activity_education_counts": dict(full_activity_education_counts),
    }


def counter_to_dataframe(counter: dict, key_columns: list[str], value_column: str) -> pd.DataFrame:
    rows = []

    for key, value in counter.items():
        if not isinstance(key, tuple):
            key = (key,)

        row = {}

        for index, column in enumerate(key_columns):
            row[column] = key[index]

        row[value_column] = value
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. VALIDÁCIÓS TÁBLÁK
# ============================================================

def create_sex_validation(flat_targets: pd.DataFrame, counts: dict) -> pd.DataFrame:
    generated = counter_to_dataframe(
        counts["sex_counts"],
        ["sex_flat_label"],
        "generated_sex_count",
    )

    validation = flat_targets.merge(
        generated,
        on="sex_flat_label",
        how="outer",
    )

    validation["flat_sex_count"] = pd.to_numeric(
        validation["flat_sex_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["generated_sex_count"] = pd.to_numeric(
        validation["generated_sex_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_generated_minus_flat"] = (
        validation["generated_sex_count"] - validation["flat_sex_count"]
    )

    validation["absolute_difference"] = validation["difference_generated_minus_flat"].abs()

    validation["relative_abs_error"] = validation.apply(
        lambda row: safe_divide(row["absolute_difference"], row["flat_sex_count"]),
        axis=1,
    )

    return validation.sort_values("absolute_difference", ascending=False).reset_index(drop=True)


def create_age_group_validation(flat_targets: pd.DataFrame, counts: dict) -> pd.DataFrame:
    generated = counter_to_dataframe(
        counts["age_group_counts"],
        ["clean_age_group_label"],
        "generated_age_group_count",
    )

    validation = flat_targets.merge(
        generated,
        on="clean_age_group_label",
        how="outer",
    )

    validation["flat_age_group_count"] = pd.to_numeric(
        validation["flat_age_group_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["generated_age_group_count"] = pd.to_numeric(
        validation["generated_age_group_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_generated_minus_flat"] = (
        validation["generated_age_group_count"] - validation["flat_age_group_count"]
    )

    validation["absolute_difference"] = validation["difference_generated_minus_flat"].abs()

    validation["relative_abs_error"] = validation.apply(
        lambda row: safe_divide(row["absolute_difference"], row["flat_age_group_count"]),
        axis=1,
    )

    validation["age_group_sort_order"] = validation["clean_age_group_label"].apply(
        lambda label: CLEAN_AGE_GROUP_ORDER.index(label)
        if label in CLEAN_AGE_GROUP_ORDER
        else 999
    )

    return validation.sort_values("age_group_sort_order").reset_index(drop=True)


def create_education_validation(hier_targets: pd.DataFrame, counts: dict) -> pd.DataFrame:
    generated = counter_to_dataframe(
        counts["education_counts"],
        ["education_level"],
        "generated_education_count",
    )

    validation = hier_targets.merge(
        generated,
        on="education_level",
        how="outer",
    )

    validation["hier_education_count"] = pd.to_numeric(
        validation["hier_education_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["generated_education_count"] = pd.to_numeric(
        validation["generated_education_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_generated_minus_hier"] = (
        validation["generated_education_count"] - validation["hier_education_count"]
    )

    validation["absolute_difference"] = validation["difference_generated_minus_hier"].abs()

    validation["relative_abs_error"] = validation.apply(
        lambda row: safe_divide(row["absolute_difference"], row["hier_education_count"]),
        axis=1,
    )

    return validation.sort_values(
        ["absolute_difference", "hier_education_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def create_activity_education_validation(hier_targets: pd.DataFrame, counts: dict) -> pd.DataFrame:
    generated = counter_to_dataframe(
        counts["activity_education_counts"],
        [
            "economic_activity_status",
            "education_level",
        ],
        "generated_activity_education_count",
    )

    validation = hier_targets.merge(
        generated,
        on=[
            "economic_activity_status",
            "education_level",
        ],
        how="outer",
    )

    validation["hier_activity_education_count"] = pd.to_numeric(
        validation["hier_activity_education_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["generated_activity_education_count"] = pd.to_numeric(
        validation["generated_activity_education_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_generated_minus_hier"] = (
        validation["generated_activity_education_count"]
        - validation["hier_activity_education_count"]
    )

    validation["absolute_difference"] = validation["difference_generated_minus_hier"].abs()

    validation["relative_abs_error"] = validation.apply(
        lambda row: safe_divide(row["absolute_difference"], row["hier_activity_education_count"]),
        axis=1,
    )

    return validation.sort_values(
        ["absolute_difference", "hier_activity_education_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def create_full_hier_activity_education_validation(hier_targets: pd.DataFrame, counts: dict) -> pd.DataFrame:
    generated = counter_to_dataframe(
        counts["full_activity_education_counts"],
        [
            "county_name",
            "settlement_type",
            "sex",
            "age_group",
            "economic_activity_status",
            "education_level",
        ],
        "generated_full_count",
    )

    validation = hier_targets.merge(
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

    validation["hier_full_count"] = pd.to_numeric(
        validation["hier_full_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["generated_full_count"] = pd.to_numeric(
        validation["generated_full_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_generated_minus_hier"] = (
        validation["generated_full_count"] - validation["hier_full_count"]
    )

    validation["absolute_difference"] = validation["difference_generated_minus_hier"].abs()

    validation["relative_abs_error"] = validation.apply(
        lambda row: safe_divide(row["absolute_difference"], row["hier_full_count"]),
        axis=1,
    )

    return validation.sort_values(
        ["absolute_difference", "hier_full_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


# ============================================================
# 7. REFERENCE ÉS SUMMARY
# ============================================================

def write_reference_targets(
    flat_sex_targets: pd.DataFrame,
    flat_age_targets: pd.DataFrame,
    hier_long: pd.DataFrame,
) -> None:
    rows = []

    for _, row in flat_sex_targets.iterrows():
        rows.append({
            "reference_name": "flat_sex",
            "source_file": "flat_A_nepesseg_adatok_telepulesenkent.xlsx / 05_flat_combined_long_format_table.csv",
            "category": row["sex_flat_label"],
            "target_count": row["flat_sex_count"],
            "note": "Nem szerinti flat target.",
        })

    for _, row in flat_age_targets.iterrows():
        rows.append({
            "reference_name": "flat_clean_10_year_age_group",
            "source_file": "flat_A_nepesseg_adatok_telepulesenkent.xlsx / 05_flat_combined_long_format_table.csv",
            "category": row["clean_age_group_label"],
            "target_count": row["flat_age_group_count"],
            "note": "Tiszta 10 éves korcsoport flat target.",
        })

    rows.append({
        "reference_name": "hier_activity_education_15plus",
        "source_file": "hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg.xlsx / 56_activity_education_hier_long.csv",
        "category": "total",
        "target_count": hier_long["value"].sum(),
        "note": "15+ nem × kor × gazdasági aktivitás × iskolai végzettség hier target összesen.",
    })

    pd.DataFrame(rows).to_csv(
        REFERENCE_TARGETS_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def create_summary(
    counts: dict,
    sex_validation: pd.DataFrame,
    age_validation: pd.DataFrame,
    education_validation: pd.DataFrame,
    activity_education_validation: pd.DataFrame,
    full_hier_validation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.append({
        "metric": "current_agents_total",
        "value": counts["total_agents"],
        "note": "A 64-es jelenlegi agentfájl összes agentje.",
    })

    rows.append({
        "metric": "current_under_15_agents",
        "value": counts["under_15_agents"],
        "note": "Exact_age < 15 agentek.",
    })

    rows.append({
        "metric": "current_15plus_agents",
        "value": counts["age_15plus_agents"],
        "note": "Exact_age >= 15 agentek.",
    })

    rows.append({
        "metric": "flat_sex_total",
        "value": sex_validation["flat_sex_count"].sum(),
        "note": "Flat Férfi+Nő target total.",
    })

    rows.append({
        "metric": "generated_sex_total",
        "value": sex_validation["generated_sex_count"].sum(),
        "note": "Generated sex total.",
    })

    rows.append({
        "metric": "sex_generated_minus_flat_total",
        "value": sex_validation["generated_sex_count"].sum() - sex_validation["flat_sex_count"].sum(),
        "note": "Generated sex total - flat sex total.",
    })

    rows.append({
        "metric": "sex_sum_absolute_difference",
        "value": sex_validation["absolute_difference"].sum(),
        "note": "Nem szerinti abszolút eltérések összege.",
    })

    rows.append({
        "metric": "sex_max_absolute_difference",
        "value": sex_validation["absolute_difference"].max(),
        "note": "Legnagyobb nem szerinti eltérés.",
    })

    rows.append({
        "metric": "flat_age_group_total",
        "value": age_validation["flat_age_group_count"].sum(),
        "note": "Flat tiszta 10 éves korcsoport total.",
    })

    rows.append({
        "metric": "generated_age_group_total",
        "value": age_validation["generated_age_group_count"].sum(),
        "note": "Generated 10 éves korcsoport total.",
    })

    rows.append({
        "metric": "age_group_generated_minus_flat_total",
        "value": age_validation["generated_age_group_count"].sum() - age_validation["flat_age_group_count"].sum(),
        "note": "Generated korcsoport total - flat korcsoport total.",
    })

    rows.append({
        "metric": "age_group_sum_absolute_difference",
        "value": age_validation["absolute_difference"].sum(),
        "note": "Tiszta 10 éves korcsoport abszolút eltérések összege.",
    })

    rows.append({
        "metric": "age_group_max_absolute_difference",
        "value": age_validation["absolute_difference"].max(),
        "note": "Legnagyobb 10 éves korcsoport eltérés.",
    })

    rows.append({
        "metric": "hier_education_total",
        "value": education_validation["hier_education_count"].sum(),
        "note": "Hier 15+ végzettség target total.",
    })

    rows.append({
        "metric": "generated_education_total",
        "value": education_validation["generated_education_count"].sum(),
        "note": "Generated 15+ végzettség total.",
    })

    rows.append({
        "metric": "education_generated_minus_hier_total",
        "value": education_validation["generated_education_count"].sum() - education_validation["hier_education_count"].sum(),
        "note": "Generated 15+ végzettség total - hier 15+ target total.",
    })

    rows.append({
        "metric": "education_sum_absolute_difference",
        "value": education_validation["absolute_difference"].sum(),
        "note": "Végzettség szerinti abszolút eltérések összege.",
    })

    rows.append({
        "metric": "education_max_absolute_difference",
        "value": education_validation["absolute_difference"].max(),
        "note": "Legnagyobb végzettség szerinti eltérés.",
    })

    rows.append({
        "metric": "activity_education_sum_absolute_difference",
        "value": activity_education_validation["absolute_difference"].sum(),
        "note": "Activity × education abszolút eltérések összege a hier targethez képest.",
    })

    rows.append({
        "metric": "activity_education_max_absolute_difference",
        "value": activity_education_validation["absolute_difference"].max(),
        "note": "Legnagyobb activity × education eltérés.",
    })

    rows.append({
        "metric": "full_hier_sum_absolute_difference",
        "value": full_hier_validation["absolute_difference"].sum(),
        "note": "Teljes hier dimenzió szerinti abszolút eltérés.",
    })

    rows.append({
        "metric": "full_hier_max_absolute_difference",
        "value": full_hier_validation["absolute_difference"].max(),
        "note": "Legnagyobb teljes hier cellaeltérés.",
    })

    return pd.DataFrame(rows)


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script nem módosít agenteket, csak validálja a 64-es jelenlegi agentfájlt.",
        },
        {
            "order": 2,
            "note": "A sex validáció a flat Férfi/Nő targetekhez hasonlít.",
        },
        {
            "order": 3,
            "note": "Az age validáció a flat tiszta 10 éves korcsoportokhoz hasonlít.",
        },
        {
            "order": 4,
            "note": "Az education validáció a hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg 15+ táblából aggregált végzettség targethez hasonlít.",
        },
        {
            "order": 5,
            "note": "Az activity × education validáció ugyanehhez a hier táblához hasonlít, de aggregált activity × education szinten.",
        },
        {
            "order": 6,
            "note": "A teljes hier validáció county × settlement_type × sex × age_group × activity × education szinten hasonlít.",
        },
        {
            "order": 7,
            "note": "Mivel a 64-es fájl activity-kalibráció után van, az activity mező már nem feltétlenül adja vissza pontosan az eredeti hier activity × education eloszlást.",
        },
        {
            "order": 8,
            "note": "A 15 év alatti agenteknek nincs hier végzettségi validációja ebben a táblában, mert a hier activity/education tábla 15+ népességet tartalmaz.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 8. FŐ FUTTATÁS
# ============================================================

def run_current_demographic_education_validation() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Flat long table beolvasása...")
    flat_long_table = load_flat_long_table(
        flat_long_table_csv=FLAT_LONG_TABLE_CSV
    )

    print("Flat sex targetek kinyerése...")
    flat_sex_targets = extract_flat_sex_targets(
        flat_long_table=flat_long_table
    )

    print("Flat age group targetek kinyerése...")
    flat_age_targets = extract_flat_age_group_targets(
        flat_long_table=flat_long_table
    )

    print("Hier activity/education long beolvasása...")
    hier_long = load_activity_education_hier_long(
        input_csv=ACTIVITY_EDUCATION_HIER_LONG_CSV
    )

    print("Hier education targetek készítése...")
    hier_education_targets = create_hier_education_targets(
        hier_long=hier_long
    )

    print("Hier activity × education targetek készítése...")
    hier_activity_education_targets = create_hier_activity_education_targets(
        hier_long=hier_long
    )

    print("Teljes hier activity/education targetek készítése...")
    full_hier_targets = create_full_hier_activity_education_targets(
        hier_long=hier_long
    )

    print("Current 64-es agentfájl számlálása...")
    counts = count_current_agents(
        current_agents_csv=CURRENT_AGENTS_CSV
    )

    print("Sex validáció...")
    sex_validation = create_sex_validation(
        flat_targets=flat_sex_targets,
        counts=counts
    )

    print("Age group validáció...")
    age_validation = create_age_group_validation(
        flat_targets=flat_age_targets,
        counts=counts
    )

    print("Education validáció...")
    education_validation = create_education_validation(
        hier_targets=hier_education_targets,
        counts=counts
    )

    print("Activity × education validáció...")
    activity_education_validation = create_activity_education_validation(
        hier_targets=hier_activity_education_targets,
        counts=counts
    )

    print("Teljes hier activity/education validáció...")
    full_hier_validation = create_full_hier_activity_education_validation(
        hier_targets=full_hier_targets,
        counts=counts
    )

    print("Reference target fájl írása...")
    write_reference_targets(
        flat_sex_targets=flat_sex_targets,
        flat_age_targets=flat_age_targets,
        hier_long=hier_long,
    )

    print("Summary készítése...")
    summary = create_summary(
        counts=counts,
        sex_validation=sex_validation,
        age_validation=age_validation,
        education_validation=education_validation,
        activity_education_validation=activity_education_validation,
        full_hier_validation=full_hier_validation,
    )

    print("Method notes írása...")
    write_method_notes()

    print("Outputok mentése...")

    sex_validation.to_csv(
        SEX_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    age_validation.to_csv(
        AGE_GROUP_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    education_validation.to_csv(
        EDUCATION_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    activity_education_validation.to_csv(
        ACTIVITY_EDUCATION_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    full_hier_validation.to_csv(
        FULL_HIER_ACTIVITY_EDUCATION_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Current agentek összesen: {counts['total_agents']:,}")
    print(f"15 év alatti agentek: {counts['under_15_agents']:,}")
    print(f"15+ agentek: {counts['age_15plus_agents']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_current_demographic_education_validation()