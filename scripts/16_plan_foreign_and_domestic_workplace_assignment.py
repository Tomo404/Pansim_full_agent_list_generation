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

PREVIOUS_WORKER_WORKPLACE_CSV = Path(
    "../data/raw_reference/agents_with_workplaces.csv"
)

FOREIGN_EMPLOYED_HIER_XLSX = Path(
    "../data/raw_hier_tables/hier_kulfold_foglalkoztatott.xlsx"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/foreign_domestic_workplace_plan"
)

REFERENCE_TARGETS_CSV = OUTPUT_FOLDER / "83_foreign_domestic_workplace_reference_targets.csv"
FOREIGN_HIER_LONG_CSV = OUTPUT_FOLDER / "84_foreign_employed_hier_long.csv"
CURRENT_EMPLOYED_COUNTS_CSV = OUTPUT_FOLDER / "85_current_employed_counts_for_foreign_matching.csv"
FOREIGN_MATCHING_FEASIBILITY_CSV = OUTPUT_FOLDER / "86_foreign_matching_feasibility_comparison.csv"
DOMESTIC_WORKER_POOL_PLAN_CSV = OUTPUT_FOLDER / "87_domestic_worker_pool_usage_plan.csv"
PLAN_SUMMARY_CSV = OUTPUT_FOLDER / "88_foreign_domestic_workplace_plan_summary.csv"
METHOD_NOTES_CSV = OUTPUT_FOLDER / "89_foreign_domestic_workplace_method_notes.csv"

SHEET_NAME = "Adattábla"

MAX_ROWS_TO_PROCESS = None


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


def normalize_settlement_type(raw_settlement_type: str) -> str:
    settlement_type = normalize_text(raw_settlement_type)

    manual = {
        "Fováros": "Főváros",
        "fováros": "Főváros",
        "főváros": "Főváros",
        "Főváros": "Főváros",
    }

    return manual.get(settlement_type, settlement_type)


def normalize_sex_label(raw_sex: str) -> str:
    sex = normalize_text(raw_sex).lower()

    if sex in ["férfi", "ferfi", "male"]:
        return "male"

    if sex in ["nő", "no", "female"]:
        return "female"

    return sex


def normalize_age_group(raw_age_group: str) -> str:
    text = normalize_text(raw_age_group)
    text = text.replace("-", "–")
    return text


def normalize_education(raw_education: str) -> str:
    return normalize_text(raw_education)


def normalize_activity(raw_activity: str) -> str:
    activity = normalize_text(raw_activity)

    manual = {
        "Foglalkoztatott": "Foglalkoztatott",
        "employed": "Foglalkoztatott",
        "Munkanélküli": "Munkanélküli",
        "unemployed": "Munkanélküli",
        "Ellátásban részesülő inaktív": "Ellátásban részesülő inaktív",
        "inactive_benefit": "Ellátásban részesülő inaktív",
        "Eltartott": "Eltartott",
        "dependent": "Eltartott",
        "15 évesnél fiatalabb": "15 évesnél fiatalabb",
    }

    return manual.get(activity, activity)


def map_exact_age_to_worker_age_group(exact_age: int) -> str:
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


def make_foreign_match_key(
    county_name: str,
    settlement_type: str,
    sex: str,
    age_group: str,
    education: str,
) -> tuple[str, str, str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_sex_label(sex),
        normalize_age_group(age_group),
        normalize_education(education),
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# 3. HIER KÜLFÖLDI FOGLALKOZTATOTT TÁBLA LONG FORMÁTUM
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

        has_left_category = any(
            normalize_text(cell) != ""
            for cell in left_cells
        )

        has_numeric_value = any(
            pd.to_numeric(pd.Series([cell]), errors="coerce").notna().iloc[0]
            for cell in right_cells
        )

        if has_left_category and has_numeric_value:
            return row_index

    raise ValueError("Nem találtam első kategória-adatsort.")


def find_header_row_containing(raw_table: pd.DataFrame, text_to_find: str) -> int | None:
    target = text_to_find.lower()

    for row_index in range(min(20, len(raw_table))):
        row_text = " ".join(
            normalize_text(value).lower()
            for value in raw_table.iloc[row_index].tolist()
            if normalize_text(value) != ""
        )

        if target in row_text:
            return row_index

    return None


def build_foreign_geography_header(
    raw_table: pd.DataFrame,
    first_value_column_index: int,
    first_data_row_index: int,
) -> pd.DataFrame:
    """
    A külföldi foglalkoztatott táblában az értékoszlopok felett
    vármegye/régió és településtípus fejléc van.

    A Census export néha nem tartalmazza szó szerint a 'Vármegye, régió'
    cellaszöveget, ezért nem fix feliratot keresünk, hanem a sor tartalma alapján
    próbáljuk felismerni a földrajzi fejlécet.
    """

    known_county_fragments = [
        "budapest",
        "pest",
        "fejér",
        "komárom-esztergom",
        "veszprém",
        "győr-moson-sopron",
        "vas",
        "zala",
        "baranya",
        "somogy",
        "tolna",
        "borsod-abaúj-zemplén",
        "heves",
        "nógrád",
        "hajdú-bihar",
        "jász-nagykun-szolnok",
        "szabolcs-szatmár-bereg",
        "bács-kiskun",
        "békés",
        "csongrád-csanád",
    ]

    known_settlement_type_fragments = [
        "főváros",
        "megyei jogú város",
        "egyéb város",
        "község",
    ]

    best_county_row_index = None
    best_county_score = -1

    best_settlement_type_row_index = None
    best_settlement_type_score = -1

    # Csak az adatsor előtti fejléczónában keresünk.
    for row_index in range(0, first_data_row_index):
        row_values = raw_table.iloc[row_index, first_value_column_index:].tolist()

        normalized_values = [
            normalize_text(value).lower()
            for value in row_values
            if normalize_text(value) != ""
        ]

        county_score = 0

        for value in normalized_values:
            for county_fragment in known_county_fragments:
                if county_fragment in value:
                    county_score += 1
                    break

        settlement_type_score = 0

        for value in normalized_values:
            for settlement_type_fragment in known_settlement_type_fragments:
                if settlement_type_fragment in value:
                    settlement_type_score += 1
                    break

        if county_score > best_county_score:
            best_county_score = county_score
            best_county_row_index = row_index

        if settlement_type_score > best_settlement_type_score:
            best_settlement_type_score = settlement_type_score
            best_settlement_type_row_index = row_index

    if best_county_row_index is None or best_county_score <= 0:
        raise ValueError(
            "Nem találtam vármegye/régió fejlécsort tartalom alapján sem."
        )

    if best_settlement_type_row_index is None or best_settlement_type_score <= 0:
        raise ValueError(
            "Nem találtam településtípus fejlécsort tartalom alapján sem."
        )

    print(
        f"Felismert vármegye/régió fejlécsor Excel sor: {best_county_row_index + 1}, "
        f"score: {best_county_score}"
    )

    print(
        f"Felismert településtípus fejlécsor Excel sor: {best_settlement_type_row_index + 1}, "
        f"score: {best_settlement_type_score}"
    )

    county_values = raw_table.iloc[
        best_county_row_index,
        first_value_column_index:
    ].ffill()

    settlement_type_values = raw_table.iloc[
        best_settlement_type_row_index,
        first_value_column_index:
    ].ffill()

    geography = pd.DataFrame({
        "value_column_index": raw_table.columns[first_value_column_index:],
        "county_name": county_values.values,
        "settlement_type": settlement_type_values.values,
    })

    geography["county_name"] = geography["county_name"].apply(normalize_county_name)
    geography["settlement_type"] = geography["settlement_type"].apply(normalize_settlement_type)

    return geography


def read_foreign_employed_hier_long(input_xlsx: Path) -> pd.DataFrame:
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

    geography = build_foreign_geography_header(
        raw_table=raw_table,
        first_value_column_index=first_value_column_index,
        first_data_row_index=first_data_row_index,
    )

    category_part = raw_table.iloc[
        first_data_row_index:,
        :first_value_column_index
    ].copy().ffill()

    # A mentett tábla bal oldali kategóriaoszlopai a screenshot alapján:
    # Nemek, Korcsoport, Legmagasabb befejezett iskolai végzettség
    category_part = category_part.rename(
        columns={
            0: "sex_raw",
            1: "age_group_raw",
            2: "education_raw",
        }
    )

    value_part = raw_table.iloc[
        first_data_row_index:,
        first_value_column_index:
    ].copy()

    value_part_numeric = value_part.apply(pd.to_numeric, errors="coerce").fillna(0)

    long_rows = []

    for local_row_position, original_row_index in enumerate(value_part_numeric.index):
        category_row = category_part.iloc[local_row_position]

        sex = normalize_sex_label(category_row["sex_raw"])
        age_group = normalize_age_group(category_row["age_group_raw"])
        education = normalize_education(category_row["education_raw"])

        if sex not in ["male", "female"]:
            continue

        for value_column_index in value_part_numeric.columns:
            value = convert_count_to_integer(
                value_part_numeric.loc[original_row_index, value_column_index]
            )

            if value == 0:
                continue

            geo_row = geography[
                geography["value_column_index"] == value_column_index
            ].iloc[0]

            long_rows.append({
                "county_name": geo_row["county_name"],
                "settlement_type": geo_row["settlement_type"],
                "sex": sex,
                "age_group": age_group,
                "education": education,
                "foreign_employed_count": value,
            })

    long_table = pd.DataFrame(long_rows)

    return long_table


# ============================================================
# 4. CURRENT EMPLOYED AGENTEK SZÁMLÁLÁSA
# ============================================================

def count_current_employed_agents(current_agents_csv: Path) -> dict:
    counts_by_foreign_key = defaultdict(int)
    counts_by_county = defaultdict(int)
    counts_by_settlement_type = defaultdict(int)
    counts_by_sex = defaultdict(int)
    counts_by_age_group = defaultdict(int)
    counts_by_education = defaultdict(int)
    activity_counts = defaultdict(int)

    total_agents = 0
    employed_agents = 0

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = [
            "county_name",
            "settlement_type",
            "sex",
            "exact_age",
            "education_level_calibrated",
            "economic_activity_status_calibrated",
        ]

        for required_column in required_columns:
            if required_column not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop a current agent fájlban: {required_column}"
                )

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_agents >= MAX_ROWS_TO_PROCESS:
                break

            activity = normalize_activity(row["economic_activity_status_calibrated"])
            activity_counts[activity] += 1

            if activity == "Foglalkoztatott":
                county_name = normalize_county_name(row["county_name"])
                settlement_type = normalize_settlement_type(row["settlement_type"])
                sex = normalize_sex_label(row["sex"])
                exact_age = convert_count_to_integer(row["exact_age"])
                age_group = map_exact_age_to_worker_age_group(exact_age)
                education = normalize_education(row["education_level_calibrated"])

                key = make_foreign_match_key(
                    county_name=county_name,
                    settlement_type=settlement_type,
                    sex=sex,
                    age_group=age_group,
                    education=education,
                )

                counts_by_foreign_key[key] += 1
                counts_by_county[county_name] += 1
                counts_by_settlement_type[settlement_type] += 1
                counts_by_sex[sex] += 1
                counts_by_age_group[age_group] += 1
                counts_by_education[education] += 1

                employed_agents += 1

            total_agents += 1

            if total_agents % 1_000_000 == 0:
                print(f"Current agentek feldolgozva: {total_agents:,}")

    return {
        "total_agents": total_agents,
        "employed_agents": employed_agents,
        "activity_counts": dict(activity_counts),
        "counts_by_foreign_key": dict(counts_by_foreign_key),
        "counts_by_county": dict(counts_by_county),
        "counts_by_settlement_type": dict(counts_by_settlement_type),
        "counts_by_sex": dict(counts_by_sex),
        "counts_by_age_group": dict(counts_by_age_group),
        "counts_by_education": dict(counts_by_education),
    }


# ============================================================
# 5. RÉGI DOMESTIC WORKER POOL SZÁMLÁLÁSA
# ============================================================

def count_previous_worker_records(previous_workers_csv: Path) -> dict:
    total_rows = 0
    missing_workplace_id = 0
    used_fallback_true = 0
    used_fallback_false = 0
    used_teaor_fallback_true = 0
    used_teaor_fallback_false = 0

    counts_by_county = defaultdict(int)
    counts_by_workplace_county = defaultdict(int)
    counts_by_teaor = defaultdict(int)

    with previous_workers_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            county = normalize_county_name(row.get("county", ""))
            workplace_county = normalize_county_name(row.get("workplace_county", ""))
            workplace_teaor = normalize_text(row.get("workplace_teaor_code", ""))

            counts_by_county[county] += 1
            counts_by_workplace_county[workplace_county] += 1
            counts_by_teaor[workplace_teaor] += 1

            workplace_id = normalize_text(row.get("workplace_id", ""))

            if workplace_id == "":
                missing_workplace_id += 1

            used_fallback = normalize_text(row.get("used_fallback", "")).lower()

            if used_fallback == "true":
                used_fallback_true += 1
            elif used_fallback == "false":
                used_fallback_false += 1

            used_teaor_fallback = normalize_text(row.get("used_teaor_fallback", "")).lower()

            if used_teaor_fallback == "true":
                used_teaor_fallback_true += 1
            elif used_teaor_fallback == "false":
                used_teaor_fallback_false += 1

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"Régi worker rekordok feldolgozva: {total_rows:,}")

    return {
        "total_rows": total_rows,
        "missing_workplace_id": missing_workplace_id,
        "used_fallback_true": used_fallback_true,
        "used_fallback_false": used_fallback_false,
        "used_teaor_fallback_true": used_teaor_fallback_true,
        "used_teaor_fallback_false": used_teaor_fallback_false,
        "counts_by_county": dict(counts_by_county),
        "counts_by_workplace_county": dict(counts_by_workplace_county),
        "counts_by_teaor": dict(counts_by_teaor),
    }


# ============================================================
# 6. FEASIBILITY TÁBLÁK
# ============================================================

def foreign_counts_to_dataframe(counter: dict, count_column: str) -> pd.DataFrame:
    rows = []

    for key, count in counter.items():
        county_name, settlement_type, sex, age_group, education = key

        rows.append({
            "county_name": county_name,
            "settlement_type": settlement_type,
            "sex": sex,
            "age_group": age_group,
            "education": education,
            count_column: count,
        })

    return pd.DataFrame(rows)


def create_foreign_matching_feasibility(
    foreign_long: pd.DataFrame,
    current_employed_counts: dict,
) -> pd.DataFrame:
    foreign_targets = (
        foreign_long
        .groupby(
            [
                "county_name",
                "settlement_type",
                "sex",
                "age_group",
                "education",
            ],
            dropna=False,
        )["foreign_employed_count"]
        .sum()
        .reset_index()
        .rename(columns={"foreign_employed_count": "foreign_target_count"})
    )

    current_counts = foreign_counts_to_dataframe(
        current_employed_counts["counts_by_foreign_key"],
        "current_employed_count",
    )

    comparison = foreign_targets.merge(
        current_counts,
        on=[
            "county_name",
            "settlement_type",
            "sex",
            "age_group",
            "education",
        ],
        how="outer",
    )

    comparison["foreign_target_count"] = pd.to_numeric(
        comparison["foreign_target_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["current_employed_count"] = pd.to_numeric(
        comparison["current_employed_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["assignable_foreign_count_exact_key"] = comparison.apply(
        lambda row: min(row["foreign_target_count"], row["current_employed_count"]),
        axis=1,
    )

    comparison["foreign_target_surplus_over_current"] = comparison.apply(
        lambda row: max(row["foreign_target_count"] - row["current_employed_count"], 0),
        axis=1,
    )

    comparison["current_employed_surplus_after_foreign"] = comparison.apply(
        lambda row: max(row["current_employed_count"] - row["foreign_target_count"], 0),
        axis=1,
    )

    comparison["difference_current_minus_foreign_target"] = (
        comparison["current_employed_count"] - comparison["foreign_target_count"]
    )

    comparison["absolute_difference"] = comparison[
        "difference_current_minus_foreign_target"
    ].abs()

    comparison["exact_foreign_target_feasible"] = (
        comparison["foreign_target_surplus_over_current"] == 0
    )

    comparison = comparison.sort_values(
        [
            "foreign_target_surplus_over_current",
            "absolute_difference",
            "foreign_target_count",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return comparison


def create_domestic_worker_pool_usage_plan(
    current_employed_total: int,
    foreign_target_total: int,
    previous_worker_total: int,
) -> pd.DataFrame:
    domestic_target = current_employed_total - foreign_target_total

    rows = [
        {
            "metric": "current_employed_total",
            "value": current_employed_total,
            "note": "Jelenlegi activity-kalibrált foglalkoztatott agentek száma.",
        },
        {
            "metric": "foreign_workplace_target",
            "value": foreign_target_total,
            "note": "Külföldön foglalkoztatottak targetje a hier_kulfold_foglalkoztatott táblából.",
        },
        {
            "metric": "domestic_workplace_target",
            "value": domestic_target,
            "note": "current_employed_total - foreign_workplace_target.",
        },
        {
            "metric": "previous_domestic_worker_records_available",
            "value": previous_worker_total,
            "note": "Régi agents_with_workplaces.csv rekordok száma; belföldi worker/workplace assignment pool.",
        },
        {
            "metric": "previous_domestic_worker_records_needed",
            "value": domestic_target,
            "note": "Ennyi régi domestic worker rekordot kellene felhasználni, ha explicit foreign workplace réteget vezetünk be.",
        },
        {
            "metric": "previous_domestic_worker_records_unused",
            "value": previous_worker_total - domestic_target,
            "note": "Ennyi régi domestic worker rekord maradna közvetlenül fel nem használva.",
        },
        {
            "metric": "previous_domestic_pool_usage_share",
            "value": safe_divide(domestic_target, previous_worker_total),
            "note": "A régi domestic worker pool mekkora részét kellene felhasználni.",
        },
    ]

    return pd.DataFrame(rows)


def simple_counter_to_dataframe(counter: dict, key_name: str, count_name: str) -> pd.DataFrame:
    rows = []

    for key, count in counter.items():
        rows.append({
            key_name: key,
            count_name: count,
        })

    return pd.DataFrame(rows).sort_values(count_name, ascending=False).reset_index(drop=True)


# ============================================================
# 7. SUMMARY, REFERENCE, NOTES
# ============================================================

def write_reference_targets(
    current_employed_total: int,
    foreign_target_total: int,
    previous_worker_total: int,
) -> None:
    domestic_target = current_employed_total - foreign_target_total

    rows = [
        {
            "reference_name": "current_activity_calibrated_employed",
            "source_file": "64_agents_with_activity_education_activity_calibrated.csv",
            "category": "total_employed",
            "target_count": current_employed_total,
            "note": "Jelenlegi teljes populáció foglalkoztatott agentjei.",
        },
        {
            "reference_name": "foreign_workplace_target",
            "source_file": "hier_kulfold_foglalkoztatott.xlsx",
            "category": "Munkavégzés helye = Külföld",
            "target_count": foreign_target_total,
            "note": "Népszámlálási foreign workplace target.",
        },
        {
            "reference_name": "domestic_workplace_target",
            "source_file": "derived",
            "category": "domestic_workplace",
            "target_count": domestic_target,
            "note": "current_employed - foreign_workplace_target.",
        },
        {
            "reference_name": "previous_domestic_worker_pool",
            "source_file": "agents_with_workplaces.csv",
            "category": "previous_worker_records",
            "target_count": previous_worker_total,
            "note": "Korábbi domestic worker/workplace assignment pool.",
        },
    ]

    pd.DataFrame(rows).to_csv(
        REFERENCE_TARGETS_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def create_summary(
    current_employed_counts: dict,
    previous_worker_counts: dict,
    foreign_long: pd.DataFrame,
    foreign_feasibility: pd.DataFrame,
    domestic_plan: pd.DataFrame,
) -> pd.DataFrame:
    current_employed_total = current_employed_counts["employed_agents"]
    previous_worker_total = previous_worker_counts["total_rows"]
    foreign_target_total = foreign_long["foreign_employed_count"].sum()
    domestic_target = current_employed_total - foreign_target_total

    exact_assignable_foreign = foreign_feasibility["assignable_foreign_count_exact_key"].sum()
    foreign_unmatched_exact = foreign_feasibility["foreign_target_surplus_over_current"].sum()

    rows = [
        {
            "metric": "current_agents_total",
            "value": current_employed_counts["total_agents"],
            "note": "Jelenlegi 64-es agentfájl összes agentje.",
        },
        {
            "metric": "current_employed_total",
            "value": current_employed_total,
            "note": "Jelenlegi activity-kalibrált foglalkoztatott agentek száma.",
        },
        {
            "metric": "foreign_workplace_target_total",
            "value": foreign_target_total,
            "note": "Külföldön foglalkoztatottak targetje a hier_kulfold_foglalkoztatott táblából.",
        },
        {
            "metric": "domestic_workplace_target_total",
            "value": domestic_target,
            "note": "current_employed_total - foreign_workplace_target_total.",
        },
        {
            "metric": "previous_domestic_worker_records_total",
            "value": previous_worker_total,
            "note": "Korábbi agents_with_workplaces.csv rekordok száma.",
        },
        {
            "metric": "previous_domestic_worker_records_unused_if_foreign_explicit",
            "value": previous_worker_total - domestic_target,
            "note": "Ennyi régi worker record maradna fel nem használva, ha explicit foreign workplace réteget vezetünk be.",
        },
        {
            "metric": "previous_domestic_pool_usage_share",
            "value": safe_divide(domestic_target, previous_worker_total),
            "note": "Régi domestic worker pool felhasználási aránya.",
        },
        {
            "metric": "foreign_exact_key_assignable_total",
            "value": exact_assignable_foreign,
            "note": "Ennyi foreign target párosítható pontos county × settlement_type × sex × age_group × education kulcson.",
        },
        {
            "metric": "foreign_exact_key_assignable_share",
            "value": safe_divide(exact_assignable_foreign, foreign_target_total),
            "note": "Pontos kulcson foreignként kijelölhető arány.",
        },
        {
            "metric": "foreign_target_unmatched_on_exact_key",
            "value": foreign_unmatched_exact,
            "note": "Ennyi foreign targethez nincs elég current employed agent pontos kulcson; fallback kijelölés kellhet.",
        },
        {
            "metric": "foreign_target_unmatched_share_on_exact_key",
            "value": safe_divide(foreign_unmatched_exact, foreign_target_total),
            "note": "Pontos kulcson nem fedhető foreign target arány.",
        },
        {
            "metric": "previous_worker_missing_workplace_id",
            "value": previous_worker_counts["missing_workplace_id"],
            "note": "Régi worker rekordok hiányzó workplace_id-val. Ideálisan 0.",
        },
        {
            "metric": "previous_worker_used_fallback_true",
            "value": previous_worker_counts["used_fallback_true"],
            "note": "Régi pipeline-ban fallbackkel kiosztott worker rekordok.",
        },
        {
            "metric": "previous_worker_used_teaor_fallback_true",
            "value": previous_worker_counts["used_teaor_fallback_true"],
            "note": "Régi pipeline-ban TEÁOR fallbackkel kiosztott worker rekordok.",
        },
    ]

    return pd.DataFrame(rows)


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script nem módosít agenteket és nem rendel munkahelyet. Csak megtervezi az explicit foreign/domestic workplace assignment célokat.",
        },
        {
            "order": 2,
            "note": "A jelenlegi foglalkoztatotti total a 64-es activity-kalibrált agentfájlból jön.",
        },
        {
            "order": 3,
            "note": "A foreign workplace target a hier_kulfold_foglalkoztatott.xlsx táblából jön, ahol Munkavégzés helye = Külföld.",
        },
        {
            "order": 4,
            "note": "A domestic workplace target = current employed total - foreign workplace target.",
        },
        {
            "order": 5,
            "note": "A régi agents_with_workplaces.csv domestic worker/workplace assignment poolként értelmezendő.",
        },
        {
            "order": 6,
            "note": "Ha explicit foreign workplace réteget vezetünk be, akkor a régi domestic worker pool egy része közvetlenül fel nem használt maradhat.",
        },
        {
            "order": 7,
            "note": "A foreign agenteknél később külön jelölést javasolt használni: workplace_assignment_type=foreign_workplace, workplace_id=FOREIGN_WORKPLACE.",
        },
        {
            "order": 8,
            "note": "A domestic TEÁOR/workplace validációból a foreign_workplace agenteket külön kell kezelni, különben torzíthatják a magyarországi TEÁOR validációt.",
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

def run_foreign_domestic_workplace_plan() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Foreign employed hier tábla long formátumra alakítása...")
    foreign_long = read_foreign_employed_hier_long(
        input_xlsx=FOREIGN_EMPLOYED_HIER_XLSX
    )

    print("Current employed agentek számlálása...")
    current_employed_counts = count_current_employed_agents(
        current_agents_csv=CURRENT_AGENTS_CSV
    )

    print("Régi domestic worker pool számlálása...")
    previous_worker_counts = count_previous_worker_records(
        previous_workers_csv=PREVIOUS_WORKER_WORKPLACE_CSV
    )

    print("Foreign matching feasibility készítése...")
    foreign_feasibility = create_foreign_matching_feasibility(
        foreign_long=foreign_long,
        current_employed_counts=current_employed_counts,
    )

    current_employed_total = current_employed_counts["employed_agents"]
    foreign_target_total = foreign_long["foreign_employed_count"].sum()
    previous_worker_total = previous_worker_counts["total_rows"]

    print("Domestic worker pool usage plan készítése...")
    domestic_plan = create_domestic_worker_pool_usage_plan(
        current_employed_total=current_employed_total,
        foreign_target_total=foreign_target_total,
        previous_worker_total=previous_worker_total,
    )

    print("Reference targetek írása...")
    write_reference_targets(
        current_employed_total=current_employed_total,
        foreign_target_total=foreign_target_total,
        previous_worker_total=previous_worker_total,
    )

    print("Summary készítése...")
    summary = create_summary(
        current_employed_counts=current_employed_counts,
        previous_worker_counts=previous_worker_counts,
        foreign_long=foreign_long,
        foreign_feasibility=foreign_feasibility,
        domestic_plan=domestic_plan,
    )

    print("Method notes írása...")
    write_method_notes()

    print("Outputok mentése...")

    foreign_long.to_csv(
        FOREIGN_HIER_LONG_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    foreign_counts_to_dataframe(
        current_employed_counts["counts_by_foreign_key"],
        "current_employed_count",
    ).to_csv(
        CURRENT_EMPLOYED_COUNTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    foreign_feasibility.to_csv(
        FOREIGN_MATCHING_FEASIBILITY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    domestic_plan.to_csv(
        DOMESTIC_WORKER_POOL_PLAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        PLAN_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Current employed total: {current_employed_total:,}")
    print(f"Foreign workplace target: {foreign_target_total:,}")
    print(f"Domestic workplace target: {current_employed_total - foreign_target_total:,}")
    print(f"Previous domestic worker records: {previous_worker_total:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_foreign_domestic_workplace_plan()