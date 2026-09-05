from pathlib import Path
import csv
import re
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

AGENTS_WITH_EXACT_AGE_SEX_CSV = Path(
    "../outputs/household_generation/29_generated_agents_with_exact_age_sex_experimental.csv"
)

AGE_SEX_DISTRIBUTION_LONG_CSV = Path(
    "../outputs/household_generation/30_age_sex_distribution_long.csv"
)

FLAT_LONG_TABLE_CSV = Path(
    "../outputs/flat_validation/05_flat_combined_long_format_table.csv"
)

OUTPUT_FOLDER = Path("../outputs/household_generation")

HIER_AGE_SEX_COMPARISON_CSV = OUTPUT_FOLDER / "38_compare_generated_age_sex_vs_hier_age_distribution.csv"

HIER_AGE_SEX_SUMMARY_CSV = OUTPUT_FOLDER / "39_compare_generated_age_sex_vs_hier_summary.csv"

FLAT_GENDER_COMPARISON_CLEAN_CSV = OUTPUT_FOLDER / "40_flat_settlement_gender_vs_generated_agents_clean.csv"

FLAT_AGE_GROUP_COMPARISON_CLEAN_CSV = OUTPUT_FOLDER / "41_flat_settlement_age_group_vs_generated_agents_clean.csv"

VALIDATION_SUMMARY_CLEAN_CSV = OUTPUT_FOLDER / "42_exact_age_sex_validation_summary_clean.csv"


# Teszteléshez állíthatod például 100_000-re.
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
    'Főváros' -> 'Budapest'
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
    KSH nem címke vagy generated sex label egységesítése.
    """
    if pd.isna(raw_sex_label):
        return ""

    sex_label = str(raw_sex_label).strip().lower()

    if sex_label in ["férfi", "ferfi", "male"]:
        return "male"

    if sex_label in ["nő", "no", "female"]:
        return "female"

    return sex_label


def normalize_settlement_name(raw_name: str) -> str:
    """
    Ugyanaz a settlement_key logika, amit a korábbi flat feldolgozásnál használtunk.
    """
    if pd.isna(raw_name):
        return ""

    normalized = str(raw_name).strip().lower()

    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[\.,;:()\[\]/\\]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def map_exact_age_to_clean_10_year_age_group(exact_age: int) -> str:
    """
    Exact age → tiszta 10 éves flat korcsoport label.

    Ez direkt szigorú:
    0_9, 10_19, ..., 80_89, 90_plus
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


# ============================================================
# 3. FLAT KATEGÓRIA-FELISMERÉS — TISZTÍTOTT VERZIÓ
# ============================================================

def parse_clean_flat_10_year_age_group_category(category: str) -> tuple[int, int, str] | None:
    """
    Csak a tiszta korcsoportokat fogadja el.

    Elfogadott:
    - '10 évesnél fiatalabb' -> 0_9
    - '10–19 éves'
    - ...
    - '80–89 éves'
    - '90 éves és idősebb'

    Nem fogadja el:
    - '15–64 éves férfi'
    - '15–64 éves nő'
    - '65 éves és idősebb férfi'
    - '15 éves és idősebb nő 2 élve született gyermekkel'
    - '15 évesnél fiatalabb személy'
    """
    text = str(category).strip().lower()

    # KSH flat fájlban az első korcsoport így szerepel:
    # "10 évesnél fiatalabb"
    # Ez 0-9 éveseket jelent.
    if text == "10 évesnél fiatalabb":
        return 0, 9, "0_9"

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

        if (min_age, max_age) in accepted_ranges:
            return min_age, max_age, accepted_ranges[(min_age, max_age)]

        return None

    if text == "90 éves és idősebb":
        return 90, 200, "90_plus"

    return None


# ============================================================
# 4. GENERATED AGENTEK AGGREGÁLÁSA
# ============================================================

def count_generated_agents_for_validation(
    agents_csv: Path,
) -> dict:
    """
    Beolvassa a nagy 29-es agent fájlt, és többféle aggregált számlálót készít.

    Nem tölti memóriába az egész agent fájlt.
    """
    counts_by_hier_age_sex = defaultdict(int)
    counts_by_settlement = defaultdict(int)
    counts_by_settlement_sex = defaultdict(int)
    counts_by_settlement_age_group = defaultdict(int)

    total_agents_seen = 0

    with agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents_seen >= MAX_AGENTS_TO_PROCESS:
                break

            county_name = normalize_county_name(row["county_name"])
            settlement_type = normalize_settlement_type(row["settlement_type"])
            sex = normalize_sex_label(row["sex"])
            exact_age = convert_count_to_integer(row["exact_age"])

            settlement_key = row["settlement_key"]
            settlement_name = row["settlement_name"]

            clean_age_group_label = map_exact_age_to_clean_10_year_age_group(
                exact_age=exact_age
            )

            counts_by_hier_age_sex[
                (
                    county_name,
                    settlement_type,
                    sex,
                    exact_age,
                )
            ] += 1

            counts_by_settlement[
                (
                    settlement_key,
                    settlement_name,
                )
            ] += 1

            counts_by_settlement_sex[
                (
                    settlement_key,
                    settlement_name,
                    sex,
                )
            ] += 1

            counts_by_settlement_age_group[
                (
                    settlement_key,
                    settlement_name,
                    clean_age_group_label,
                )
            ] += 1

            total_agents_seen += 1

            if total_agents_seen % 1_000_000 == 0:
                print(f"Beolvasott generated agentek: {total_agents_seen:,}")

    return {
        "total_agents_seen": total_agents_seen,
        "counts_by_hier_age_sex": counts_by_hier_age_sex,
        "counts_by_settlement": counts_by_settlement,
        "counts_by_settlement_sex": counts_by_settlement_sex,
        "counts_by_settlement_age_group": counts_by_settlement_age_group,
    }


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


# ============================================================
# 5. HIER AGE DISTRIBUTION VALIDÁCIÓ
# ============================================================

def load_age_distribution_long(age_distribution_long_csv: Path) -> pd.DataFrame:
    """
    Beolvassa a 30_age_sex_distribution_long.csv fájlt.
    """
    age_distribution_long = pd.read_csv(age_distribution_long_csv)

    age_distribution_long["county_name"] = age_distribution_long["county_name"].apply(
        normalize_county_name
    )

    age_distribution_long["settlement_type"] = age_distribution_long["settlement_type"].apply(
        normalize_settlement_type
    )

    age_distribution_long["sex"] = age_distribution_long["sex"].apply(
        normalize_sex_label
    )

    age_distribution_long["exact_age"] = age_distribution_long["exact_age"].apply(
        convert_count_to_integer
    )

    age_distribution_long["value"] = age_distribution_long["value"].apply(
        convert_count_to_integer
    )

    return age_distribution_long


def create_hier_age_sex_comparison(
    age_distribution_long: pd.DataFrame,
    counts_by_hier_age_sex: dict,
) -> pd.DataFrame:
    """
    Generated agents visszamérése az age_distribution hierarchikus forrásra.

    Szint:
    county_name × settlement_type × sex × exact_age
    """
    hier_targets = (
        age_distribution_long
        .groupby(
            ["county_name", "settlement_type", "sex", "exact_age"],
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "hier_age_distribution_count"})
    )

    generated = convert_counter_to_dataframe(
        counter=counts_by_hier_age_sex,
        key_column_names=[
            "county_name",
            "settlement_type",
            "sex",
            "exact_age",
        ],
        value_column_name="generated_agent_count",
    )

    comparison = hier_targets.merge(
        generated,
        on=["county_name", "settlement_type", "sex", "exact_age"],
        how="outer",
    )

    comparison["hier_age_distribution_count"] = pd.to_numeric(
        comparison["hier_age_distribution_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["generated_agent_count"] = pd.to_numeric(
        comparison["generated_agent_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["difference_generated_minus_hier"] = (
        comparison["generated_agent_count"]
        - comparison["hier_age_distribution_count"]
    )

    comparison["absolute_difference"] = comparison[
        "difference_generated_minus_hier"
    ].abs()

    comparison["relative_abs_error"] = comparison.apply(
        lambda row: safe_divide(
            row["absolute_difference"],
            row["hier_age_distribution_count"],
        ),
        axis=1,
    )

    comparison = comparison.sort_values(
        ["absolute_difference", "hier_age_distribution_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return comparison


def create_hier_age_sex_summary(
    hier_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rövid összefoglaló a hier age/sex validációról.
    """
    rows = []

    hier_total = hier_comparison["hier_age_distribution_count"].sum()
    generated_total = hier_comparison["generated_agent_count"].sum()
    absolute_difference_total = hier_comparison["absolute_difference"].sum()

    rows.append({
        "metric": "hier_age_distribution_total",
        "value": hier_total,
        "note": "Age_distribution forrás összesített népessége.",
    })

    rows.append({
        "metric": "generated_agent_total_in_hier_comparison",
        "value": generated_total,
        "note": "Generated agentek összesen a hier age/sex összevetésben.",
    })

    rows.append({
        "metric": "difference_generated_minus_hier_total",
        "value": generated_total - hier_total,
        "note": "Generated összeg mínusz hier forrás összeg.",
    })

    rows.append({
        "metric": "sum_absolute_cell_difference",
        "value": absolute_difference_total,
        "note": "Cellaszintű abszolút eltérések összege county × settlement_type × sex × exact_age szinten.",
    })

    rows.append({
        "metric": "mean_absolute_cell_difference",
        "value": hier_comparison["absolute_difference"].mean(),
        "note": "Átlagos cellaszintű abszolút eltérés.",
    })

    rows.append({
        "metric": "max_absolute_cell_difference",
        "value": hier_comparison["absolute_difference"].max(),
        "note": "Legnagyobb cellaszintű abszolút eltérés.",
    })

    rows.append({
        "metric": "nonzero_difference_rows",
        "value": (
            hier_comparison["difference_generated_minus_hier"] != 0
        ).sum(),
        "note": "Olyan hier age/sex cellák száma, ahol van eltérés.",
    })

    summary = pd.DataFrame(rows)

    return summary


# ============================================================
# 6. FLAT TARGETEK BETÖLTÉSE ÉS TISZTÍTOTT VALIDÁCIÓ
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
    csak Férfi / Nő sorok.
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


def extract_clean_flat_settlement_age_group_targets(
    flat_long_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flat településszintű, tiszta 10 éves korcsoport targetek.

    Csak ezt engedjük be:
    0–9, 10–19, ..., 80–89, 90+
    """
    candidate_rows = flat_long_table[
        flat_long_table["source_file"].str.contains("nepesseg", case=False, regex=False)
    ].copy()

    parsed_rows = []

    for _, row in candidate_rows.iterrows():
        parsed = parse_clean_flat_10_year_age_group_category(row["category"])

        if parsed is None:
            continue

        min_age, max_age, age_group_label = parsed

        parsed_rows.append({
            "settlement_key": row["settlement_key"],
            "settlement_name_flat": row["settlement"],
            "clean_age_group_label": age_group_label,
            "clean_age_group_min_age": min_age,
            "clean_age_group_max_age": max_age,
            "flat_age_group_count": convert_count_to_integer(row["value"]),
            "source_category": row["category"],
        })

    targets = pd.DataFrame(parsed_rows)

    return targets


def create_flat_population_comparison(
    flat_population_targets: pd.DataFrame,
    counts_by_settlement: dict,
) -> pd.DataFrame:
    generated = convert_counter_to_dataframe(
        counter=counts_by_settlement,
        key_column_names=["settlement_key", "settlement_name_generated"],
        value_column_name="generated_agent_count",
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
            row["absolute_difference"],
            row["flat_population_total"],
        ),
        axis=1,
    )

    return comparison


def create_flat_gender_comparison(
    flat_gender_targets: pd.DataFrame,
    counts_by_settlement_sex: dict,
) -> pd.DataFrame:
    generated = convert_counter_to_dataframe(
        counter=counts_by_settlement_sex,
        key_column_names=[
            "settlement_key",
            "settlement_name_generated",
            "sex",
        ],
        value_column_name="generated_sex_count",
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
            row["absolute_difference"],
            row["flat_gender_count"],
        ),
        axis=1,
    )

    comparison = comparison.sort_values(
        ["absolute_difference", "flat_gender_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return comparison


def create_flat_age_group_comparison(
    flat_age_group_targets: pd.DataFrame,
    counts_by_settlement_age_group: dict,
) -> pd.DataFrame:
    generated = convert_counter_to_dataframe(
        counter=counts_by_settlement_age_group,
        key_column_names=[
            "settlement_key",
            "settlement_name_generated",
            "clean_age_group_label",
        ],
        value_column_name="generated_age_group_count",
    )

    comparison = flat_age_group_targets.merge(
        generated,
        on=["settlement_key", "clean_age_group_label"],
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
            row["absolute_difference"],
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
# 7. ÖSSZEFOGLALÓ
# ============================================================

def create_clean_validation_summary(
    total_agents_seen: int,
    hier_summary: pd.DataFrame,
    flat_population_comparison: pd.DataFrame,
    flat_gender_comparison: pd.DataFrame,
    flat_age_group_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def get_hier_metric(metric_name: str):
        value_series = hier_summary.loc[
            hier_summary["metric"] == metric_name,
            "value",
        ]

        if len(value_series) == 0:
            return None

        return value_series.iloc[0]

    rows.append({
        "metric": "generated_agents_total",
        "value": total_agents_seen,
        "note": "Beolvasott generated agentek száma a 29-es fájlból.",
    })

    rows.append({
        "metric": "hier_age_distribution_total",
        "value": get_hier_metric("hier_age_distribution_total"),
        "note": "Age_distribution hierarchikus forrás összesített népessége.",
    })

    rows.append({
        "metric": "hier_generated_minus_source_total",
        "value": get_hier_metric("difference_generated_minus_hier_total"),
        "note": "Generated agent összeg mínusz age_distribution forrásösszeg.",
    })

    rows.append({
        "metric": "hier_sum_absolute_cell_difference",
        "value": get_hier_metric("sum_absolute_cell_difference"),
        "note": "Hier age/sex cellaszintű abszolút eltérések összege.",
    })

    rows.append({
        "metric": "hier_max_absolute_cell_difference",
        "value": get_hier_metric("max_absolute_cell_difference"),
        "note": "Legnagyobb county × settlement_type × sex × exact_age cellaeltérés.",
    })

    rows.append({
        "metric": "flat_population_total",
        "value": flat_population_comparison["flat_population_total"].sum(),
        "note": "Flat Férfi+Nő alapján településszintű teljes népesség.",
    })

    rows.append({
        "metric": "flat_population_generated_total",
        "value": flat_population_comparison["generated_agent_count"].sum(),
        "note": "Generated agentek száma településszintű flat összevetésben.",
    })

    rows.append({
        "metric": "flat_population_generated_minus_flat_total",
        "value": (
            flat_population_comparison["generated_agent_count"].sum()
            - flat_population_comparison["flat_population_total"].sum()
        ),
        "note": "Várhatóan -6057 körül, a residual miatt.",
    })

    rows.append({
        "metric": "flat_population_max_settlement_absolute_difference",
        "value": flat_population_comparison["absolute_difference"].max(),
        "note": "Legnagyobb településszintű teljes népességeltérés.",
    })

    rows.append({
        "metric": "flat_gender_total_target",
        "value": flat_gender_comparison["flat_gender_count"].sum(),
        "note": "Flat Férfi/Nő target összesen.",
    })

    rows.append({
        "metric": "flat_gender_generated_total",
        "value": flat_gender_comparison["generated_sex_count"].sum(),
        "note": "Generated sex count összesen.",
    })

    rows.append({
        "metric": "flat_gender_generated_minus_flat_total",
        "value": (
            flat_gender_comparison["generated_sex_count"].sum()
            - flat_gender_comparison["flat_gender_count"].sum()
        ),
        "note": "Országos sex target eltérés. Várhatóan a residual miatt kb. -6057.",
    })

    rows.append({
        "metric": "flat_gender_sum_absolute_cell_difference",
        "value": flat_gender_comparison["absolute_difference"].sum(),
        "note": "Település × sex cellák abszolút eltéréseinek összege.",
    })

    rows.append({
        "metric": "flat_gender_max_absolute_cell_difference",
        "value": flat_gender_comparison["absolute_difference"].max(),
        "note": "Legnagyobb település × sex eltérés.",
    })

    rows.append({
        "metric": "flat_age_group_total_target_clean",
        "value": flat_age_group_comparison["flat_age_group_count"].sum(),
        "note": "Tisztított flat 10 éves korcsoport target összesen.",
    })

    rows.append({
        "metric": "flat_age_group_generated_total_clean",
        "value": flat_age_group_comparison["generated_age_group_count"].sum(),
        "note": "Generated exact_age visszaaggregálva tiszta 10 éves korcsoportokra.",
    })

    rows.append({
        "metric": "flat_age_group_generated_minus_flat_total_clean",
        "value": (
            flat_age_group_comparison["generated_age_group_count"].sum()
            - flat_age_group_comparison["flat_age_group_count"].sum()
        ),
        "note": "Most már nem tartalmazhat átfedő nő/férfi/családi kategóriákat.",
    })

    rows.append({
        "metric": "flat_age_group_sum_absolute_cell_difference_clean",
        "value": flat_age_group_comparison["absolute_difference"].sum(),
        "note": "Település × tiszta 10 éves korcsoport abszolút eltérések összege.",
    })

    rows.append({
        "metric": "flat_age_group_max_absolute_cell_difference_clean",
        "value": flat_age_group_comparison["absolute_difference"].max(),
        "note": "Legnagyobb település × tiszta 10 éves korcsoport eltérés.",
    })

    summary = pd.DataFrame(rows)

    return summary


# ============================================================
# 8. FŐ FUTTATÁS
# ============================================================

def run_clean_exact_age_sex_validation() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Generated exact age/sex agentek aggregálása...")
    generated_counts = count_generated_agents_for_validation(
        agents_csv=AGENTS_WITH_EXACT_AGE_SEX_CSV
    )

    print("Age distribution long tábla beolvasása...")
    age_distribution_long = load_age_distribution_long(
        AGE_SEX_DISTRIBUTION_LONG_CSV
    )

    print("Hier age/sex összevetés készítése...")
    hier_comparison = create_hier_age_sex_comparison(
        age_distribution_long=age_distribution_long,
        counts_by_hier_age_sex=generated_counts["counts_by_hier_age_sex"],
    )

    print("Hier age/sex summary készítése...")
    hier_summary = create_hier_age_sex_summary(
        hier_comparison=hier_comparison
    )

    print("Flat long tábla beolvasása...")
    flat_long_table = load_flat_long_table(
        FLAT_LONG_TABLE_CSV
    )

    print("Flat population targetek kinyerése...")
    flat_population_targets = extract_flat_settlement_population_targets(
        flat_long_table=flat_long_table
    )

    print("Flat gender targetek kinyerése...")
    flat_gender_targets = extract_flat_settlement_gender_targets(
        flat_long_table=flat_long_table
    )

    print("Tisztított flat 10 éves korcsoport targetek kinyerése...")
    flat_age_group_targets = extract_clean_flat_settlement_age_group_targets(
        flat_long_table=flat_long_table
    )

    print("Tisztított flat korcsoport targetek országos összege:")
    print(
        flat_age_group_targets
        .groupby("clean_age_group_label")["flat_age_group_count"]
        .sum()
        .reset_index()
        .sort_values("clean_age_group_label")
    )

    print("Flat population összevetés...")
    flat_population_comparison = create_flat_population_comparison(
        flat_population_targets=flat_population_targets,
        counts_by_settlement=generated_counts["counts_by_settlement"],
    )

    print("Flat gender összevetés...")
    flat_gender_comparison = create_flat_gender_comparison(
        flat_gender_targets=flat_gender_targets,
        counts_by_settlement_sex=generated_counts["counts_by_settlement_sex"],
    )

    print("Flat age-group összevetés tisztított kategóriákkal...")
    flat_age_group_comparison = create_flat_age_group_comparison(
        flat_age_group_targets=flat_age_group_targets,
        counts_by_settlement_age_group=generated_counts["counts_by_settlement_age_group"],
    )

    print("Összefoglaló készítése...")
    clean_summary = create_clean_validation_summary(
        total_agents_seen=generated_counts["total_agents_seen"],
        hier_summary=hier_summary,
        flat_population_comparison=flat_population_comparison,
        flat_gender_comparison=flat_gender_comparison,
        flat_age_group_comparison=flat_age_group_comparison,
    )

    print("Outputok mentése...")

    hier_comparison.to_csv(
        HIER_AGE_SEX_COMPARISON_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    hier_summary.to_csv(
        HIER_AGE_SEX_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    flat_gender_comparison.to_csv(
        FLAT_GENDER_COMPARISON_CLEAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    flat_age_group_comparison.to_csv(
        FLAT_AGE_GROUP_COMPARISON_CLEAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    clean_summary.to_csv(
        VALIDATION_SUMMARY_CLEAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Beolvasott generated agentek: {generated_counts['total_agents_seen']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


# ============================================================
# 9. PROGRAM INDÍTÁSA
# ============================================================

if __name__ == "__main__":
    run_clean_exact_age_sex_validation()