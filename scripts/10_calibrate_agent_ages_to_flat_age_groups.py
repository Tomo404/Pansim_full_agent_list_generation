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

AGENTS_WITH_EXACT_AGE_SEX_CSV = Path(
    "../outputs/household_generation/29_generated_agents_with_exact_age_sex_experimental.csv"
)

FLAT_LONG_TABLE_CSV = Path(
    "../outputs/flat_validation/05_flat_combined_long_format_table.csv"
)

AGE_SEX_DISTRIBUTION_LONG_CSV = Path(
    "../outputs/household_generation/30_age_sex_distribution_long.csv"
)

OUTPUT_FOLDER = Path("../outputs/household_generation")

AGE_CALIBRATED_AGENTS_CSV = OUTPUT_FOLDER / "43_generated_agents_with_exact_age_sex_age_calibrated.csv"

AGE_GROUP_CALIBRATION_PLAN_CSV = OUTPUT_FOLDER / "44_age_group_calibration_plan_by_settlement.csv"

AGE_GROUP_CALIBRATION_TRANSFER_CSV = OUTPUT_FOLDER / "45_age_group_calibration_transfers_by_settlement.csv"

POST_FLAT_AGE_GROUP_VALIDATION_CSV = OUTPUT_FOLDER / "46_post_age_calibration_flat_age_group_validation.csv"

POST_HIER_AGE_SEX_VALIDATION_CSV = OUTPUT_FOLDER / "47_post_age_calibration_hier_age_sex_validation.csv"

POST_AGE_CALIBRATION_SUMMARY_CSV = OUTPUT_FOLDER / "48_post_age_calibration_summary.csv"


RANDOM_SEED = 42

# Teszteléshez állíthatod például 100_000-re.
# Teljes futtatáshoz legyen None.
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
# 2. ÁLTALÁNOS SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def convert_count_to_integer(raw_value) -> int:
    """
    Biztonságos darabszám-konverzió.

    Kezeli:
    - üres érték
    - NaN
    - normál integer / float
    - esetleges magyar ezrespont formátumot, például '1.896'
    """
    if raw_value is None:
        return 0

    if pd.isna(raw_value):
        return 0

    text_value = str(raw_value).strip()

    if text_value == "":
        return 0

    # Ha úgy néz ki, mint magyar ezres elválasztás, például 1.896 vagy 12.047
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
    if pd.isna(raw_sex_label):
        return ""

    sex_label = str(raw_sex_label).strip().lower()

    if sex_label in ["férfi", "ferfi", "male"]:
        return "male"

    if sex_label in ["nő", "no", "female"]:
        return "female"

    return sex_label


def normalize_settlement_name(raw_name: str) -> str:
    if pd.isna(raw_name):
        return ""

    normalized = str(raw_name).strip().lower()

    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[\.,;:()\[\]/\\]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def map_exact_age_to_broad_age_group(exact_age: int) -> str:
    age = int(exact_age)

    if age < 30:
        return "under_30"

    if age <= 64:
        return "age_30_64"

    return "age_65_plus"


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


def get_age_range_for_clean_age_group(clean_age_group_label: str) -> tuple[int, int]:
    if clean_age_group_label == "0_9":
        return 0, 9

    if clean_age_group_label == "10_19":
        return 10, 19

    if clean_age_group_label == "20_29":
        return 20, 29

    if clean_age_group_label == "30_39":
        return 30, 39

    if clean_age_group_label == "40_49":
        return 40, 49

    if clean_age_group_label == "50_59":
        return 50, 59

    if clean_age_group_label == "60_69":
        return 60, 69

    if clean_age_group_label == "70_79":
        return 70, 79

    if clean_age_group_label == "80_89":
        return 80, 89

    if clean_age_group_label == "90_plus":
        return 90, 99

    raise ValueError(f"Ismeretlen clean age group: {clean_age_group_label}")


# ============================================================
# 3. FLAT TISZTA 10 ÉVES KORCSOPORT TARGETEK
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
    """
    text = str(category).strip().lower()

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


def load_flat_long_table(flat_long_table_csv: Path) -> pd.DataFrame:
    flat_long_table = pd.read_csv(flat_long_table_csv)

    flat_long_table["settlement_key"] = flat_long_table["settlement"].apply(
        normalize_settlement_name
    )

    flat_long_table["value"] = flat_long_table["value"].apply(
        convert_count_to_integer
    )

    return flat_long_table


def extract_clean_flat_settlement_age_group_targets(
    flat_long_table: pd.DataFrame,
) -> pd.DataFrame:
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


# ============================================================
# 4. AGE DISTRIBUTION EXACT AGE SÚLYOK
# ============================================================

def load_age_distribution_long(age_distribution_long_csv: Path) -> pd.DataFrame:
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

    age_distribution_long["clean_age_group_label"] = age_distribution_long["exact_age"].apply(
        map_exact_age_to_clean_10_year_age_group
    )

    return age_distribution_long


def build_exact_age_weight_lookup(
    age_distribution_long: pd.DataFrame,
) -> dict:
    """
    Készít egy lookupot az exact age választáshoz.

    Kulcs:
    county_name × settlement_type × sex × clean_age_group_label

    Érték:
    exact_age lista és súlylista
    """
    lookup = {}

    grouped = (
        age_distribution_long
        .groupby(
            [
                "county_name",
                "settlement_type",
                "sex",
                "clean_age_group_label",
                "exact_age",
            ],
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
    )

    for key_columns, group in grouped.groupby(
        ["county_name", "settlement_type", "sex", "clean_age_group_label"],
        dropna=False,
    ):
        county_name, settlement_type, sex, clean_age_group_label = key_columns

        ages = group["exact_age"].astype(int).tolist()
        weights = group["value"].astype(int).tolist()

        if sum(weights) <= 0:
            min_age, max_age = get_age_range_for_clean_age_group(clean_age_group_label)
            ages = list(range(min_age, max_age + 1))
            weights = [1] * len(ages)

        lookup[
            (
                county_name,
                settlement_type,
                sex,
                clean_age_group_label,
            )
        ] = {
            "ages": ages,
            "weights": weights,
        }

    return lookup


def choose_exact_age_for_target_group(
    exact_age_weight_lookup: dict,
    county_name: str,
    settlement_type: str,
    sex: str,
    clean_age_group_label: str,
    random_generator: random.Random,
) -> int:
    key = (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_sex_label(sex),
        clean_age_group_label,
    )

    if key in exact_age_weight_lookup:
        age_data = exact_age_weight_lookup[key]

        return random_generator.choices(
            population=age_data["ages"],
            weights=age_data["weights"],
            k=1,
        )[0]

    # Fallback: egyenletes exact age választás az adott 10 éves csoporton belül.
    min_age, max_age = get_age_range_for_clean_age_group(clean_age_group_label)

    return random_generator.randint(min_age, max_age)


# ============================================================
# 5. GENERATED AGENTEK ELSŐ PASS: AKTUÁLIS KORCSOPORTOK
# ============================================================

def count_current_generated_age_groups(
    agents_csv: Path,
) -> dict:
    """
    Első pass a 29-es agent fájlon.

    Kiszámolja:
    - settlement total
    - settlement × clean age group current generated count
    - settlement metaadatok
    """
    counts_by_settlement_age_group = defaultdict(int)
    counts_by_settlement = defaultdict(int)
    settlement_metadata = {}

    total_agents_seen = 0

    with agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents_seen >= MAX_AGENTS_TO_PROCESS:
                break

            settlement_key = row["settlement_key"]
            settlement_name = row["settlement_name"]
            county_name = normalize_county_name(row["county_name"])
            settlement_type = normalize_settlement_type(row["settlement_type"])

            exact_age = convert_count_to_integer(row["exact_age"])
            clean_age_group_label = map_exact_age_to_clean_10_year_age_group(
                exact_age=exact_age
            )

            counts_by_settlement[
                settlement_key
            ] += 1

            counts_by_settlement_age_group[
                (
                    settlement_key,
                    clean_age_group_label,
                )
            ] += 1

            if settlement_key not in settlement_metadata:
                settlement_metadata[settlement_key] = {
                    "settlement_key": settlement_key,
                    "settlement_name_generated": settlement_name,
                    "county_name": county_name,
                    "settlement_type": settlement_type,
                }

            total_agents_seen += 1

            if total_agents_seen % 1_000_000 == 0:
                print(f"Első pass beolvasott agentek: {total_agents_seen:,}")

    return {
        "total_agents_seen": total_agents_seen,
        "counts_by_settlement": counts_by_settlement,
        "counts_by_settlement_age_group": counts_by_settlement_age_group,
        "settlement_metadata": settlement_metadata,
    }


# ============================================================
# 6. INTEGER TARGET SKÁLÁZÁS ÉS KALIBRÁCIÓS TERV
# ============================================================

def allocate_integer_targets_by_largest_remainder(
    raw_counts_by_age_group: dict,
    total_count_to_allocate: int,
) -> dict:
    """
    Egy settlement flat korcsoportjait skálázza a generated settlement totalra.

    Példa:
    flat korcsoport total = 1000
    generated settlement total = 980

    Akkor minden flat korcsoportot arányosan 980-ra skálázunk,
    largest remainder kerekítéssel.
    """
    raw_total = sum(raw_counts_by_age_group.values())

    if total_count_to_allocate <= 0:
        return {
            age_group: 0
            for age_group in CLEAN_AGE_GROUP_ORDER
        }

    if raw_total <= 0:
        return None

    rows = []

    for age_group in CLEAN_AGE_GROUP_ORDER:
        raw_count = raw_counts_by_age_group.get(age_group, 0)
        expected_count = safe_divide(raw_count * total_count_to_allocate, raw_total)
        floor_count = math.floor(expected_count)
        remainder = expected_count - floor_count

        rows.append({
            "clean_age_group_label": age_group,
            "raw_count": raw_count,
            "expected_count": expected_count,
            "allocated_count": floor_count,
            "remainder": remainder,
        })

    allocated_total = sum(row["allocated_count"] for row in rows)
    remaining = total_count_to_allocate - allocated_total

    rows = sorted(
        rows,
        key=lambda row: (row["remainder"], row["expected_count"]),
        reverse=True,
    )

    for index in range(remaining):
        rows[index]["allocated_count"] += 1

    return {
        row["clean_age_group_label"]: row["allocated_count"]
        for row in rows
    }


def build_flat_age_targets_lookup(
    flat_age_group_targets: pd.DataFrame,
) -> dict:
    """
    Flat age group targets settlementenként.
    """
    lookup = defaultdict(dict)

    for _, row in flat_age_group_targets.iterrows():
        settlement_key = row["settlement_key"]
        age_group = row["clean_age_group_label"]
        flat_count = convert_count_to_integer(row["flat_age_group_count"])

        lookup[settlement_key][age_group] = flat_count

    return lookup


def create_age_group_calibration_plan(
    current_counts: dict,
    flat_age_group_targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Település × age group szintű calibration plan.

    Fontos:
    A flat targetet settlementen belül a generated settlement totalra skálázzuk.
    """
    counts_by_settlement = current_counts["counts_by_settlement"]
    counts_by_settlement_age_group = current_counts["counts_by_settlement_age_group"]
    settlement_metadata = current_counts["settlement_metadata"]

    flat_target_lookup = build_flat_age_targets_lookup(
        flat_age_group_targets=flat_age_group_targets
    )

    plan_rows = []

    for settlement_key, generated_settlement_total in counts_by_settlement.items():
        metadata = settlement_metadata.get(
            settlement_key,
            {
                "settlement_key": settlement_key,
                "settlement_name_generated": "",
                "county_name": "",
                "settlement_type": "",
            },
        )

        raw_flat_targets = flat_target_lookup.get(settlement_key, {})

        scaled_targets = allocate_integer_targets_by_largest_remainder(
            raw_counts_by_age_group=raw_flat_targets,
            total_count_to_allocate=generated_settlement_total,
        )

        if scaled_targets is None:
            scaled_targets = {
                age_group: counts_by_settlement_age_group.get(
                    (
                        settlement_key,
                        age_group,
                    ),
                    0,
                )
                for age_group in CLEAN_AGE_GROUP_ORDER
            }

        for age_group in CLEAN_AGE_GROUP_ORDER:
            current_count = counts_by_settlement_age_group.get(
                (
                    settlement_key,
                    age_group,
                ),
                0,
            )

            flat_raw_target = raw_flat_targets.get(age_group, 0)
            scaled_target = scaled_targets.get(age_group, 0)

            difference_current_minus_scaled_target = current_count - scaled_target

            plan_rows.append({
                "settlement_key": settlement_key,
                "settlement_name_generated": metadata["settlement_name_generated"],
                "county_name": metadata["county_name"],
                "settlement_type": metadata["settlement_type"],
                "clean_age_group_label": age_group,
                "generated_settlement_total": generated_settlement_total,
                "flat_raw_age_group_target": flat_raw_target,
                "flat_raw_age_group_total_for_settlement": sum(raw_flat_targets.values()),
                "scaled_age_group_target_for_generated_total": scaled_target,
                "current_generated_age_group_count": current_count,
                "difference_current_minus_scaled_target": difference_current_minus_scaled_target,
                "surplus_to_move_out": max(difference_current_minus_scaled_target, 0),
                "deficit_to_fill": max(-difference_current_minus_scaled_target, 0),
            })

    plan = pd.DataFrame(plan_rows)

    return plan


def create_transfer_plan_from_calibration_plan(
    calibration_plan: pd.DataFrame,
) -> pd.DataFrame:
    """
    Settlementen belüli age group transfer terv.

    Példa:
    settlement X:
    30_39 surplus 100
    0_9 deficit 60
    10_19 deficit 40

    Transfer:
    30_39 -> 0_9: 60
    30_39 -> 10_19: 40
    """
    transfer_rows = []

    for settlement_key, settlement_plan in calibration_plan.groupby("settlement_key", dropna=False):
        metadata_row = settlement_plan.iloc[0]

        surplus_items = []
        deficit_items = []

        for _, row in settlement_plan.iterrows():
            age_group = row["clean_age_group_label"]
            surplus_count = convert_count_to_integer(row["surplus_to_move_out"])
            deficit_count = convert_count_to_integer(row["deficit_to_fill"])

            if surplus_count > 0:
                surplus_items.append({
                    "from_age_group": age_group,
                    "remaining_surplus": surplus_count,
                })

            if deficit_count > 0:
                deficit_items.append({
                    "to_age_group": age_group,
                    "remaining_deficit": deficit_count,
                })

        surplus_index = 0
        deficit_index = 0

        while surplus_index < len(surplus_items) and deficit_index < len(deficit_items):
            surplus_item = surplus_items[surplus_index]
            deficit_item = deficit_items[deficit_index]

            transfer_count = min(
                surplus_item["remaining_surplus"],
                deficit_item["remaining_deficit"],
            )

            if transfer_count > 0:
                transfer_rows.append({
                    "settlement_key": settlement_key,
                    "settlement_name_generated": metadata_row["settlement_name_generated"],
                    "county_name": metadata_row["county_name"],
                    "settlement_type": metadata_row["settlement_type"],
                    "from_age_group": surplus_item["from_age_group"],
                    "to_age_group": deficit_item["to_age_group"],
                    "transfer_count": transfer_count,
                })

                surplus_item["remaining_surplus"] -= transfer_count
                deficit_item["remaining_deficit"] -= transfer_count

            if surplus_item["remaining_surplus"] == 0:
                surplus_index += 1

            if deficit_item["remaining_deficit"] == 0:
                deficit_index += 1

    transfer_plan = pd.DataFrame(transfer_rows)

    return transfer_plan


def build_transfer_queues(
    transfer_plan: pd.DataFrame,
) -> dict:
    """
    Második passhoz transfer queue-k.

    Kulcs:
    settlement_key × from_age_group

    Érték:
    megkevert lista to_age_group címkékkel.
    """
    queues = defaultdict(list)

    random_generator = random.Random(RANDOM_SEED)

    for _, row in transfer_plan.iterrows():
        key = (
            row["settlement_key"],
            row["from_age_group"],
        )

        transfer_count = convert_count_to_integer(row["transfer_count"])

        queues[key].extend(
            [row["to_age_group"]] * transfer_count
        )

    for key in queues:
        random_generator.shuffle(queues[key])

    return queues


# ============================================================
# 7. MÁSODIK PASS: AGENTEK KOR-KALIBRÁLÁSA
# ============================================================

def calibrate_agent_ages_to_transfer_plan(
    input_agents_csv: Path,
    output_agents_csv: Path,
    current_counts: dict,
    transfer_queues: dict,
    exact_age_weight_lookup: dict,
) -> dict:
    """
    Második pass a 29-es agentfájlon.

    Ha egy agent olyan settlement × age_group csoportban van,
    ahonnan ki kell mozgatni embereket, akkor véletlenszerűen kiválasztjuk,
    és átteszük egy target korcsoportba.
    """
    random_generator = random.Random(RANDOM_SEED)

    remaining_source_rows = dict(current_counts["counts_by_settlement_age_group"])

    remaining_changes_by_source = defaultdict(int)

    for key, queue in transfer_queues.items():
        remaining_changes_by_source[key] = len(queue)

    post_counts_by_settlement_age_group = defaultdict(int)
    post_counts_by_hier_age_sex = defaultdict(int)

    total_agents_written = 0
    total_age_changed_agents = 0
    total_broad_age_group_changed_agents = 0

    with input_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)

        input_fieldnames = list(reader.fieldnames)

        extra_columns = [
            "age_calibrated",
            "original_exact_age_before_age_calibration",
            "original_broad_age_group_before_age_calibration",
            "original_clean_age_group_before_age_calibration",
            "calibrated_clean_age_group",
        ]

        output_fieldnames = input_fieldnames + [
            column
            for column in extra_columns
            if column not in input_fieldnames
        ]

        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents_written >= MAX_AGENTS_TO_PROCESS:
                break

            settlement_key = row["settlement_key"]
            county_name = normalize_county_name(row["county_name"])
            settlement_type = normalize_settlement_type(row["settlement_type"])
            sex = normalize_sex_label(row["sex"])

            original_exact_age = convert_count_to_integer(row["exact_age"])
            original_broad_age_group = row["broad_age_group"]
            original_clean_age_group = map_exact_age_to_clean_10_year_age_group(
                original_exact_age
            )

            source_key = (
                settlement_key,
                original_clean_age_group,
            )

            rows_left_in_source_group = remaining_source_rows.get(source_key, 0)
            changes_left_in_source_group = remaining_changes_by_source.get(source_key, 0)

            should_change_age = False
            target_clean_age_group = original_clean_age_group

            if rows_left_in_source_group > 0 and changes_left_in_source_group > 0:
                change_probability = changes_left_in_source_group / rows_left_in_source_group

                if random_generator.random() < change_probability:
                    should_change_age = True
                    target_clean_age_group = transfer_queues[source_key].pop()
                    remaining_changes_by_source[source_key] -= 1

            remaining_source_rows[source_key] = rows_left_in_source_group - 1

            if should_change_age:
                new_exact_age = choose_exact_age_for_target_group(
                    exact_age_weight_lookup=exact_age_weight_lookup,
                    county_name=county_name,
                    settlement_type=settlement_type,
                    sex=sex,
                    clean_age_group_label=target_clean_age_group,
                    random_generator=random_generator,
                )

                new_broad_age_group = map_exact_age_to_broad_age_group(
                    new_exact_age
                )

                row["exact_age"] = new_exact_age
                row["broad_age_group"] = new_broad_age_group

                total_age_changed_agents += 1

                if new_broad_age_group != original_broad_age_group:
                    total_broad_age_group_changed_agents += 1

                age_calibrated = "True"

            else:
                new_exact_age = original_exact_age
                new_broad_age_group = original_broad_age_group
                target_clean_age_group = original_clean_age_group
                age_calibrated = "False"

            row["county_name"] = county_name
            row["settlement_type"] = settlement_type
            row["sex"] = sex

            row["age_calibrated"] = age_calibrated
            row["original_exact_age_before_age_calibration"] = original_exact_age
            row["original_broad_age_group_before_age_calibration"] = original_broad_age_group
            row["original_clean_age_group_before_age_calibration"] = original_clean_age_group
            row["calibrated_clean_age_group"] = target_clean_age_group

            post_counts_by_settlement_age_group[
                (
                    settlement_key,
                    target_clean_age_group,
                )
            ] += 1

            post_counts_by_hier_age_sex[
                (
                    county_name,
                    settlement_type,
                    sex,
                    int(new_exact_age),
                )
            ] += 1

            writer.writerow(row)

            total_agents_written += 1

            if total_agents_written % 1_000_000 == 0:
                print(f"Második pass kiírt agentek: {total_agents_written:,}")

    remaining_unapplied_changes = sum(
        remaining_changes_by_source.values()
    )

    return {
        "total_agents_written": total_agents_written,
        "total_age_changed_agents": total_age_changed_agents,
        "total_broad_age_group_changed_agents": total_broad_age_group_changed_agents,
        "remaining_unapplied_changes": remaining_unapplied_changes,
        "post_counts_by_settlement_age_group": post_counts_by_settlement_age_group,
        "post_counts_by_hier_age_sex": post_counts_by_hier_age_sex,
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


def create_post_flat_age_group_validation(
    calibration_plan: pd.DataFrame,
    post_counts_by_settlement_age_group: dict,
) -> pd.DataFrame:
    post_generated = convert_counter_to_dataframe(
        counter=post_counts_by_settlement_age_group,
        key_column_names=[
            "settlement_key",
            "clean_age_group_label",
        ],
        value_column_name="post_calibrated_age_group_count",
    )

    validation = calibration_plan.merge(
        post_generated,
        on=[
            "settlement_key",
            "clean_age_group_label",
        ],
        how="outer",
    )

    validation["post_calibrated_age_group_count"] = pd.to_numeric(
        validation["post_calibrated_age_group_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_post_minus_scaled_target"] = (
        validation["post_calibrated_age_group_count"]
        - validation["scaled_age_group_target_for_generated_total"]
    )

    validation["absolute_difference_post_vs_scaled_target"] = validation[
        "difference_post_minus_scaled_target"
    ].abs()

    validation["difference_post_minus_raw_flat_target"] = (
        validation["post_calibrated_age_group_count"]
        - validation["flat_raw_age_group_target"]
    )

    validation["absolute_difference_post_vs_raw_flat_target"] = validation[
        "difference_post_minus_raw_flat_target"
    ].abs()

    validation["relative_abs_error_post_vs_raw_flat_target"] = validation.apply(
        lambda row: safe_divide(
            row["absolute_difference_post_vs_raw_flat_target"],
            row["flat_raw_age_group_target"],
        ),
        axis=1,
    )

    validation = validation.sort_values(
        [
            "absolute_difference_post_vs_raw_flat_target",
            "flat_raw_age_group_target",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return validation


def create_post_hier_age_sex_validation(
    age_distribution_long: pd.DataFrame,
    post_counts_by_hier_age_sex: dict,
) -> pd.DataFrame:
    hier_targets = (
        age_distribution_long
        .groupby(
            [
                "county_name",
                "settlement_type",
                "sex",
                "exact_age",
            ],
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "hier_age_distribution_count"})
    )

    post_generated = convert_counter_to_dataframe(
        counter=post_counts_by_hier_age_sex,
        key_column_names=[
            "county_name",
            "settlement_type",
            "sex",
            "exact_age",
        ],
        value_column_name="post_calibrated_agent_count",
    )

    validation = hier_targets.merge(
        post_generated,
        on=[
            "county_name",
            "settlement_type",
            "sex",
            "exact_age",
        ],
        how="outer",
    )

    validation["hier_age_distribution_count"] = pd.to_numeric(
        validation["hier_age_distribution_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["post_calibrated_agent_count"] = pd.to_numeric(
        validation["post_calibrated_agent_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    validation["difference_post_minus_hier"] = (
        validation["post_calibrated_agent_count"]
        - validation["hier_age_distribution_count"]
    )

    validation["absolute_difference_post_vs_hier"] = validation[
        "difference_post_minus_hier"
    ].abs()

    validation["relative_abs_error_post_vs_hier"] = validation.apply(
        lambda row: safe_divide(
            row["absolute_difference_post_vs_hier"],
            row["hier_age_distribution_count"],
        ),
        axis=1,
    )

    validation = validation.sort_values(
        [
            "absolute_difference_post_vs_hier",
            "hier_age_distribution_count",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return validation


def create_post_age_calibration_summary(
    current_counts: dict,
    calibration_plan: pd.DataFrame,
    transfer_plan: pd.DataFrame,
    calibration_outputs: dict,
    post_flat_age_validation: pd.DataFrame,
    post_hier_age_sex_validation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.append({
        "metric": "input_agents_total",
        "value": current_counts["total_agents_seen"],
        "note": "Beolvasott agentek száma a 29-es fájlból.",
    })

    rows.append({
        "metric": "output_agents_total",
        "value": calibration_outputs["total_agents_written"],
        "note": "Kiírt agentek száma a 43-as age-calibrated fájlba.",
    })

    rows.append({
        "metric": "age_changed_agents",
        "value": calibration_outputs["total_age_changed_agents"],
        "note": "Ennyi agent exact_age mezője módosult.",
    })

    rows.append({
        "metric": "broad_age_group_changed_agents",
        "value": calibration_outputs["total_broad_age_group_changed_agents"],
        "note": "Ennyi agent broad_age_group mezője is megváltozott.",
    })

    rows.append({
        "metric": "remaining_unapplied_age_changes",
        "value": calibration_outputs["remaining_unapplied_changes"],
        "note": "Ennek 0-nak kell lennie.",
    })

    rows.append({
        "metric": "planned_transfer_total",
        "value": transfer_plan["transfer_count"].sum() if len(transfer_plan) > 0 else 0,
        "note": "Tervezett settlementen belüli korcsoport-áthelyezések száma.",
    })

    rows.append({
        "metric": "pre_calibration_sum_absolute_difference_vs_scaled_target",
        "value": calibration_plan["difference_current_minus_scaled_target"].abs().sum(),
        "note": "Kalibráció előtti abszolút eltérés a generated totalra skálázott flat korcsoport targethez képest.",
    })

    rows.append({
        "metric": "post_calibration_sum_absolute_difference_vs_scaled_target",
        "value": post_flat_age_validation["absolute_difference_post_vs_scaled_target"].sum(),
        "note": "Kalibráció utáni abszolút eltérés a skálázott flat korcsoport targethez képest. Ideálisan 0.",
    })

    rows.append({
        "metric": "post_calibration_max_absolute_difference_vs_scaled_target",
        "value": post_flat_age_validation["absolute_difference_post_vs_scaled_target"].max(),
        "note": "Kalibráció utáni max eltérés a skálázott targethez képest. Ideálisan 0.",
    })

    rows.append({
        "metric": "post_calibration_sum_absolute_difference_vs_raw_flat_target",
        "value": post_flat_age_validation["absolute_difference_post_vs_raw_flat_target"].sum(),
        "note": "Kalibráció utáni abszolút eltérés a nyers flat korcsoport targethez képest. Residual miatt nem feltétlenül 0.",
    })

    rows.append({
        "metric": "post_calibration_max_absolute_difference_vs_raw_flat_target",
        "value": post_flat_age_validation["absolute_difference_post_vs_raw_flat_target"].max(),
        "note": "Legnagyobb settlement × age group eltérés a nyers flat targethez képest.",
    })

    rows.append({
        "metric": "post_hier_age_distribution_total",
        "value": post_hier_age_sex_validation["hier_age_distribution_count"].sum(),
        "note": "Hier age_distribution forrás összesen.",
    })

    rows.append({
        "metric": "post_hier_generated_total",
        "value": post_hier_age_sex_validation["post_calibrated_agent_count"].sum(),
        "note": "Age-calibrated generated agentek összesen a hier összevetésben.",
    })

    rows.append({
        "metric": "post_hier_generated_minus_source_total",
        "value": (
            post_hier_age_sex_validation["post_calibrated_agent_count"].sum()
            - post_hier_age_sex_validation["hier_age_distribution_count"].sum()
        ),
        "note": "Ez várhatóan kb. -5154.",
    })

    rows.append({
        "metric": "post_hier_sum_absolute_cell_difference",
        "value": post_hier_age_sex_validation["absolute_difference_post_vs_hier"].sum(),
        "note": "Kalibráció utáni hier age/sex cellaszintű abszolút eltérés.",
    })

    rows.append({
        "metric": "post_hier_max_absolute_cell_difference",
        "value": post_hier_age_sex_validation["absolute_difference_post_vs_hier"].max(),
        "note": "Kalibráció utáni legnagyobb hier age/sex cellaeltérés.",
    })

    summary = pd.DataFrame(rows)

    return summary


# ============================================================
# 9. FŐ FUTTATÁS
# ============================================================

def run_agent_age_calibration() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Flat long tábla beolvasása...")
    flat_long_table = load_flat_long_table(
        FLAT_LONG_TABLE_CSV
    )

    print("Flat tiszta 10 éves korcsoport targetek kinyerése...")
    flat_age_group_targets = extract_clean_flat_settlement_age_group_targets(
        flat_long_table=flat_long_table
    )

    print("Age distribution long tábla beolvasása...")
    age_distribution_long = load_age_distribution_long(
        AGE_SEX_DISTRIBUTION_LONG_CSV
    )

    print("Exact age választási súlyok építése...")
    exact_age_weight_lookup = build_exact_age_weight_lookup(
        age_distribution_long=age_distribution_long
    )

    print("Generated agentek első pass: current age group countok...")
    current_counts = count_current_generated_age_groups(
        agents_csv=AGENTS_WITH_EXACT_AGE_SEX_CSV
    )

    print("Age group calibration plan készítése...")
    calibration_plan = create_age_group_calibration_plan(
        current_counts=current_counts,
        flat_age_group_targets=flat_age_group_targets,
    )

    print("Age group transfer plan készítése...")
    transfer_plan = create_transfer_plan_from_calibration_plan(
        calibration_plan=calibration_plan
    )

    print("Transfer queue-k építése...")
    transfer_queues = build_transfer_queues(
        transfer_plan=transfer_plan
    )

    print("Agent exact_age kalibráció második pass...")
    calibration_outputs = calibrate_agent_ages_to_transfer_plan(
        input_agents_csv=AGENTS_WITH_EXACT_AGE_SEX_CSV,
        output_agents_csv=AGE_CALIBRATED_AGENTS_CSV,
        current_counts=current_counts,
        transfer_queues=transfer_queues,
        exact_age_weight_lookup=exact_age_weight_lookup,
    )

    print("Post flat age group validáció...")
    post_flat_age_validation = create_post_flat_age_group_validation(
        calibration_plan=calibration_plan,
        post_counts_by_settlement_age_group=calibration_outputs[
            "post_counts_by_settlement_age_group"
        ],
    )

    print("Post hier age/sex validáció...")
    post_hier_age_sex_validation = create_post_hier_age_sex_validation(
        age_distribution_long=age_distribution_long,
        post_counts_by_hier_age_sex=calibration_outputs[
            "post_counts_by_hier_age_sex"
        ],
    )

    print("Post age calibration summary...")
    summary = create_post_age_calibration_summary(
        current_counts=current_counts,
        calibration_plan=calibration_plan,
        transfer_plan=transfer_plan,
        calibration_outputs=calibration_outputs,
        post_flat_age_validation=post_flat_age_validation,
        post_hier_age_sex_validation=post_hier_age_sex_validation,
    )

    print("Outputok mentése...")

    calibration_plan.to_csv(
        AGE_GROUP_CALIBRATION_PLAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    transfer_plan.to_csv(
        AGE_GROUP_CALIBRATION_TRANSFER_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    post_flat_age_validation.to_csv(
        POST_FLAT_AGE_GROUP_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    post_hier_age_sex_validation.to_csv(
        POST_HIER_AGE_SEX_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        POST_AGE_CALIBRATION_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Input agentek: {current_counts['total_agents_seen']:,}")
    print(f"Output agentek: {calibration_outputs['total_agents_written']:,}")
    print(f"Age módosított agentek: {calibration_outputs['total_age_changed_agents']:,}")
    print(f"Broad age group módosított agentek: {calibration_outputs['total_broad_age_group_changed_agents']:,}")
    print(f"Nem alkalmazott módosítások: {calibration_outputs['remaining_unapplied_changes']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


# ============================================================
# 10. PROGRAM INDÍTÁSA
# ============================================================

if __name__ == "__main__":
    run_agent_age_calibration()