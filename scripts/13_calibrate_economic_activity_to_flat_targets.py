from pathlib import Path
import csv
import math
import random
import re
import heapq
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

AGENTS_WITH_ACTIVITY_EDUCATION_CSV = Path(
    "../outputs/agent_attribute_generation/55_agents_with_activity_education.csv"
)

FLAT_LONG_TABLE_CSV = Path(
    "../outputs/flat_validation/05_flat_combined_long_format_table.csv"
)

OUTPUT_FOLDER = Path("../outputs/agent_attribute_generation")

REFERENCE_TARGETS_CSV = OUTPUT_FOLDER / "62_activity_calibration_reference_targets.csv"
CALIBRATION_PLAN_CSV = OUTPUT_FOLDER / "63_activity_calibration_plan.csv"
AGENTS_ACTIVITY_CALIBRATED_CSV = OUTPUT_FOLDER / "64_agents_with_activity_education_activity_calibrated.csv"
CALIBRATION_VALIDATION_CSV = OUTPUT_FOLDER / "65_activity_calibration_validation.csv"
CALIBRATION_SUMMARY_CSV = OUTPUT_FOLDER / "66_activity_calibration_summary.csv"
METHOD_NOTES_CSV = OUTPUT_FOLDER / "67_activity_calibration_method_notes.csv"

RANDOM_SEED = 42

# Teszthez pl. 100_000.
# Teljes futtatáshoz None.
MAX_AGENTS_TO_PROCESS = None

UNDER_15_ACTIVITY = "15 évesnél fiatalabb"
EMPLOYED_ACTIVITY = "Foglalkoztatott"

NON_EMPLOYED_15PLUS_ACTIVITIES = [
    "Munkanélküli",
    "Ellátásban részesülő inaktív",
    "Eltartott",
]

ACTIVITY_ORDER = [
    "Foglalkoztatott",
    "Munkanélküli",
    "Ellátásban részesülő inaktív",
    "Eltartott",
    "15 évesnél fiatalabb",
]


# ============================================================
# 2. ÁLTALÁNOS SEGÉDFÜGGVÉNYEK
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


def parse_exact_age(row: dict) -> int:
    return convert_count_to_integer(row["exact_age"])


# ============================================================
# 3. FLAT ACTIVITY TARGETEK
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
        activity = map_flat_activity_category(row["category"])

        if activity is None:
            continue

        rows.append({
            "economic_activity_status": activity,
            "flat_activity_count": convert_count_to_integer(row["value"]),
            "source_category": row["category"],
        })

    targets = (
        pd.DataFrame(rows)
        .groupby("economic_activity_status", dropna=False)["flat_activity_count"]
        .sum()
        .reset_index()
    )

    return targets


# ============================================================
# 4. GENERATED ACTIVITY DARABSZÁMOK
# ============================================================

def count_generated_activities(agents_csv: Path) -> dict:
    activity_counts = defaultdict(int)
    total_agents = 0

    with agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if MAX_AGENTS_TO_PROCESS is not None and total_agents >= MAX_AGENTS_TO_PROCESS:
                break

            activity = normalize_activity_label(
                row["economic_activity_status_calibrated"]
            )

            activity_counts[activity] += 1
            total_agents += 1

            if total_agents % 1_000_000 == 0:
                print(f"Beolvasott agentek az activity count passban: {total_agents:,}")

    return {
        "activity_counts": dict(activity_counts),
        "total_agents": total_agents,
    }


# ============================================================
# 5. MODELL TARGETEK KÉSZÍTÉSE
# ============================================================

def allocate_by_largest_remainder(
    raw_counts: dict,
    total_to_allocate: int,
) -> dict:
    raw_total = sum(raw_counts.values())

    if raw_total <= 0:
        raise ValueError("Nem lehet allokálni, mert a raw target total 0.")

    rows = []

    for category, raw_count in raw_counts.items():
        expected = raw_count * total_to_allocate / raw_total
        floor_value = math.floor(expected)
        remainder = expected - floor_value

        rows.append({
            "category": category,
            "raw_count": raw_count,
            "expected": expected,
            "allocated": floor_value,
            "remainder": remainder,
        })

    already_allocated = sum(row["allocated"] for row in rows)
    remaining = total_to_allocate - already_allocated

    rows = sorted(
        rows,
        key=lambda row: (row["remainder"], row["expected"]),
        reverse=True,
    )

    for index in range(remaining):
        rows[index]["allocated"] += 1

    return {
        row["category"]: row["allocated"]
        for row in rows
    }


def create_adjusted_model_activity_targets(
    flat_targets: pd.DataFrame,
    generated_activity_counts: dict,
) -> pd.DataFrame:
    """
    Lényeg:
    - 15 év alatti kategóriát nem módosítjuk, mert exact_age alapján következik.
    - Foglalkoztatott targetet fixen a flat / korábbi workplace targethez igazítjuk.
    - A többi 15+ nem foglalkoztatott kategóriát arányosan skálázzuk úgy,
      hogy a teljes generated agent total megmaradjon.
    """
    flat_target_lookup = {
        row["economic_activity_status"]: convert_count_to_integer(row["flat_activity_count"])
        for _, row in flat_targets.iterrows()
    }

    generated_total = sum(generated_activity_counts.values())

    generated_under_15 = generated_activity_counts.get(UNDER_15_ACTIVITY, 0)

    employed_flat_target = flat_target_lookup[EMPLOYED_ACTIVITY]

    generated_15plus_total = generated_total - generated_under_15

    remaining_15plus_after_employed = generated_15plus_total - employed_flat_target

    raw_non_employed_targets = {
        activity: flat_target_lookup[activity]
        for activity in NON_EMPLOYED_15PLUS_ACTIVITIES
    }

    adjusted_non_employed_targets = allocate_by_largest_remainder(
        raw_counts=raw_non_employed_targets,
        total_to_allocate=remaining_15plus_after_employed,
    )

    adjusted_targets = {
        UNDER_15_ACTIVITY: generated_under_15,
        EMPLOYED_ACTIVITY: employed_flat_target,
    }

    adjusted_targets.update(adjusted_non_employed_targets)

    rows = []

    for activity in ACTIVITY_ORDER:
        flat_count = flat_target_lookup.get(activity, 0)
        generated_count = generated_activity_counts.get(activity, 0)
        adjusted_target = adjusted_targets.get(activity, 0)

        rows.append({
            "economic_activity_status": activity,
            "raw_flat_activity_target": flat_count,
            "pre_calibration_generated_count": generated_count,
            "adjusted_model_activity_target": adjusted_target,
            "difference_pre_generated_minus_raw_flat": generated_count - flat_count,
            "difference_pre_generated_minus_adjusted_target": generated_count - adjusted_target,
            "target_note": create_target_note(activity),
        })

    return pd.DataFrame(rows)


def create_target_note(activity: str) -> str:
    if activity == UNDER_15_ACTIVITY:
        return (
            "Nem kalibráljuk activity alapján; exact_age < 15 szabályból következik. "
            "A target a generated 15 év alatti agentek száma."
        )

    if activity == EMPLOYED_ACTIVITY:
        return (
            "Fix target: flat activity / korábbi workplace réteg foglalkoztatotti száma."
        )

    return (
        "15+ nem foglalkoztatott kategória; a raw flat target arányai alapján "
        "a generated 15+ nem foglalkoztatott totalra skálázva."
    )


def write_reference_targets(adjusted_targets: pd.DataFrame) -> None:
    rows = []

    for _, row in adjusted_targets.iterrows():
        rows.append({
            "reference_group": "raw_flat_activity",
            "source_file": "flat_A_nepesseg_adatok_telepulesenkent.xlsx / 05_flat_combined_long_format_table.csv",
            "economic_activity_status": row["economic_activity_status"],
            "target_count": row["raw_flat_activity_target"],
            "note": "Eredeti flat gazdasági aktivitási target.",
        })

        rows.append({
            "reference_group": "adjusted_model_activity_target",
            "source_file": "derived_from_flat_activity_and_generated_population",
            "economic_activity_status": row["economic_activity_status"],
            "target_count": row["adjusted_model_activity_target"],
            "note": row["target_note"],
        })

    pd.DataFrame(rows).to_csv(
        REFERENCE_TARGETS_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 6. KALIBRÁCIÓS TERV
# ============================================================

def create_activity_transfer_plan(adjusted_targets: pd.DataFrame) -> pd.DataFrame:
    """
    Activity kategóriák közötti átmozgatási terv.

    Ha egy kategória current > target, akkor surplus.
    Ha current < target, akkor deficit.
    """
    surplus_items = []
    deficit_items = []

    for _, row in adjusted_targets.iterrows():
        activity = row["economic_activity_status"]

        if activity == UNDER_15_ACTIVITY:
            continue

        difference = convert_count_to_integer(
            row["difference_pre_generated_minus_adjusted_target"]
        )

        if difference > 0:
            surplus_items.append({
                "from_activity": activity,
                "remaining_surplus": difference,
            })

        elif difference < 0:
            deficit_items.append({
                "to_activity": activity,
                "remaining_deficit": abs(difference),
            })

    transfer_rows = []

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
                "from_activity": surplus_item["from_activity"],
                "to_activity": deficit_item["to_activity"],
                "transfer_count": transfer_count,
                "transfer_note": (
                    "Activity calibration: agent nem törlődik, csak "
                    "economic_activity_status_calibrated változik."
                ),
            })

            surplus_item["remaining_surplus"] -= transfer_count
            deficit_item["remaining_deficit"] -= transfer_count

        if surplus_item["remaining_surplus"] == 0:
            surplus_index += 1

        if deficit_item["remaining_deficit"] == 0:
            deficit_index += 1

    transfer_plan = pd.DataFrame(transfer_rows)

    planned_transfer_total = (
        transfer_plan["transfer_count"].sum()
        if len(transfer_plan) > 0
        else 0
    )

    expected_transfer_total = sum(
        max(
            convert_count_to_integer(row["difference_pre_generated_minus_adjusted_target"]),
            0,
        )
        for _, row in adjusted_targets.iterrows()
        if row["economic_activity_status"] != UNDER_15_ACTIVITY
    )

    if planned_transfer_total != expected_transfer_total:
        raise ValueError(
            f"A transfer plan nem zár. "
            f"planned={planned_transfer_total}, expected={expected_transfer_total}"
        )

    return transfer_plan


# ============================================================
# 7. ÁTMOZGATANDÓ AGENTEK KIVÁLASZTÁSA
# ============================================================

def build_source_activity_transfer_counts(transfer_plan: pd.DataFrame) -> dict:
    transfer_counts = defaultdict(int)

    for _, row in transfer_plan.iterrows():
        transfer_counts[row["from_activity"]] += convert_count_to_integer(
            row["transfer_count"]
        )

    return dict(transfer_counts)


def create_candidate_score(row: dict, source_activity: str, random_generator: random.Random) -> float:
    """
    Demográfiailag enyhén hihetőbb kiválasztás.

    Mivel a jelenlegi fő deficit várhatóan az
    'Ellátásban részesülő inaktív' kategóriában van,
    a magasabb életkorú forrás-agenteket preferáljuk.

    Ez nem változtatja az exact_age-et, csak azt dönti el,
    mely agentek kerüljenek át másik activity kategóriába.
    """
    exact_age = parse_exact_age(row)

    random_noise = random_generator.random()

    if source_activity == "Foglalkoztatott":
        return exact_age * 10.0 + random_noise

    if source_activity == "Munkanélküli":
        return exact_age * 10.0 + random_noise

    if source_activity == "Eltartott":
        return exact_age * 10.0 + random_noise

    return random_noise


def select_agent_rows_for_activity_change(
    agents_csv: Path,
    source_activity_transfer_counts: dict,
) -> dict:
    """
    Első pass:
    kiválasztja, mely CSV sorok activity státusza változzon.

    Nem tároljuk az egész agentfájlt memóriában.
    Minden source activity-hez csak annyi heap elemet tartunk,
    ahány agentet át kell mozgatni.
    """
    random_generator = random.Random(RANDOM_SEED)

    heaps_by_source_activity = {
        activity: []
        for activity in source_activity_transfer_counts
    }

    total_seen = 0

    with agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row_number, row in enumerate(reader, start=1):
            if MAX_AGENTS_TO_PROCESS is not None and total_seen >= MAX_AGENTS_TO_PROCESS:
                break

            activity = normalize_activity_label(
                row["economic_activity_status_calibrated"]
            )

            if activity in source_activity_transfer_counts:
                max_selected = source_activity_transfer_counts[activity]
                score = create_candidate_score(
                    row=row,
                    source_activity=activity,
                    random_generator=random_generator,
                )

                heap = heaps_by_source_activity[activity]

                candidate = (score, row_number)

                if len(heap) < max_selected:
                    heapq.heappush(heap, candidate)
                else:
                    if candidate[0] > heap[0][0]:
                        heapq.heapreplace(heap, candidate)

            total_seen += 1

            if total_seen % 1_000_000 == 0:
                print(f"Kiválasztási pass agentek: {total_seen:,}")

    selected_rows_by_source_activity = {}

    for activity, heap in heaps_by_source_activity.items():
        expected_count = source_activity_transfer_counts[activity]

        if len(heap) != expected_count:
            raise ValueError(
                f"Nincs elég kiválasztható agent ehhez a source activityhez: {activity}. "
                f"expected={expected_count}, selected={len(heap)}"
            )

        selected_row_numbers = [
            row_number
            for _, row_number in heap
        ]

        selected_rows_by_source_activity[activity] = selected_row_numbers

    return selected_rows_by_source_activity


def build_selected_row_to_target_activity_lookup(
    selected_rows_by_source_activity: dict,
    transfer_plan: pd.DataFrame,
) -> dict:
    """
    A kiválasztott sorokhoz hozzárendeli, melyik target activity-be menjenek.
    """
    random_generator = random.Random(RANDOM_SEED)

    row_to_target_activity = {}

    for source_activity, selected_rows in selected_rows_by_source_activity.items():
        target_activity_queue = []

        source_transfers = transfer_plan[
            transfer_plan["from_activity"] == source_activity
        ].copy()

        for _, transfer_row in source_transfers.iterrows():
            target_activity_queue.extend(
                [transfer_row["to_activity"]]
                * convert_count_to_integer(transfer_row["transfer_count"])
            )

        if len(target_activity_queue) != len(selected_rows):
            raise ValueError(
                f"A kiválasztott sorok és target queue mérete nem egyezik: "
                f"{source_activity}, selected={len(selected_rows)}, "
                f"queue={len(target_activity_queue)}"
            )

        random_generator.shuffle(target_activity_queue)

        for row_number, target_activity in zip(selected_rows, target_activity_queue):
            row_to_target_activity[row_number] = target_activity

    return row_to_target_activity


# ============================================================
# 8. MÁSODIK PASS: ACTIVITY KALIBRÁLT AGENTFÁJL
# ============================================================

def write_activity_calibrated_agents(
    input_agents_csv: Path,
    output_agents_csv: Path,
    row_to_target_activity: dict,
) -> dict:
    post_activity_counts = defaultdict(int)

    total_written = 0
    changed_agents = 0

    with input_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file, \
            output_agents_csv.open("w", encoding="utf-8-sig", newline="") as output_file:

        reader = csv.DictReader(input_file)

        input_fieldnames = list(reader.fieldnames)

        new_columns = [
            "economic_activity_status_before_flat_activity_calibration",
            "flat_activity_calibrated",
            "flat_activity_calibration_source",
        ]

        output_fieldnames = input_fieldnames + [
            column
            for column in new_columns
            if column not in input_fieldnames
        ]

        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for row_number, row in enumerate(reader, start=1):
            if MAX_AGENTS_TO_PROCESS is not None and total_written >= MAX_AGENTS_TO_PROCESS:
                break

            original_activity = normalize_activity_label(
                row["economic_activity_status_calibrated"]
            )

            final_activity = original_activity

            row["economic_activity_status_before_flat_activity_calibration"] = original_activity

            if row_number in row_to_target_activity:
                final_activity = row_to_target_activity[row_number]
                row["economic_activity_status_calibrated"] = final_activity
                row["flat_activity_calibrated"] = "True"
                row["flat_activity_calibration_source"] = "flat_activity_adjusted_model_target"
                changed_agents += 1
            else:
                row["economic_activity_status_calibrated"] = final_activity
                row["flat_activity_calibrated"] = "False"
                row["flat_activity_calibration_source"] = "unchanged"

            post_activity_counts[final_activity] += 1

            writer.writerow(row)

            total_written += 1

            if total_written % 1_000_000 == 0:
                print(f"Kiírt activity-kalibrált agentek: {total_written:,}")

    return {
        "total_written": total_written,
        "changed_agents": changed_agents,
        "post_activity_counts": dict(post_activity_counts),
    }


# ============================================================
# 9. VALIDÁCIÓ ÉS SUMMARY
# ============================================================

def create_validation_table(
    adjusted_targets: pd.DataFrame,
    post_activity_counts: dict,
) -> pd.DataFrame:
    validation = adjusted_targets.copy()

    validation["post_calibration_generated_count"] = validation[
        "economic_activity_status"
    ].map(post_activity_counts).fillna(0).apply(convert_count_to_integer)

    validation["difference_post_generated_minus_raw_flat"] = (
        validation["post_calibration_generated_count"]
        - validation["raw_flat_activity_target"]
    )

    validation["difference_post_generated_minus_adjusted_target"] = (
        validation["post_calibration_generated_count"]
        - validation["adjusted_model_activity_target"]
    )

    validation["absolute_difference_post_vs_raw_flat"] = validation[
        "difference_post_generated_minus_raw_flat"
    ].abs()

    validation["absolute_difference_post_vs_adjusted_target"] = validation[
        "difference_post_generated_minus_adjusted_target"
    ].abs()

    validation["relative_abs_error_post_vs_raw_flat"] = validation.apply(
        lambda row: safe_divide(
            row["absolute_difference_post_vs_raw_flat"],
            row["raw_flat_activity_target"],
        ),
        axis=1,
    )

    validation = validation.sort_values(
        ["absolute_difference_post_vs_raw_flat", "raw_flat_activity_target"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return validation


def create_summary_table(
    generated_count_result: dict,
    adjusted_targets: pd.DataFrame,
    transfer_plan: pd.DataFrame,
    write_result: dict,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    pre_abs_adjusted = adjusted_targets[
        "difference_pre_generated_minus_adjusted_target"
    ].abs().sum()

    post_abs_adjusted = validation[
        "absolute_difference_post_vs_adjusted_target"
    ].sum()

    pre_abs_raw = adjusted_targets[
        "difference_pre_generated_minus_raw_flat"
    ].abs().sum()

    post_abs_raw = validation[
        "absolute_difference_post_vs_raw_flat"
    ].sum()

    flat_total = validation["raw_flat_activity_target"].sum()
    adjusted_total = validation["adjusted_model_activity_target"].sum()
    pre_generated_total = adjusted_targets["pre_calibration_generated_count"].sum()
    post_generated_total = validation["post_calibration_generated_count"].sum()

    rows = [
        {
            "metric": "input_agents_total",
            "value": generated_count_result["total_agents"],
            "note": "Input agentek száma az 55-ös fájlban.",
        },
        {
            "metric": "output_agents_total",
            "value": write_result["total_written"],
            "note": "Output agentek száma a 64-es fájlban.",
        },
        {
            "metric": "changed_activity_agents",
            "value": write_result["changed_agents"],
            "note": "Ennyi agent economic_activity_status_calibrated mezője változott.",
        },
        {
            "metric": "planned_transfer_total",
            "value": transfer_plan["transfer_count"].sum() if len(transfer_plan) > 0 else 0,
            "note": "Tervezett activity áthelyezések száma.",
        },
        {
            "metric": "raw_flat_activity_total",
            "value": flat_total,
            "note": "Flat gazdasági aktivitás target összesen.",
        },
        {
            "metric": "adjusted_model_activity_total",
            "value": adjusted_total,
            "note": "A generated populációhoz igazított activity target összesen.",
        },
        {
            "metric": "pre_calibration_generated_activity_total",
            "value": pre_generated_total,
            "note": "Kalibráció előtti generated activity total.",
        },
        {
            "metric": "post_calibration_generated_activity_total",
            "value": post_generated_total,
            "note": "Kalibráció utáni generated activity total.",
        },
        {
            "metric": "pre_generated_minus_raw_flat_total",
            "value": pre_generated_total - flat_total,
            "note": "Kalibráció előtti total eltérés a raw flat targethez képest.",
        },
        {
            "metric": "post_generated_minus_raw_flat_total",
            "value": post_generated_total - flat_total,
            "note": "Kalibráció utáni total eltérés a raw flat targethez képest. Residual miatt nem feltétlenül 0.",
        },
        {
            "metric": "pre_sum_absolute_difference_vs_raw_flat",
            "value": pre_abs_raw,
            "note": "Kalibráció előtti kategóriaszintű abszolút eltérés a raw flat targethez képest.",
        },
        {
            "metric": "post_sum_absolute_difference_vs_raw_flat",
            "value": post_abs_raw,
            "note": "Kalibráció utáni kategóriaszintű abszolút eltérés a raw flat targethez képest.",
        },
        {
            "metric": "pre_sum_absolute_difference_vs_adjusted_target",
            "value": pre_abs_adjusted,
            "note": "Kalibráció előtti kategóriaszintű abszolút eltérés az adjusted targethez képest.",
        },
        {
            "metric": "post_sum_absolute_difference_vs_adjusted_target",
            "value": post_abs_adjusted,
            "note": "Kalibráció utáni kategóriaszintű abszolút eltérés az adjusted targethez képest. Ideálisan 0.",
        },
        {
            "metric": "post_max_absolute_difference_vs_adjusted_target",
            "value": validation["absolute_difference_post_vs_adjusted_target"].max(),
            "note": "Kalibráció utáni max kategóriaeltérés az adjusted targethez képest. Ideálisan 0.",
        },
        {
            "metric": "post_employed_count",
            "value": int(
                validation.loc[
                    validation["economic_activity_status"] == EMPLOYED_ACTIVITY,
                    "post_calibration_generated_count",
                ].iloc[0]
            ),
            "note": "Ennek 4 718 660-nak kell lennie.",
        },
        {
            "metric": "post_employed_difference_vs_raw_flat",
            "value": int(
                validation.loc[
                    validation["economic_activity_status"] == EMPLOYED_ACTIVITY,
                    "difference_post_generated_minus_raw_flat",
                ].iloc[0]
            ),
            "note": "Ennek 0-nak kell lennie.",
        },
    ]

    return pd.DataFrame(rows)


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Input: 55_agents_with_activity_education.csv.",
        },
        {
            "order": 2,
            "note": "Output: 64_agents_with_activity_education_activity_calibrated.csv.",
        },
        {
            "order": 3,
            "note": "Cél: a gazdasági aktivitási kategóriák igazítása a flat activity targethez, különösen a foglalkoztatottak számának visszaigazítása 4 718 660-ra.",
        },
        {
            "order": 4,
            "note": "Agentet nem törlünk és új agentet nem generálunk.",
        },
        {
            "order": 5,
            "note": "Település, household, sex, exact_age és education_level_calibrated nem változik.",
        },
        {
            "order": 6,
            "note": "Csak az economic_activity_status_calibrated mező változhat.",
        },
        {
            "order": 7,
            "note": "A 15 évesnél fiatalabb kategóriát nem módosítjuk, mert az exact_age alapján szabályból következik.",
        },
        {
            "order": 8,
            "note": "A Foglalkoztatott kategória targetje fixen a flat / korábbi workplace réteg foglalkoztatotti száma.",
        },
        {
            "order": 9,
            "note": "A többi 15+ nem foglalkoztatott kategória targetje a raw flat arányok alapján skálázódik a generated 15+ nem foglalkoztatott totalra.",
        },
        {
            "order": 10,
            "note": "A raw flat targethez képest maradhat total eltérés, mert a generated populáció totalja kisebb a flat activity totalnál.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 10. FŐ FUTTATÁS
# ============================================================

def run_activity_calibration() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Flat long table beolvasása...")
    flat_long_table = load_flat_long_table(
        flat_long_table_csv=FLAT_LONG_TABLE_CSV
    )

    print("Flat activity targetek kinyerése...")
    flat_targets = extract_flat_activity_targets(
        flat_long_table=flat_long_table
    )

    print("Generated activity countok számolása...")
    generated_count_result = count_generated_activities(
        agents_csv=AGENTS_WITH_ACTIVITY_EDUCATION_CSV
    )

    print("Adjusted model activity targetek készítése...")
    adjusted_targets = create_adjusted_model_activity_targets(
        flat_targets=flat_targets,
        generated_activity_counts=generated_count_result["activity_counts"],
    )

    print("Reference target fájl írása...")
    write_reference_targets(
        adjusted_targets=adjusted_targets
    )

    print("Activity transfer plan készítése...")
    transfer_plan = create_activity_transfer_plan(
        adjusted_targets=adjusted_targets
    )

    print("Átmozgatandó agentek kiválasztása...")
    source_activity_transfer_counts = build_source_activity_transfer_counts(
        transfer_plan=transfer_plan
    )

    selected_rows_by_source_activity = select_agent_rows_for_activity_change(
        agents_csv=AGENTS_WITH_ACTIVITY_EDUCATION_CSV,
        source_activity_transfer_counts=source_activity_transfer_counts,
    )

    row_to_target_activity = build_selected_row_to_target_activity_lookup(
        selected_rows_by_source_activity=selected_rows_by_source_activity,
        transfer_plan=transfer_plan,
    )

    print("Activity-kalibrált agentfájl írása...")
    write_result = write_activity_calibrated_agents(
        input_agents_csv=AGENTS_WITH_ACTIVITY_EDUCATION_CSV,
        output_agents_csv=AGENTS_ACTIVITY_CALIBRATED_CSV,
        row_to_target_activity=row_to_target_activity,
    )

    print("Validáció készítése...")
    validation = create_validation_table(
        adjusted_targets=adjusted_targets,
        post_activity_counts=write_result["post_activity_counts"],
    )

    print("Summary készítése...")
    summary = create_summary_table(
        generated_count_result=generated_count_result,
        adjusted_targets=adjusted_targets,
        transfer_plan=transfer_plan,
        write_result=write_result,
        validation=validation,
    )

    print("Method notes írása...")
    write_method_notes()

    print("Outputok mentése...")

    transfer_plan.to_csv(
        CALIBRATION_PLAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    validation.to_csv(
        CALIBRATION_VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        CALIBRATION_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Input agentek: {generated_count_result['total_agents']:,}")
    print(f"Output agentek: {write_result['total_written']:,}")
    print(f"Módosított activity státuszú agentek: {write_result['changed_agents']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_activity_calibration()