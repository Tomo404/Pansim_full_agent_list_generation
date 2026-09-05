from pathlib import Path
import csv
from collections import defaultdict
import math

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

CURRENT_AGENTS_CSV = Path(
    "../outputs/agent_attribute_generation/school_attendance_assignment/"
    "116_agents_with_school_attendance.csv"
)

ORIGINAL_AGE_BASED_HEALTH_TARGETS_CSV = Path(
    "../outputs/agent_attribute_generation/health_status_assignment/"
    "127_age_based_health_targets.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/health_status_assignment"
)

SCALED_HEALTH_TARGETS_CSV = OUTPUT_FOLDER / "137_generated_scaled_age_based_health_targets.csv"
SCALED_HEALTH_TARGET_COMPARISON_CSV = OUTPUT_FOLDER / "138_generated_scaled_health_target_comparison.csv"
SCALED_HEALTH_TARGET_SUMMARY_CSV = OUTPUT_FOLDER / "139_generated_scaled_health_target_summary.csv"

MAX_ROWS_TO_PROCESS = None

UNDER_5_NOT_IN_SOURCE = "UNDER_5_NOT_IN_SOURCE"


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def normalize_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value).strip()


def convert_count_to_integer(value) -> int:
    text = normalize_text(value)

    if text == "":
        return 0

    return int(round(float(text)))


def normalize_county_name(value) -> str:
    county_name = normalize_text(value)

    if county_name.endswith(" vármegye"):
        county_name = county_name.replace(" vármegye", "").strip()

    manual = {
        "Gyor-Moson-Sopron": "Győr-Moson-Sopron",
        "Győr-Moson-Sopron": "Győr-Moson-Sopron",
        "Főváros": "Budapest",
        "főváros": "Budapest",
        "Fováros": "Budapest",
        "fováros": "Budapest",
        "Budapest": "Budapest",
    }

    return manual.get(county_name, county_name)


def normalize_settlement_type(value) -> str:
    settlement_type = normalize_text(value)

    manual = {
        "Főváros": "Főváros",
        "főváros": "Főváros",
        "Fováros": "Főváros",
        "fováros": "Főváros",
        "Megyei jogú város(ok)": "Megyei jogú város(ok)",
        "Egyéb város(ok)": "Egyéb város(ok)",
        "Község(ek)": "Község(ek)",
    }

    return manual.get(settlement_type, settlement_type)


def map_exact_age_to_health_age_group(exact_age: int) -> str:
    age = int(exact_age)

    if age < 5:
        return UNDER_5_NOT_IN_SOURCE

    if age <= 14:
        return "5–14 éves"

    if age <= 24:
        return "15–24 éves"

    if age <= 34:
        return "25–34 éves"

    if age <= 44:
        return "35–44 éves"

    if age <= 54:
        return "45–54 éves"

    if age <= 64:
        return "55–64 éves"

    if age <= 74:
        return "65–74 éves"

    if age <= 84:
        return "75–84 éves"

    return "85 éves és idősebb"


def make_demo_key(
    county_name: str,
    settlement_type: str,
    health_age_group: str,
) -> tuple[str, str, str]:
    return (
        normalize_county_name(county_name),
        normalize_settlement_type(settlement_type),
        normalize_text(health_age_group),
    )


def largest_remainder_allocation(
    category_weights: dict[str, int],
    target_total: int,
) -> dict[str, int]:
    """
    Egy cellán belül a health response kategóriák számát úgy skálázza,
    hogy az összeg pontosan target_total legyen.

    Az arányokat az eredeti health tábla response distributionje adja.
    """
    source_total = sum(category_weights.values())

    if target_total == 0:
        return {
            category: 0
            for category in category_weights
        }

    if source_total == 0:
        raise ValueError(
            "Nem lehet 0 source_totalból arányokat számolni nem nulla target_totalhoz."
        )

    raw_allocations = []

    for category, source_count in category_weights.items():
        raw_value = source_count * target_total / source_total
        floor_value = math.floor(raw_value)
        remainder = raw_value - floor_value

        raw_allocations.append({
            "category": category,
            "floor_value": floor_value,
            "remainder": remainder,
        })

    allocated_total = sum(item["floor_value"] for item in raw_allocations)
    remaining = target_total - allocated_total

    raw_allocations = sorted(
        raw_allocations,
        key=lambda item: item["remainder"],
        reverse=True,
    )

    result = {}

    for index, item in enumerate(raw_allocations):
        extra = 1 if index < remaining else 0
        result[item["category"]] = item["floor_value"] + extra

    return result


# ============================================================
# 3. CURRENT AGENTEK SZÁMLÁLÁSA
# ============================================================

def count_current_5plus_agents_by_demo_key(current_agents_csv: Path) -> dict:
    total_rows = 0
    age_5plus_total = 0
    under_5_total = 0

    candidate_counts_by_demo_key = defaultdict(int)

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            exact_age = convert_count_to_integer(row["exact_age"])

            if exact_age < 5:
                under_5_total += 1
            else:
                demo_key = make_demo_key(
                    county_name=row["county_name"],
                    settlement_type=row["settlement_type"],
                    health_age_group=map_exact_age_to_health_age_group(exact_age),
                )

                candidate_counts_by_demo_key[demo_key] += 1
                age_5plus_total += 1

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"Current 5+ health target scaling count pass: {total_rows:,} agent")

    return {
        "total_rows": total_rows,
        "age_5plus_total": age_5plus_total,
        "under_5_total": under_5_total,
        "candidate_counts_by_demo_key": dict(candidate_counts_by_demo_key),
    }


# ============================================================
# 4. TARGET SKÁLÁZÁS
# ============================================================

def load_original_health_targets(targets_csv: Path) -> pd.DataFrame:
    targets = pd.read_csv(
        targets_csv,
        keep_default_na=False,
    )

    targets["county_name"] = targets["county_name"].apply(normalize_county_name)
    targets["settlement_type"] = targets["settlement_type"].apply(normalize_settlement_type)
    targets["health_age_group"] = targets["health_age_group"].apply(normalize_text)
    targets["health_question"] = targets["health_question"].apply(normalize_text)
    targets["health_response_category"] = targets["health_response_category"].apply(normalize_text)
    targets["target_count"] = targets["target_count"].apply(convert_count_to_integer)

    return targets


def build_response_weights(targets: pd.DataFrame) -> dict:
    weights = defaultdict(lambda: defaultdict(int))

    for _, row in targets.iterrows():
        demo_key = make_demo_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            health_age_group=row["health_age_group"],
        )

        health_question = normalize_text(row["health_question"])
        response_category = normalize_text(row["health_response_category"])
        target_count = convert_count_to_integer(row["target_count"])

        weights[(health_question, demo_key)][response_category] += target_count

    return {
        key: dict(counter)
        for key, counter in weights.items()
    }


def build_fallback_response_weights(targets: pd.DataFrame) -> dict:
    """
    Ha valamilyen generated demo cellához nincs eredeti health target,
    akkor előbb országos health_age_group × question arányt,
    végső esetben országos question arányt használunk.
    """
    by_question_age = defaultdict(lambda: defaultdict(int))
    by_question = defaultdict(lambda: defaultdict(int))

    for _, row in targets.iterrows():
        health_question = normalize_text(row["health_question"])
        health_age_group = normalize_text(row["health_age_group"])
        response_category = normalize_text(row["health_response_category"])
        target_count = convert_count_to_integer(row["target_count"])

        by_question_age[(health_question, health_age_group)][response_category] += target_count
        by_question[health_question][response_category] += target_count

    return {
        "by_question_age": {
            key: dict(counter)
            for key, counter in by_question_age.items()
        },
        "by_question": {
            key: dict(counter)
            for key, counter in by_question.items()
        },
    }


def create_scaled_health_targets(
    original_targets: pd.DataFrame,
    candidate_counts_by_demo_key: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    response_weights = build_response_weights(
        targets=original_targets
    )

    fallback_weights = build_fallback_response_weights(
        targets=original_targets
    )

    health_questions = sorted(original_targets["health_question"].unique())

    scaled_rows = []
    comparison_rows = []

    original_demo_totals = (
        original_targets
        .groupby(
            [
                "health_question",
                "county_name",
                "settlement_type",
                "health_age_group",
            ],
            dropna=False,
        )["target_count"]
        .sum()
        .reset_index()
    )

    original_total_lookup = {}

    for _, row in original_demo_totals.iterrows():
        demo_key = make_demo_key(
            county_name=row["county_name"],
            settlement_type=row["settlement_type"],
            health_age_group=row["health_age_group"],
        )

        original_total_lookup[
            (
                normalize_text(row["health_question"]),
                demo_key,
            )
        ] = convert_count_to_integer(row["target_count"])

    all_demo_keys = set(candidate_counts_by_demo_key.keys())

    for health_question in health_questions:
        for demo_key in sorted(all_demo_keys):
            generated_cell_total = candidate_counts_by_demo_key.get(demo_key, 0)

            weight_key = (health_question, demo_key)
            category_weights = response_weights.get(weight_key)

            weight_source = "exact_health_cell_distribution"

            if category_weights is None or sum(category_weights.values()) == 0:
                category_weights = fallback_weights["by_question_age"].get(
                    (
                        health_question,
                        demo_key[2],
                    )
                )

                weight_source = "national_age_group_distribution"

            if category_weights is None or sum(category_weights.values()) == 0:
                category_weights = fallback_weights["by_question"].get(health_question)
                weight_source = "national_question_distribution"

            if category_weights is None or sum(category_weights.values()) == 0:
                raise ValueError(
                    f"Nincs használható health response distribution ehhez: "
                    f"{health_question}, {demo_key}"
                )

            allocation = largest_remainder_allocation(
                category_weights=category_weights,
                target_total=generated_cell_total,
            )

            original_cell_total = original_total_lookup.get(weight_key, 0)

            comparison_rows.append({
                "health_question": health_question,
                "county_name": demo_key[0],
                "settlement_type": demo_key[1],
                "health_age_group": demo_key[2],
                "original_health_target_total": original_cell_total,
                "generated_agent_cell_total": generated_cell_total,
                "difference_generated_minus_original": generated_cell_total - original_cell_total,
                "absolute_difference": abs(generated_cell_total - original_cell_total),
                "response_distribution_source": weight_source,
            })

            for response_category, scaled_count in allocation.items():
                scaled_rows.append({
                    "county_name": demo_key[0],
                    "settlement_type": demo_key[1],
                    "health_age_group": demo_key[2],
                    "health_question": health_question,
                    "health_response_category": response_category,
                    "target_count": int(scaled_count),
                    "original_cell_total": original_cell_total,
                    "generated_cell_total": generated_cell_total,
                    "response_distribution_source": weight_source,
                })

    return pd.DataFrame(scaled_rows), pd.DataFrame(comparison_rows)


# ============================================================
# 5. SUMMARY
# ============================================================

def create_summary(
    current_counts: dict,
    original_targets: pd.DataFrame,
    scaled_targets: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "metric": "current_agents_total",
            "value": current_counts["total_rows"],
            "note": "Aktuális agentfájl teljes sorainak száma.",
        },
        {
            "metric": "current_age_5plus_agents_total",
            "value": current_counts["age_5plus_total"],
            "note": "Aktuális 5+ agentek száma. Ehhez skálázzuk a health targetet.",
        },
        {
            "metric": "current_under_5_agents_total",
            "value": current_counts["under_5_total"],
            "note": "5 év alatti agentek száma, akik nem szerepelnek a health forrástáblában.",
        },
    ]

    original_by_question = (
        original_targets
        .groupby("health_question", dropna=False)["target_count"]
        .sum()
        .reset_index()
    )

    scaled_by_question = (
        scaled_targets
        .groupby("health_question", dropna=False)["target_count"]
        .sum()
        .reset_index()
    )

    merged = original_by_question.merge(
        scaled_by_question,
        on="health_question",
        how="outer",
        suffixes=("_original", "_scaled"),
    ).fillna(0)

    for _, row in merged.iterrows():
        health_question = normalize_text(row["health_question"])
        original_total = int(row["target_count_original"])
        scaled_total = int(row["target_count_scaled"])

        rows.append({
            "metric": f"original_health_target_total__{health_question}",
            "value": original_total,
            "note": "Eredeti health tábla targetje az adott kérdésre.",
        })

        rows.append({
            "metric": f"scaled_health_target_total__{health_question}",
            "value": scaled_total,
            "note": "Generated 5+ agentekhez igazított health target az adott kérdésre.",
        })

        rows.append({
            "metric": f"scaled_minus_original__{health_question}",
            "value": scaled_total - original_total,
            "note": "Várhatóan -5956 körüli, mert a generated 5+ agent count kisebb a health source 5+ bázisánál.",
        })

    rows.append({
        "metric": "cell_sum_absolute_difference_generated_vs_original",
        "value": int(comparison["absolute_difference"].sum()),
        "note": "Cellaszintű current generated count vs original health target abszolút eltérés összege kérdéseken együtt.",
    })

    rows.append({
        "metric": "cell_max_absolute_difference_generated_vs_original",
        "value": int(comparison["absolute_difference"].max()),
        "note": "Legnagyobb cellaszintű eltérés.",
    })

    rows.append({
        "metric": "fallback_distribution_cells",
        "value": int((comparison["response_distribution_source"] != "exact_health_cell_distribution").sum()),
        "note": "Olyan cellák száma, ahol nem volt exact health distribution, ezért fallback arányt kellett volna használni. Ideálisan 0.",
    })

    return pd.DataFrame(rows)


# ============================================================
# 6. FŐ FUTTATÁS
# ============================================================

def run_generated_scaled_health_target_preparation() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Current 5+ agentek számlálása...")
    current_counts = count_current_5plus_agents_by_demo_key(
        current_agents_csv=CURRENT_AGENTS_CSV
    )

    print("Eredeti age-based health targetek betöltése...")
    original_targets = load_original_health_targets(
        targets_csv=ORIGINAL_AGE_BASED_HEALTH_TARGETS_CSV
    )

    print("Generated 5+ agentekhez igazított health targetek előállítása...")
    scaled_targets, comparison = create_scaled_health_targets(
        original_targets=original_targets,
        candidate_counts_by_demo_key=current_counts["candidate_counts_by_demo_key"],
    )

    print("Summary készítése...")
    summary = create_summary(
        current_counts=current_counts,
        original_targets=original_targets,
        scaled_targets=scaled_targets,
        comparison=comparison,
    )

    print("Outputok mentése...")
    scaled_targets.to_csv(
        SCALED_HEALTH_TARGETS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        SCALED_HEALTH_TARGET_COMPARISON_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SCALED_HEALTH_TARGET_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Current 5+ agents: {current_counts['age_5plus_total']:,}")
    print(f"Current under 5 agents: {current_counts['under_5_total']:,}")

    for health_question, group in scaled_targets.groupby("health_question"):
        print(f"{health_question}: scaled target {int(group['target_count'].sum()):,}")

    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_generated_scaled_health_target_preparation()