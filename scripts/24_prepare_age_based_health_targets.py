from pathlib import Path
import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

HEALTH_LONG_CSV = Path(
    "../outputs/agent_attribute_generation/health_status_assignment/"
    "121_health_status_hier_long.csv"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/health_status_assignment"
)

AGE_BASED_HEALTH_TARGETS_CSV = OUTPUT_FOLDER / "127_age_based_health_targets.csv"
AGE_BASED_HEALTH_TARGET_SUMMARY_CSV = OUTPUT_FOLDER / "128_age_based_health_target_summary.csv"
HEALTH_DIMENSION_CLEANING_NOTES_CSV = OUTPUT_FOLDER / "129_health_dimension_cleaning_notes.csv"

BASE_POPULATION_AGE_5PLUS_EXPECTED = 9_140_394


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def create_clean_age_based_targets(health_long: pd.DataFrame) -> pd.DataFrame:
    """
    Első egészségi állapot assignmenthez csak az age_group dimenziót használjuk.

    Fontos:
    - A health tábla több dimenzióban ismétli ugyanazt a bázisnépességet.
    - Assignment targetként most csak az age_group blokkot használjuk.
    - Így automatikusan elkerüljük a 15 évesnél fiatalabb személy duplikációját,
      mert az age_group blokkban nincs ilyen külön duplikált kategória.
    """

    age_targets = health_long[
        health_long["dimension_group"] == "age_group"
    ].copy()

    grouped = (
        age_targets
        .groupby(
            [
                "county_name",
                "settlement_type",
                "dimension_category",
                "health_question",
                "health_response_category",
            ],
            dropna=False,
        )["health_count"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "dimension_category": "health_age_group",
                "health_count": "target_count",
            }
        )
    )

    return grouped


def create_summary(age_based_targets: pd.DataFrame) -> pd.DataFrame:
    rows = []

    total_by_question = (
        age_based_targets
        .groupby("health_question", dropna=False)["target_count"]
        .sum()
        .reset_index()
    )

    for _, row in total_by_question.iterrows():
        rows.append({
            "metric": f"total_target__{row['health_question']}",
            "value": int(row["target_count"]),
            "note": "Age-group based health target total for one health question."
        })

    total_by_question_and_age = (
        age_based_targets
        .groupby(
            [
                "health_question",
                "health_age_group",
            ],
            dropna=False,
        )["target_count"]
        .sum()
        .reset_index()
    )

    for _, row in total_by_question_and_age.iterrows():
        rows.append({
            "metric": f"age_total__{row['health_question']}__{row['health_age_group']}",
            "value": int(row["target_count"]),
            "note": "National age-group target within one health question."
        })

    total_by_question_and_response = (
        age_based_targets
        .groupby(
            [
                "health_question",
                "health_response_category",
            ],
            dropna=False,
        )["target_count"]
        .sum()
        .reset_index()
    )

    for _, row in total_by_question_and_response.iterrows():
        rows.append({
            "metric": f"response_total__{row['health_question']}__{row['health_response_category']}",
            "value": int(row["target_count"]),
            "note": "National health response target within one health question."
        })

    for _, row in total_by_question.iterrows():
        difference = int(row["target_count"]) - BASE_POPULATION_AGE_5PLUS_EXPECTED

        rows.append({
            "metric": f"difference_from_expected_5plus_base__{row['health_question']}",
            "value": difference,
            "note": "Should be 0 if this health question covers the expected 5+ base population."
        })

    return pd.DataFrame(rows)


def write_cleaning_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Az egészségi állapot tábla több fő dimenzióban ismétli ugyanazt a bázisnépességet: nem, korcsoport, végzettség, gazdasági aktivitás."
        },
        {
            "order": 2,
            "note": "A három health_question külön attribútumként kezelendő: disability_or_severe_limitation, chronic_disease, limitation_level."
        },
        {
            "order": 3,
            "note": "Mindhárom health_question ugyanarra a kb. 5 éves és idősebb bázisnépességre vonatkozik."
        },
        {
            "order": 4,
            "note": "A 15 évesnél fiatalabb személy kategória két külön blokkban is szerepel az exportban, azonos értékekkel. Ezt assignment targetként nem szabad kétszer számolni."
        },
        {
            "order": 5,
            "note": "Az első health assignment verzióban az age_group dimenziót használjuk fő targetként, mert egészségi állapotnál az életkor a legstabilabb kiosztási alap."
        },
        {
            "order": 6,
            "note": "A sex, education és economic_activity dimenziókat első körben validációra használjuk, nem egyszerre kényszerített assignment targetként."
        },
        {
            "order": 7,
            "note": "Az 5 év alatti agentek nem szerepelnek a forrástáblában, ezért health assignmentnél külön UNDER_5_NOT_IN_SOURCE / NAPP jelölést kell kapniuk."
        },
    ]

    pd.DataFrame(notes).to_csv(
        HEALTH_DIMENSION_CLEANING_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 3. FŐ FUTTATÁS
# ============================================================

def run_age_based_health_target_preparation() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Health long tábla betöltése...")
    health_long = pd.read_csv(
        HEALTH_LONG_CSV,
        keep_default_na=False,
    )

    print("Age-based health targetek előállítása...")
    age_based_targets = create_clean_age_based_targets(
        health_long=health_long
    )

    print("Summary készítése...")
    summary = create_summary(
        age_based_targets=age_based_targets
    )

    print("Cleaning notes írása...")
    write_cleaning_notes()

    print("Outputok mentése...")
    age_based_targets.to_csv(
        AGE_BASED_HEALTH_TARGETS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        AGE_BASED_HEALTH_TARGET_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Age-based health target rows: {len(age_based_targets):,}")

    for health_question, group in age_based_targets.groupby("health_question"):
        print(f"{health_question}: {int(group['target_count'].sum()):,}")

    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_age_based_health_target_preparation()