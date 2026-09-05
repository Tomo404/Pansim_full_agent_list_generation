from pathlib import Path

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

HOUSEHOLD_GENERATION_TARGETS_CSV = Path(
    "../outputs/household_generation/01_household_generation_targets.csv"
)

OUTPUT_FOLDER = Path("../outputs/household_generation")

RESIDUAL_OUTPUT_CSV = OUTPUT_FOLDER / "22_residual_population_without_6plus_households.csv"

RESIDUAL_SUMMARY_CSV = OUTPUT_FOLDER / "23_residual_population_without_6plus_summary.csv"

RESIDUAL_BY_COUNTY_CSV = OUTPUT_FOLDER / "24_residual_population_without_6plus_by_county.csv"


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK
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


def load_household_generation_targets(input_csv: Path) -> pd.DataFrame:
    """
    Beolvassa a household generation target táblát.

    Ez az a fájl, amelyben településszinten már szerepel:
    - household méret kategóriák darabszáma
    - flat népesség
    - 1-5 fős háztartásokban becsült személyek
    - 6+ háztartásokba becsült maradék személyek
    """
    targets = pd.read_csv(input_csv)

    numeric_columns = [
        "households_1_person",
        "households_2_person",
        "households_3_person",
        "households_4_person",
        "households_5_person",
        "households_6plus_person",
        "total_households_from_size_categories",
        "occupied_dwellings_from_flat",
        "population_from_flat_gender_total",
        "population_from_hierarchy",
        "dwelling_count_from_hierarchy",
        "known_people_in_1_to_5_person_households",
        "estimated_people_in_6plus_households",
        "estimated_total_people_after_household_size_estimation",
        "difference_estimated_household_people_vs_population",
    ]

    for column_name in numeric_columns:
        if column_name in targets.columns:
            targets[column_name] = targets[column_name].apply(convert_count_to_integer)

    return targets


# ============================================================
# 3. RESIDUAL TELEPÜLÉSEK KISZŰRÉSE
# ============================================================

def extract_residual_settlements_without_6plus_households(
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kiszűri azokat a településeket, ahol:

    - nincs 6+ fős háztartás
    - mégis pozitív maradék személy jön ki a népességből

    Ezeket érdemes később kézzel / OpenStreetMap alapján ellenőrizni,
    mert lehetnek intézeti népességhez kapcsolódó esetek:
    - börtön
    - kollégium
    - idősotthon
    - gyermekotthon
    - szociális intézmény
    - munkásszálló
    """
    residual_mask = (
        (targets["households_6plus_person"] == 0)
        & (targets["estimated_people_in_6plus_households"] > 0)
    )

    residual_settlements = targets.loc[residual_mask].copy()

    output_columns = [
        "settlement_key",
        "settlement_name_flat",
        "settlement_ksh_code",
        "county_name",
        "county_code",
        "settlement_type",
        "district_code",
        "district_name",
        "population_from_flat_gender_total",
        "population_from_hierarchy",
        "households_1_person",
        "households_2_person",
        "households_3_person",
        "households_4_person",
        "households_5_person",
        "households_6plus_person",
        "total_households_from_size_categories",
        "known_people_in_1_to_5_person_households",
        "estimated_people_in_6plus_households",
        "occupied_dwellings_from_flat",
        "dwelling_count_from_hierarchy",
        "difference_households_vs_occupied_dwellings",
        "difference_households_vs_hierarchy_dwelling_count",
    ]

    existing_output_columns = [
        column_name
        for column_name in output_columns
        if column_name in residual_settlements.columns
    ]

    residual_settlements = residual_settlements[existing_output_columns].copy()

    residual_settlements = residual_settlements.sort_values(
        ["estimated_people_in_6plus_households", "population_from_flat_gender_total"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return residual_settlements


def create_residual_summary(
    targets: pd.DataFrame,
    residual_settlements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Országos összefoglalót készít a residual problémáról.
    """
    summary_rows = []

    summary_rows.append({
        "metric": "total_settlements_in_household_targets",
        "value": len(targets),
        "note": "Települések száma a household target táblában.",
    })

    summary_rows.append({
        "metric": "settlements_with_positive_residual_and_no_6plus_households",
        "value": len(residual_settlements),
        "note": "Olyan települések száma, ahol nincs 6+ háztartás, de van pozitív residual személy.",
    })

    summary_rows.append({
        "metric": "total_residual_population_without_6plus_households",
        "value": residual_settlements["estimated_people_in_6plus_households"].sum(),
        "note": "Ezek adják a jelenlegi household-agent generálásból kimaradó residual népességet.",
    })

    summary_rows.append({
        "metric": "total_flat_population",
        "value": targets["population_from_flat_gender_total"].sum(),
        "note": "Flat Férfi + Nő alapján számolt teljes népesség.",
    })

    summary_rows.append({
        "metric": "residual_share_of_flat_population",
        "value": (
            residual_settlements["estimated_people_in_6plus_households"].sum()
            / targets["population_from_flat_gender_total"].sum()
        ),
        "note": "A residual aránya a teljes flat népességhez képest.",
    })

    summary_rows.append({
        "metric": "generated_agent_population_without_residual",
        "value": (
            targets["population_from_flat_gender_total"].sum()
            - residual_settlements["estimated_people_in_6plus_households"].sum()
        ),
        "note": "Ennyi agent jön ki, ha a residualt még nem kezeljük külön.",
    })

    summary = pd.DataFrame(summary_rows)

    return summary


def create_residual_by_county_table(
    residual_settlements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vármegye szerinti residual összesítést készít.

    Ez segít priorizálni, hol érdemes először kézzel / OSM alapján keresni.
    """
    residual_by_county = (
        residual_settlements
        .groupby(["county_name"], dropna=False)
        .agg(
            residual_settlement_count=("settlement_key", "count"),
            residual_population_total=("estimated_people_in_6plus_households", "sum"),
            largest_single_settlement_residual=("estimated_people_in_6plus_households", "max"),
        )
        .reset_index()
        .sort_values(
            ["residual_population_total", "residual_settlement_count"],
            ascending=[False, False],
        )
    )

    return residual_by_county


# ============================================================
# 4. FŐ FUTTATÁS
# ============================================================

def run_residual_population_extraction() -> None:
    """
    Fő folyamat:

    1. Beolvassa a household generation target táblát.
    2. Kiszűri azokat a településeket, ahol nincs 6+ household,
       de van pozitív residual személy.
    3. Készít országos és vármegyei összefoglalót.
    4. CSV-kbe menti az eredményeket.
    """
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Household generation target tábla beolvasása...")
    targets = load_household_generation_targets(HOUSEHOLD_GENERATION_TARGETS_CSV)

    print("Residual települések kiszűrése...")
    residual_settlements = extract_residual_settlements_without_6plus_households(
        targets=targets
    )

    print("Residual összefoglaló készítése...")
    residual_summary = create_residual_summary(
        targets=targets,
        residual_settlements=residual_settlements,
    )

    print("Vármegyei residual összesítés készítése...")
    residual_by_county = create_residual_by_county_table(
        residual_settlements=residual_settlements
    )

    print("Outputok mentése...")

    residual_settlements.to_csv(
        RESIDUAL_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    residual_summary.to_csv(
        RESIDUAL_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    residual_by_county.to_csv(
        RESIDUAL_BY_COUNTY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Residual települések száma: {len(residual_settlements):,}")
    print(
        "Residual népesség összesen: "
        f"{residual_settlements['estimated_people_in_6plus_households'].sum():,}"
    )
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


# ============================================================
# 5. PROGRAM INDÍTÁSA
# ============================================================

if __name__ == "__main__":
    run_residual_population_extraction()