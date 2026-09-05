from pathlib import Path
import csv
import re
import unicodedata

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# Itt módosíts, ha máshol vannak a fájljaid.
# ============================================================

FLAT_LONG_TABLE_CSV = Path("../outputs/flat_validation/05_flat_combined_long_format_table.csv")
SETTLEMENT_HIERARCHY_FILE = Path("../data/raw_reference/telepules_hierarchia.xlsx")

OUTPUT_FOLDER = Path("../outputs/household_generation")

# A teljes household fájl kb. 4 millió sort fog tartalmazni.
# Ez nagy fájl, de kezelhető. Ha csak target táblákat akarsz tesztelni,
# állítsd False-ra.
GENERATE_FULL_HOUSEHOLD_FILE = True

HOUSEHOLD_ID_PREFIX = "HH"
DWELLING_ID_PREFIX = "DW"

HOUSEHOLD_SIZE_CATEGORIES = {
    "Egyszemélyes háztartás": 1,
    "Kétszemélyes háztartás": 2,
    "Háromszemélyes háztartás": 3,
    "Négyszemélyes háztartás": 4,
    "Ötszemélyes háztartás": 5,
    "Hat vagy többszemélyes háztartás": 6,
}

SIX_PLUS_HOUSEHOLD_SIZE_CATEGORY = "Hat vagy többszemélyes háztartás"

OCCUPIED_DWELLING_CATEGORY = "Lakott lakás"


# ============================================================
# 2. ÁLTALÁNOS SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    """
    Létrehozza az output mappát, ha még nem létezik.
    """
    output_folder.mkdir(parents=True, exist_ok=True)


def normalize_settlement_name(raw_name: str) -> str:
    """
    Településnevek egységesítése összekapcsoláshoz.

    Példa:
    'Pécs' -> 'pecs'
    'Győr' -> 'gyor'
    'Budapest 13. kerület' -> 'budapest 13 kerulet'
    """
    if pd.isna(raw_name):
        return ""

    normalized = str(raw_name).strip().lower()

    # Különféle írásjelek egységesítése szóközre.
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[\.,;:()\[\]/\\]+", " ", normalized)

    # Többszörös szóközök takarítása.
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized

def normalize_settlement_name_ascii(raw_name: str) -> str:
    """
    Ékezetmentes fallback kulcs.

    Ezt NEM használjuk elsődleges matchingre, csak akkor,
    ha az ékezetes kulcs nem talált párt, és az ékezetmentes kulcs
    mindkét oldalon egyértelmű.
    """
    exact_key = normalize_settlement_name(raw_name)

    ascii_key = unicodedata.normalize("NFKD", exact_key)
    ascii_key = "".join(
        character
        for character in ascii_key
        if not unicodedata.combining(character)
    )

    return ascii_key

def add_safe_settlement_match_key(
    targets: pd.DataFrame,
    settlement_master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Biztonságos settlement matching kulcsot készít.

    Logika:
    1. alapból az ékezetes settlement_key-t használjuk
    2. ha nincs pontos match,
       akkor ékezetmentes fallbacket használunk,
       de csak akkor, ha az adott fallback kulcs mindkét oldalon egyedi
    """
    targets = targets.copy()
    settlement_master = settlement_master.copy()

    targets["settlement_match_key"] = targets["settlement_key"]
    settlement_master["settlement_match_key"] = settlement_master["settlement_key"]

    targets["settlement_key_ascii"] = targets["settlement_key"].apply(
        normalize_settlement_name_ascii
    )
    settlement_master["settlement_key_ascii"] = settlement_master["settlement_key"].apply(
        normalize_settlement_name_ascii
    )

    exact_master_keys = set(settlement_master["settlement_key"])

    needs_fallback = ~targets["settlement_key"].isin(exact_master_keys)

    flat_ascii_counts = targets["settlement_key_ascii"].value_counts()
    master_ascii_counts = settlement_master["settlement_key_ascii"].value_counts()

    unique_ascii_to_master_key = (
        settlement_master[
            settlement_master["settlement_key_ascii"].map(master_ascii_counts) == 1
        ]
        .set_index("settlement_key_ascii")["settlement_key"]
        .to_dict()
    )

    safe_fallback_mask = (
        needs_fallback
        & (targets["settlement_key_ascii"].map(flat_ascii_counts) == 1)
        & (targets["settlement_key_ascii"].map(master_ascii_counts) == 1)
    )

    targets.loc[safe_fallback_mask, "settlement_match_key"] = (
        targets.loc[safe_fallback_mask, "settlement_key_ascii"]
        .map(unique_ascii_to_master_key)
    )

    settlement_master = settlement_master.rename(
        columns={
            "settlement_key": "settlement_key_hierarchy"
        }
    )

    settlement_master = settlement_master.drop(
        columns=["settlement_key_ascii"],
        errors="ignore",
    )

    targets = targets.drop(
        columns=["settlement_key_ascii"],
        errors="ignore",
    )

    return targets, settlement_master

MANUAL_SETTLEMENT_KEY_CORRECTIONS = {
    "kömlo": "kömlő",
    "kömöro": "kömörő",
    "kömlod": "kömlőd",
}


def apply_manual_settlement_key_corrections(settlement_key: str) -> str:
    """
    Kézi javítás olyan településnevekre, ahol a forrásban hiányzik
    vagy eltér a hosszú magyar ékezet.

    Ezeket később bővíthetjük, ha a matching report újabb hibákat mutat.
    """
    return MANUAL_SETTLEMENT_KEY_CORRECTIONS.get(settlement_key, settlement_key)

def convert_count_to_integer(raw_value) -> int:
    """
    Biztonságosan egész számmá alakít egy darabszámot.

    A KSH táblákban néha floatként jönnek be az egész értékek,
    például 12.0 formában.
    """
    if pd.isna(raw_value):
        return 0

    return int(round(float(raw_value)))


# ============================================================
# 3. TELEPÜLÉS MASTER TÁBLA
# ============================================================

def load_settlement_master_table(settlement_hierarchy_file: Path) -> pd.DataFrame:
    """
    Beolvassa a telepules_hierarchia.xlsx fájlt, és settlement master táblát készít.

    Ez lesz a kulcstábla:
    - településnév
    - KSH kód
    - vármegye
    - településtípus
    - népesség
    - lakások száma
    """
    settlement_master = pd.read_excel(
        settlement_hierarchy_file,
        sheet_name="telepules_hierarchia",
        engine="openpyxl",
    )

    needed_columns = [
        "Helység megnevezése",
        "Helység KSH kódja",
        "Helység jogállása",
        "Vármegye megnevezése",
        "Vármegye",
        "Településtípus",
        "Járás kódja",
        "Járás neve",
        "Lakó-népesség",
        "Lakások száma",
    ]

    settlement_master = settlement_master[needed_columns].copy()

    settlement_master = settlement_master.rename(
        columns={
            "Helység megnevezése": "settlement_name_hierarchy",
            "Helység KSH kódja": "settlement_ksh_code",
            "Helység jogállása": "settlement_legal_status",
            "Vármegye megnevezése": "county_name",
            "Vármegye": "county_code",
            "Településtípus": "settlement_type",
            "Járás kódja": "district_code",
            "Járás neve": "district_name",
            "Lakó-népesség": "population_from_hierarchy",
            "Lakások száma": "dwelling_count_from_hierarchy",
        }
    )

    settlement_master["settlement_key"] = settlement_master["settlement_name_hierarchy"].apply(
        normalize_settlement_name
    )

    settlement_master["settlement_key"] = settlement_master["settlement_key"].apply(
        apply_manual_settlement_key_corrections
    )

    # A Budapest összesítő sort kiszűrjük, mert a flat adatokban
    # a 23 kerületet használjuk külön egységként.
    settlement_master = settlement_master[
        settlement_master["settlement_key"] != "budapest"
    ].copy()

    return settlement_master


# ============================================================
# 4. FLAT HOSSZÚ ADATBÁZIS BEOLVASÁSA
# ============================================================

def load_flat_long_table(flat_long_table_csv: Path) -> pd.DataFrame:
    """
    Beolvassa a korábban létrehozott flat long format táblát.

    Elvárt oszlopok:
    - source_file
    - category
    - settlement
    - value
    """
    flat_long_table = pd.read_csv(flat_long_table_csv)

    flat_long_table["settlement_key"] = flat_long_table["settlement"].apply(
        normalize_settlement_name
    )

    flat_long_table["settlement_key"] = flat_long_table["settlement_key"].apply(
        apply_manual_settlement_key_corrections
    )

    # Ez nem valódi település, és jellemzően 0 értékű technikai sor.
    flat_long_table = flat_long_table[
        flat_long_table["settlement_key"] != "budapest kerületre nem bontható adatai"
    ].copy()

    flat_long_table["value"] = pd.to_numeric(
        flat_long_table["value"],
        errors="coerce",
    ).fillna(0)

    return flat_long_table


# ============================================================
# 5. HOUSEHOLD TARGET TÁBLA ELKÉSZÍTÉSE
# ============================================================

def extract_household_size_targets(flat_long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Kiveszi a flat háztartási táblából a háztartásméret-kategóriákat.

    Output:
    settlement_key
    settlement_name_flat
    households_1_person
    households_2_person
    ...
    households_6plus_person
    total_households_from_size_categories
    """
    household_rows = flat_long_table[
        flat_long_table["source_file"].str.contains("haztartasok", case=False, regex=False)
        & flat_long_table["category"].isin(HOUSEHOLD_SIZE_CATEGORIES.keys())
    ].copy()

    household_pivot = household_rows.pivot_table(
        index=["settlement_key", "settlement"],
        columns="category",
        values="value",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    household_pivot = household_pivot.rename(
        columns={
            "settlement": "settlement_name_flat",
            "Egyszemélyes háztartás": "households_1_person",
            "Kétszemélyes háztartás": "households_2_person",
            "Háromszemélyes háztartás": "households_3_person",
            "Négyszemélyes háztartás": "households_4_person",
            "Ötszemélyes háztartás": "households_5_person",
            "Hat vagy többszemélyes háztartás": "households_6plus_person",
        }
    )

    household_count_columns = [
        "households_1_person",
        "households_2_person",
        "households_3_person",
        "households_4_person",
        "households_5_person",
        "households_6plus_person",
    ]

    for column_name in household_count_columns:
        household_pivot[column_name] = household_pivot[column_name].apply(convert_count_to_integer)

    household_pivot["total_households_from_size_categories"] = household_pivot[
        household_count_columns
    ].sum(axis=1)

    return household_pivot


def extract_occupied_dwelling_targets(flat_long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Kiveszi a flat lakás táblából a 'Lakott lakás' sort.

    Output:
    settlement_key
    occupied_dwellings_from_flat
    """
    dwelling_rows = flat_long_table[
        flat_long_table["source_file"].str.contains("lakasok", case=False, regex=False)
        & (flat_long_table["category"] == OCCUPIED_DWELLING_CATEGORY)
    ].copy()

    dwelling_targets = (
        dwelling_rows
        .groupby(["settlement_key", "settlement"], dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "settlement": "settlement_name_flat_dwelling",
                "value": "occupied_dwellings_from_flat",
            }
        )
    )

    dwelling_targets["occupied_dwellings_from_flat"] = dwelling_targets[
        "occupied_dwellings_from_flat"
    ].apply(convert_count_to_integer)

    return dwelling_targets

def extract_population_targets_from_flat_table(flat_long_table: pd.DataFrame) -> pd.DataFrame:
    """
    Kiveszi a flat népességi táblából a településenkénti teljes népességet.

    Első körben a Férfi + Nő kategóriákat használjuk, mert ezek adják vissza
    legstabilabban a teljes népességet településszinten.
    """
    population_rows = flat_long_table[
        flat_long_table["source_file"].str.contains("nepesseg", case=False, regex=False)
        & flat_long_table["category"].isin(["Férfi", "Nő"])
    ].copy()

    population_targets = (
        population_rows
        .groupby(["settlement_key", "settlement"], dropna=False)["value"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "settlement": "settlement_name_flat_population",
                "value": "population_from_flat_gender_total",
            }
        )
    )

    population_targets["population_from_flat_gender_total"] = population_targets[
        "population_from_flat_gender_total"
    ].apply(convert_count_to_integer)

    return population_targets

def build_household_generation_targets(
    settlement_master: pd.DataFrame,
    household_size_targets: pd.DataFrame,
    occupied_dwelling_targets: pd.DataFrame,
    population_targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Összerakja a végső településszintű household target táblát.

    Ebben már együtt van:
    - települési master adat
    - háztartásméret-kategóriák
    - lakott lakások száma
    """
    targets = household_size_targets.merge(
        occupied_dwelling_targets[
            ["settlement_key", "occupied_dwellings_from_flat"]
        ],
        on="settlement_key",
        how="left",
    )

    targets = targets.merge(
        population_targets[
            ["settlement_key", "population_from_flat_gender_total"]
        ],
        on="settlement_key",
        how="left",
    )

    targets, settlement_master_for_matching = add_safe_settlement_match_key(
        targets=targets,
        settlement_master=settlement_master,
    )

    targets = targets.merge(
        settlement_master_for_matching,
        on="settlement_match_key",
        how="left",
        indicator="settlement_master_match_status",
    )

    count_columns = [
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
    ]

    for column_name in count_columns:
        if column_name in targets.columns:
            targets[column_name] = pd.to_numeric(
                targets[column_name],
                errors="coerce",
            ).fillna(0).apply(convert_count_to_integer)

    targets["difference_households_vs_occupied_dwellings"] = (
        targets["total_households_from_size_categories"]
        - targets["occupied_dwellings_from_flat"]
    )

    targets["difference_households_vs_hierarchy_dwelling_count"] = (
        targets["total_households_from_size_categories"]
        - targets["dwelling_count_from_hierarchy"]
    )

    targets["known_people_in_1_to_5_person_households"] = (
        1 * targets["households_1_person"]
        + 2 * targets["households_2_person"]
        + 3 * targets["households_3_person"]
        + 4 * targets["households_4_person"]
        + 5 * targets["households_5_person"]
    )

    targets["estimated_people_in_6plus_households"] = (
        targets["population_from_flat_gender_total"]
        - targets["known_people_in_1_to_5_person_households"]
    )

    targets["estimated_average_size_6plus_households"] = 0.0

    has_6plus_households = targets["households_6plus_person"] > 0

    targets.loc[
        has_6plus_households,
        "estimated_average_size_6plus_households"
    ] = (
        targets.loc[has_6plus_households, "estimated_people_in_6plus_households"]
        / targets.loc[has_6plus_households, "households_6plus_person"]
    )

    targets["generated_size_for_6plus_households"] = (
        targets["estimated_average_size_6plus_households"]
        .round()
        .astype(int)
    )

    targets.loc[
        has_6plus_households,
        "generated_size_for_6plus_households"
    ] = targets.loc[
        has_6plus_households,
        "generated_size_for_6plus_households"
    ].clip(lower=6)

    targets.loc[
        ~has_6plus_households,
        "generated_size_for_6plus_households"
    ] = 0

    targets["estimated_total_people_after_household_size_estimation"] = (
        targets["known_people_in_1_to_5_person_households"]
        + targets["generated_size_for_6plus_households"] * targets["households_6plus_person"]
    )

    targets["base_size_for_6plus_households"] = 0
    targets["extra_people_to_distribute_among_6plus_households"] = 0

    targets.loc[
        has_6plus_households,
        "base_size_for_6plus_households"
    ] = (
        targets.loc[has_6plus_households, "estimated_people_in_6plus_households"]
        // targets.loc[has_6plus_households, "households_6plus_person"]
    ).astype(int)

    targets.loc[
        has_6plus_households,
        "extra_people_to_distribute_among_6plus_households"
    ] = (
        targets.loc[has_6plus_households, "estimated_people_in_6plus_households"]
        % targets.loc[has_6plus_households, "households_6plus_person"]
    ).astype(int)

    targets["difference_estimated_household_people_vs_population"] = (
        targets["estimated_total_people_after_household_size_estimation"]
        - targets["population_from_flat_gender_total"]
    )

    targets = targets.sort_values(
        ["county_name", "settlement_name_flat"],
        na_position="last",
    ).reset_index(drop=True)

    return targets


# ============================================================
# 6. MATCHING ÉS VALIDÁCIÓS RIPORTOK
# ============================================================

def create_settlement_name_matching_report(
    settlement_master: pd.DataFrame,
    household_size_targets: pd.DataFrame,
    occupied_dwelling_targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Megnézi, hogy a flat és hierarchy településnevek összeköthetők-e.

    Fontos:
    Itt ugyanazt a biztonságos fallback matching logikát használjuk,
    mint a fő household target építésnél.
    """
    flat_settlements = household_size_targets[
        ["settlement_key", "settlement_name_flat"]
    ].drop_duplicates()

    flat_settlements = flat_settlements.merge(
        occupied_dwelling_targets[
            ["settlement_key", "settlement_name_flat_dwelling"]
        ].drop_duplicates(),
        on="settlement_key",
        how="outer",
    )

    flat_settlements, settlement_master_for_matching = add_safe_settlement_match_key(
        targets=flat_settlements,
        settlement_master=settlement_master,
    )

    matching_report = flat_settlements.merge(
        settlement_master_for_matching[
            [
                "settlement_match_key",
                "settlement_key_hierarchy",
                "settlement_name_hierarchy",
                "settlement_ksh_code",
                "county_name",
                "settlement_type",
            ]
        ],
        on="settlement_match_key",
        how="outer",
        indicator="match_status",
    )

    matching_report = matching_report.sort_values(
        ["match_status", "settlement_match_key"]
    ).reset_index(drop=True)

    return matching_report


def create_household_target_validation(targets: pd.DataFrame) -> pd.DataFrame:
    """
    Országos validációs összegző táblát készít.
    """
    validation_rows = []

    validation_rows.append({
        "check_name": "Háztartásméret-kategóriák összege",
        "observed_total": targets["total_households_from_size_categories"].sum(),
        "comparison_total": None,
        "difference": None,
        "note": "Ez lesz a generált household objektumok darabszáma.",
    })

    validation_rows.append({
        "check_name": "Lakott lakás flat tábla alapján",
        "observed_total": targets["occupied_dwellings_from_flat"].sum(),
        "comparison_total": None,
        "difference": None,
        "note": "Validációs összevetéshez használt lakott lakás érték.",
    })

    validation_rows.append({
        "check_name": "Háztartásméret összeg - lakott lakás",
        "observed_total": targets["total_households_from_size_categories"].sum(),
        "comparison_total": targets["occupied_dwellings_from_flat"].sum(),
        "difference": (
            targets["total_households_from_size_categories"].sum()
            - targets["occupied_dwellings_from_flat"].sum()
        ),
        "note": "Kicsi eltérés elfogadható; forrásonként eltérhet a definíció.",
    })

    validation_rows.append({
        "check_name": "Településhierarchia lakások száma",
        "observed_total": targets["dwelling_count_from_hierarchy"].sum(),
        "comparison_total": None,
        "difference": None,
        "note": "Ez nem feltétlenül csak lakott lakás, ezért óvatos validációs pont.",
    })

    validation_rows.append({
        "check_name": "Háztartásméret összeg - hierarchy lakások száma",
        "observed_total": targets["total_households_from_size_categories"].sum(),
        "comparison_total": targets["dwelling_count_from_hierarchy"].sum(),
        "difference": (
            targets["total_households_from_size_categories"].sum()
            - targets["dwelling_count_from_hierarchy"].sum()
        ),
        "note": "A hierarchy lakásszám más definíciójú lehet, ezért csak tájékoztató.",
    })

    validation_rows.append({
        "check_name": "Flat népesség Férfi + Nő alapján",
        "observed_total": targets["population_from_flat_gender_total"].sum(),
        "comparison_total": None,
        "difference": None,
        "note": "Ezt használjuk a 6+ háztartások átlagos méretének becsléséhez.",
    })

    validation_rows.append({
        "check_name": "1-5 fős háztartásokban becsült személyek",
        "observed_total": targets["known_people_in_1_to_5_person_households"].sum(),
        "comparison_total": None,
        "difference": None,
        "note": "1*h1 + 2*h2 + 3*h3 + 4*h4 + 5*h5.",
    })

    validation_rows.append({
        "check_name": "6+ háztartásokba becsült maradék személyek",
        "observed_total": targets["estimated_people_in_6plus_households"].sum(),
        "comparison_total": None,
        "difference": None,
        "note": "Flat népesség - 1-5 fős háztartásokban becsült személyek.",
    })

    validation_rows.append({
        "check_name": "Kerekített household méretekkel becsült teljes személyszám",
        "observed_total": targets["estimated_total_people_after_household_size_estimation"].sum(),
        "comparison_total": targets["population_from_flat_gender_total"].sum(),
        "difference": (
            targets["estimated_total_people_after_household_size_estimation"].sum()
            - targets["population_from_flat_gender_total"].sum()
        ),
        "note": "A 6+ háztartások kerekített mérete miatt lehet eltérés.",
    })

    validation_rows.append({
        "check_name": "Pontos 6+ elosztással becsült teljes személyszám",
        "observed_total": (
            targets["known_people_in_1_to_5_person_households"].sum()
            + targets["estimated_people_in_6plus_households"].sum()
        ),
        "comparison_total": targets["population_from_flat_gender_total"].sum(),
        "difference": (
            targets["known_people_in_1_to_5_person_households"].sum()
            + targets["estimated_people_in_6plus_households"].sum()
            - targets["population_from_flat_gender_total"].sum()
        ),
        "note": "A 6+ háztartásokon belüli maradék személyek elosztásával ennek 0 körül kell lennie.",
    })

    no_6plus_household_mask = targets["households_6plus_person"] == 0

    validation_rows.append({
        "check_name": "Maradék személyek olyan településeken, ahol nincs 6+ háztartás",
        "observed_total": targets.loc[
            no_6plus_household_mask,
            "estimated_people_in_6plus_households"
        ].sum(),
        "comparison_total": None,
        "difference": None,
        "note": "Ezeket nem lehet 6+ háztartásokba osztani, mert az adott településen nincs ilyen háztartás. Később külön kezelendő residual.",
    })

    validation_rows.append({
        "check_name": "Pontos 6+ elosztással ténylegesen kiosztható személyszám",
        "observed_total": (
            targets["known_people_in_1_to_5_person_households"].sum()
            + (
                targets["base_size_for_6plus_households"]
                * targets["households_6plus_person"]
            ).sum()
            + targets["extra_people_to_distribute_among_6plus_households"].sum()
        ),
        "comparison_total": targets["population_from_flat_gender_total"].sum(),
        "difference": (
            targets["known_people_in_1_to_5_person_households"].sum()
            + (
                targets["base_size_for_6plus_households"]
                * targets["households_6plus_person"]
            ).sum()
            + targets["extra_people_to_distribute_among_6plus_households"].sum()
            - targets["population_from_flat_gender_total"].sum()
        ),
        "note": "Ez a ténylegesen kiosztható személykapacitás. Ha vannak települések 6+ háztartás nélkül, maradhat kis negatív eltérés.",
    })
    
    validation_table = pd.DataFrame(validation_rows)

    return validation_table


def create_household_tensor_by_county_settlement_type(
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tenzorszerű összefoglaló:
    vármegye × településtípus × household_size_category.

    Ez hasonló logikájú, mint a korábbi workplace tensor,
    csak itt TEÁOR / company size helyett háztartásméret van.
    """
    household_size_columns = {
        "households_1_person": "1_person",
        "households_2_person": "2_person",
        "households_3_person": "3_person",
        "households_4_person": "4_person",
        "households_5_person": "5_person",
        "households_6plus_person": "6plus_person_capped_to_6",
    }

    tensor_long_rows = []

    for _, row in targets.iterrows():
        for source_column, size_label in household_size_columns.items():
            tensor_long_rows.append({
                "county_name": row["county_name"],
                "settlement_type": row["settlement_type"],
                "household_size_category": size_label,
                "household_count": row[source_column],
            })

    tensor_long = pd.DataFrame(tensor_long_rows)

    tensor_summary = (
        tensor_long
        .groupby(
            ["county_name", "settlement_type", "household_size_category"],
            dropna=False,
        )["household_count"]
        .sum()
        .reset_index()
        .sort_values(["county_name", "settlement_type", "household_size_category"])
    )

    return tensor_summary


# ============================================================
# 7. KONKRÉT HOUSEHOLDOK GENERÁLÁSA
# ============================================================

def generate_households_to_csv(
    targets: pd.DataFrame,
    output_csv_path: Path,
) -> None:
    """
    Legyártja a konkrét háztartás/lakás objektumokat CSV-be.

    Minden sor egy generált household.

    FIGYELEM:
    Ez kb. 4 millió sort ír ki, tehát eltarthat pár percig.
    """
    household_size_columns = [
        ("households_1_person", "Egyszemélyes háztartás", 1, False),
        ("households_2_person", "Kétszemélyes háztartás", 2, False),
        ("households_3_person", "Háromszemélyes háztartás", 3, False),
        ("households_4_person", "Négyszemélyes háztartás", 4, False),
        ("households_5_person", "Ötszemélyes háztartás", 5, False),
        ("households_6plus_person", "Hat vagy többszemélyes háztartás", None, True),
    ]

    output_columns = [
        "household_id",
        "dwelling_id",
        "settlement_key",
        "settlement_name",
        "settlement_ksh_code",
        "county_name",
        "county_code",
        "settlement_type",
        "district_code",
        "district_name",
        "household_size_category",
        "household_size_numeric",
        "estimated_average_household_size",
        "household_size_is_estimated",
    ]

    household_counter = 1

    with output_csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_columns)
        writer.writeheader()

        for _, settlement_row in targets.iterrows():
            for count_column, size_category, size_numeric, is_capped in household_size_columns:
                household_count = convert_count_to_integer(settlement_row[count_column])

                if size_numeric is None:
                    base_size_for_6plus = convert_count_to_integer(
                        settlement_row["base_size_for_6plus_households"]
                    )
                    extra_people_to_distribute = convert_count_to_integer(
                        settlement_row["extra_people_to_distribute_among_6plus_households"]
                    )
                    estimated_average_size = settlement_row[
                        "estimated_average_size_6plus_households"
                    ]
                else:
                    base_size_for_6plus = size_numeric
                    extra_people_to_distribute = 0
                    estimated_average_size = size_numeric

                for household_index_within_category in range(household_count):
                    if size_numeric is None:
                        actual_size_numeric = base_size_for_6plus

                        if household_index_within_category < extra_people_to_distribute:
                            actual_size_numeric += 1
                    else:
                        actual_size_numeric = size_numeric

                    household_id = f"{HOUSEHOLD_ID_PREFIX}_{household_counter:010d}"
                    dwelling_id = f"{DWELLING_ID_PREFIX}_{household_counter:010d}"

                    writer.writerow({
                        "household_id": household_id,
                        "dwelling_id": dwelling_id,
                        "settlement_key": settlement_row["settlement_key"],
                        "settlement_name": settlement_row["settlement_name_flat"],
                        "settlement_ksh_code": settlement_row["settlement_ksh_code"],
                        "county_name": settlement_row["county_name"],
                        "county_code": settlement_row["county_code"],
                        "settlement_type": settlement_row["settlement_type"],
                        "district_code": settlement_row["district_code"],
                        "district_name": settlement_row["district_name"],
                        "household_size_category": size_category,
                        "household_size_numeric": actual_size_numeric,
                        "estimated_average_household_size": estimated_average_size,
                        "household_size_is_estimated": is_capped,
                    })

                    household_counter += 1

    print(f"Generált household sorok száma: {household_counter - 1}")


# ============================================================
# 8. FŐ FUTTATÁSI FÜGGVÉNY
# ============================================================

def run_household_generation() -> None:
    """
    Fő folyamat:

    1. Beolvassa a settlement master táblát.
    2. Beolvassa a flat long format adatbázist.
    3. Elkészíti a household target táblát.
    4. Validációs riportokat ír.
    5. Tenzorszerű összefoglalót készít.
    6. Legenerálja a konkrét household objektumokat.
    """
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Settlement master tábla beolvasása...")
    settlement_master = load_settlement_master_table(SETTLEMENT_HIERARCHY_FILE)

    print("Flat long format adatbázis beolvasása...")
    flat_long_table = load_flat_long_table(FLAT_LONG_TABLE_CSV)

    print("Háztartásméret targetek kinyerése...")
    household_size_targets = extract_household_size_targets(flat_long_table)

    print("Lakott lakás targetek kinyerése...")
    occupied_dwelling_targets = extract_occupied_dwelling_targets(flat_long_table)

    print("Települési népesség targetek kinyerése...")
    population_targets = extract_population_targets_from_flat_table(flat_long_table)

    print("Települési household target tábla építése...")
    household_generation_targets = build_household_generation_targets(
        settlement_master=settlement_master,
        household_size_targets=household_size_targets,
        occupied_dwelling_targets=occupied_dwelling_targets,
        population_targets=population_targets,
    )

    print("Settlement matching riport készítése...")
    settlement_matching_report = create_settlement_name_matching_report(
        settlement_master=settlement_master,
        household_size_targets=household_size_targets,
        occupied_dwelling_targets=occupied_dwelling_targets,
    )

    print("Validációs összefoglaló készítése...")
    household_target_validation = create_household_target_validation(
        targets=household_generation_targets,
    )

    print("Tenzorszerű household összefoglaló készítése...")
    household_tensor = create_household_tensor_by_county_settlement_type(
        targets=household_generation_targets,
    )

    # --------------------------------------------------------
    # Output fájlok mentése
    # --------------------------------------------------------

    household_generation_targets.to_csv(
        OUTPUT_FOLDER / "01_household_generation_targets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    settlement_matching_report.to_csv(
        OUTPUT_FOLDER / "02_settlement_name_matching_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    household_target_validation.to_csv(
        OUTPUT_FOLDER / "03_household_target_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    household_tensor.to_csv(
        OUTPUT_FOLDER / "04_household_tensor_county_settlement_type_size.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if GENERATE_FULL_HOUSEHOLD_FILE:
        print("Konkrét household objektumok generálása...")
        generate_households_to_csv(
            targets=household_generation_targets,
            output_csv_path=OUTPUT_FOLDER / "05_generated_households.csv",
        )
    else:
        print("A teljes household fájl generálása ki van kapcsolva.")

    print()
    print("Kész.")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


# ============================================================
# 9. PROGRAM INDÍTÁSA
# Ezt ne módosítsd.
# ============================================================

if __name__ == "__main__":
    run_household_generation()