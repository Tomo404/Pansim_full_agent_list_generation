from pathlib import Path
import re

import pandas as pd


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

HEALTH_XLSX = Path(
    "../data/raw_hier_tables/hier_Egeszseg_allapot_varmegyenkent_telepulestipusonkent.xlsx"
)

OUTPUT_FOLDER = Path(
    "../outputs/agent_attribute_generation/health_status_assignment"
)

HEALTH_LONG_CSV = OUTPUT_FOLDER / "121_health_status_hier_long.csv"
HEALTH_BLOCK_INVENTORY_CSV = OUTPUT_FOLDER / "122_health_status_block_inventory.csv"
HEALTH_QUESTION_TOTALS_CSV = OUTPUT_FOLDER / "123_health_status_question_totals.csv"
HEALTH_DIMENSION_TOTALS_CSV = OUTPUT_FOLDER / "124_health_status_dimension_totals.csv"
HEALTH_DUPLICATE_LABEL_CHECK_CSV = OUTPUT_FOLDER / "125_health_status_duplicate_label_check.csv"
HEALTH_PROFILE_SUMMARY_CSV = OUTPUT_FOLDER / "126_health_status_profile_summary.csv"


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


def normalize_county_name(raw_county_name: str) -> str:
    county_name = normalize_text(raw_county_name)

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


def normalize_settlement_type(raw_settlement_type: str) -> str:
    settlement_type = normalize_text(raw_settlement_type)

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


def convert_count_to_integer(raw_value) -> int:
    if raw_value is None:
        return 0

    try:
        if pd.isna(raw_value):
            return 0
    except TypeError:
        pass

    text = str(raw_value).strip()

    if text == "" or text == "…":
        return 0

    if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")

    return int(round(float(text)))


def classify_health_question(response_category: str) -> str:
    category = normalize_text(response_category)

    disability_categories = {
        "Fogyatékossága van vagy súlyosan korlátozott",
        "Nincs fogyatékossága és nem súlyosan korlátozott",
        "Nem válaszolt a fogyatékossági és a korlátozottsági kérdésekre",
    }

    chronic_disease_categories = {
        "Van tartós betegsége",
        "Nincs tartós betegsége",
        "Nem válaszolt a tartós betegségre vonatkozó kérdésre",
    }

    limitation_categories = {
        "Nem korlátozott",
        "Mérsékelten korlátozott",
        "Súlyosan korlátozott",
        "Nem válaszolt a korlátozottsági kérdésekre",
    }

    if category in disability_categories:
        return "disability_or_severe_limitation"

    if category in chronic_disease_categories:
        return "chronic_disease"

    if category in limitation_categories:
        return "limitation_level"

    return "UNKNOWN_HEALTH_QUESTION"


def classify_dimension_group(block_label: str, block_start_row_index: int) -> str:
    """
    A tábla blokkjai 10 sorosak.
    A két '15 évesnél fiatalabb személy' blokkot a pozíció alapján különítjük el:
    - education blokkban: sor 162 körül
    - economic_activity blokkban: sor 212 körül
    """

    label = normalize_text(block_label)

    sex_labels = {
        "Férfi",
        "Nő",
    }

    age_labels = {
        "5–14 éves",
        "15–24 éves",
        "25–34 éves",
        "35–44 éves",
        "45–54 éves",
        "55–64 éves",
        "65–74 éves",
        "75–84 éves",
        "85 éves és idősebb",
    }

    education_labels = {
        "Általános iskola 8. évfolyamnál alacsonyabb",
        "Általános iskola 8. évfolyam",
        "Középfokú iskola érettségi nélkül, szakmai oklevéllel",
        "Érettségi",
        "Egyetem, főiskola stb. oklevéllel",
    }

    economic_activity_labels = {
        "Foglalkoztatott",
        "Munkanélküli",
        "Ellátásban részesülő inaktív",
        "Eltartott",
    }

    if label in sex_labels:
        return "sex"

    if label in age_labels:
        return "age_group"

    if label in education_labels:
        return "education"

    if label in economic_activity_labels:
        return "economic_activity"

    if label == "15 évesnél fiatalabb személy":
        if block_start_row_index < 172:
            return "education"
        return "economic_activity"

    return "UNKNOWN_DIMENSION"


def find_block_start_rows(raw_table: pd.DataFrame) -> list[int]:
    block_start_rows = []

    for row_index in range(len(raw_table)):
        if normalize_text(raw_table.iloc[row_index, 0]) != "":
            block_start_rows.append(row_index)

    return block_start_rows


# ============================================================
# 3. LONG FORMÁTUM
# ============================================================

def convert_health_table_to_long(health_xlsx: Path) -> pd.DataFrame:
    raw_table = pd.read_excel(
        health_xlsx,
        sheet_name="Adattábla",
        header=None,
    )

    first_value_column_index = 2

    county_values = raw_table.iloc[0, first_value_column_index:].ffill()
    settlement_type_values = raw_table.iloc[1, first_value_column_index:].ffill()

    geography = pd.DataFrame({
        "value_column_index": raw_table.columns[first_value_column_index:],
        "county_name": county_values.values,
        "settlement_type": settlement_type_values.values,
    })

    geography["county_name"] = geography["county_name"].apply(normalize_county_name)
    geography["settlement_type"] = geography["settlement_type"].apply(normalize_settlement_type)

    block_start_rows = find_block_start_rows(raw_table)

    long_rows = []

    for block_start_row_index in block_start_rows:
        dimension_category = normalize_text(raw_table.iloc[block_start_row_index, 0])
        dimension_group = classify_dimension_group(
            block_label=dimension_category,
            block_start_row_index=block_start_row_index,
        )

        # Minden blokk 10 egészségügyi válaszkategória.
        for offset in range(10):
            row_index = block_start_row_index + offset

            if row_index >= len(raw_table):
                continue

            health_response_category = normalize_text(raw_table.iloc[row_index, 1])

            if health_response_category == "":
                continue

            health_question = classify_health_question(
                response_category=health_response_category
            )

            for _, geography_row in geography.iterrows():
                value_column_index = geography_row["value_column_index"]
                count = convert_count_to_integer(raw_table.iloc[row_index, value_column_index])

                if count == 0:
                    continue

                long_rows.append({
                    "county_name": geography_row["county_name"],
                    "settlement_type": geography_row["settlement_type"],
                    "dimension_group": dimension_group,
                    "dimension_category": dimension_category,
                    "health_question": health_question,
                    "health_response_category": health_response_category,
                    "health_count": count,
                    "source_excel_row": row_index + 1,
                })

    long_df = pd.DataFrame(long_rows)

    grouped = (
        long_df
        .groupby(
            [
                "county_name",
                "settlement_type",
                "dimension_group",
                "dimension_category",
                "health_question",
                "health_response_category",
            ],
            dropna=False,
        )["health_count"]
        .sum()
        .reset_index()
    )

    return grouped


# ============================================================
# 4. PROFIL OUTPUTOK
# ============================================================

def create_block_inventory(health_long: pd.DataFrame) -> pd.DataFrame:
    return (
        health_long
        .groupby(
            [
                "dimension_group",
                "dimension_category",
                "health_question",
            ],
            dropna=False,
        )["health_count"]
        .sum()
        .reset_index()
        .sort_values(
            [
                "dimension_group",
                "dimension_category",
                "health_question",
            ]
        )
    )


def create_question_totals(health_long: pd.DataFrame) -> pd.DataFrame:
    return (
        health_long
        .groupby(
            [
                "dimension_group",
                "health_question",
                "health_response_category",
            ],
            dropna=False,
        )["health_count"]
        .sum()
        .reset_index()
        .sort_values(
            [
                "dimension_group",
                "health_question",
                "health_count",
            ],
            ascending=[True, True, False],
        )
    )


def create_dimension_totals(health_long: pd.DataFrame) -> pd.DataFrame:
    """
    Itt ellenőrizzük, hogy egy dimenzió-kategória egy health_questionön belül
    milyen népességösszeget ad.

    Mivel ugyanazt a népességet három külön kérdés szerint is bontja a tábla,
    a teljes 10 válaszkategória összege nem population, hanem kb. 3 × population.
    Ezért question szinten kell ellenőrizni.
    """

    return (
        health_long
        .groupby(
            [
                "dimension_group",
                "dimension_category",
                "health_question",
            ],
            dropna=False,
        )["health_count"]
        .sum()
        .reset_index()
        .sort_values(
            [
                "dimension_group",
                "dimension_category",
                "health_question",
            ]
        )
    )


def create_duplicate_label_check(health_long: pd.DataFrame) -> pd.DataFrame:
    return (
        health_long
        .groupby(
            [
                "dimension_category",
                "dimension_group",
                "health_question",
            ],
            dropna=False,
        )["health_count"]
        .sum()
        .reset_index()
        .sort_values(
            [
                "dimension_category",
                "dimension_group",
                "health_question",
            ]
        )
    )


def create_profile_summary(health_long: pd.DataFrame) -> pd.DataFrame:
    total_raw_health_count = int(health_long["health_count"].sum())

    dimension_question_totals = (
        health_long
        .groupby(["dimension_group", "health_question"], dropna=False)["health_count"]
        .sum()
        .reset_index()
    )

    rows = [
        {
            "metric": "health_long_rows",
            "value": len(health_long),
            "note": "Long formátumú nem nulla cellák száma.",
        },
        {
            "metric": "raw_health_count_sum",
            "value": total_raw_health_count,
            "note": "Nyers összeg minden dimenzióra és mindhárom health kérdésre együtt. Ez nem népességszám.",
        },
        {
            "metric": "unique_dimension_groups",
            "value": health_long["dimension_group"].nunique(),
            "note": "Fő bontási dimenziók száma.",
        },
        {
            "metric": "unique_health_questions",
            "value": health_long["health_question"].nunique(),
            "note": "Külön health kérdések száma.",
        },
        {
            "metric": "unique_health_response_categories",
            "value": health_long["health_response_category"].nunique(),
            "note": "Health válaszkategóriák száma összesen.",
        },
    ]

    for _, row in dimension_question_totals.iterrows():
        rows.append({
            "metric": f"total__{row['dimension_group']}__{row['health_question']}",
            "value": int(row["health_count"]),
            "note": "Egy fő dimenzió és egy health kérdés összesített targetje.",
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. FŐ FUTTATÁS
# ============================================================

def run_health_status_profile() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Egészségi állapot tábla long formátumra alakítása...")
    health_long = convert_health_table_to_long(
        health_xlsx=HEALTH_XLSX
    )

    print("Outputok írása...")

    health_long.to_csv(
        HEALTH_LONG_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    create_block_inventory(
        health_long=health_long
    ).to_csv(
        HEALTH_BLOCK_INVENTORY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    create_question_totals(
        health_long=health_long
    ).to_csv(
        HEALTH_QUESTION_TOTALS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    create_dimension_totals(
        health_long=health_long
    ).to_csv(
        HEALTH_DIMENSION_TOTALS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    create_duplicate_label_check(
        health_long=health_long
    ).to_csv(
        HEALTH_DUPLICATE_LABEL_CHECK_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    create_profile_summary(
        health_long=health_long
    ).to_csv(
        HEALTH_PROFILE_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Long rows: {len(health_long):,}")
    print(f"Raw health count sum: {int(health_long['health_count'].sum()):,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_health_status_profile()