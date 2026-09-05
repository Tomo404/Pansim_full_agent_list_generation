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

OUTPUT_FOLDER = Path("../outputs/agent_attribute_generation/workplace_matching_profile")

OLD_WORKER_FILE_PROFILE_CSV = OUTPUT_FOLDER / "76_old_worker_file_profile.csv"
OLD_WORKER_COLUMNS_CSV = OUTPUT_FOLDER / "77_old_worker_columns.csv"
OLD_WORKER_COUNTS_BY_MATCH_KEY_CSV = OUTPUT_FOLDER / "78_old_worker_counts_by_county_gender_age_education.csv"
CURRENT_EMPLOYED_COUNTS_BY_MATCH_KEY_CSV = OUTPUT_FOLDER / "79_current_employed_counts_by_county_gender_age_education.csv"
MATCHING_FEASIBILITY_COMPARISON_CSV = OUTPUT_FOLDER / "80_worker_matching_feasibility_comparison.csv"
MATCHING_PROFILE_SUMMARY_CSV = OUTPUT_FOLDER / "81_worker_matching_profile_summary.csv"
MATCHING_METHOD_NOTES_CSV = OUTPUT_FOLDER / "82_worker_matching_profile_method_notes.csv"


# Teszthez állítható pl. 100_000-re.
# Teljes futtatáshoz None.
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


def normalize_for_key(raw_value) -> str:
    """
    Matching kulcshoz egyszerűsített szöveg.
    Ékezeteket most még megtartjuk, mert a magyar település/megye nevekben fontosak lehetnek.
    """
    text = normalize_text(raw_value).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[\"']", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


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


def normalize_gender_old(raw_gender: str) -> str:
    gender = normalize_text(raw_gender).lower()

    if gender in ["férfi", "ferfi", "male"]:
        return "male"

    if gender in ["nő", "no", "female"]:
        return "female"

    return gender


def normalize_sex_current(raw_sex: str) -> str:
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


def map_exact_age_to_worker_age_group(exact_age: int) -> str:
    """
    A régi worker fájlban látható 5 éves korcsoportokhoz igazít:
    15–19, 20–24, ..., 70 éves és idősebb.
    """
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


def make_match_key(
    county_name: str,
    sex: str,
    age_group: str,
    education: str,
) -> tuple[str, str, str, str]:
    return (
        normalize_county_name(county_name),
        sex,
        normalize_age_group(age_group),
        normalize_education(education),
    )


# ============================================================
# 3. FÁJL PROFIL ÉS OSZLOPOK
# ============================================================

def inspect_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames)


def profile_csv_file(csv_path: Path, file_label: str) -> dict:
    header = inspect_csv_header(csv_path)

    row_count = 0
    nonempty_counts = defaultdict(int)
    sample_values = defaultdict(list)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and row_count >= MAX_ROWS_TO_PROCESS:
                break

            for column in header:
                value = normalize_text(row.get(column, ""))

                if value != "":
                    nonempty_counts[column] += 1

                    if len(sample_values[column]) < 5 and value not in sample_values[column]:
                        sample_values[column].append(value)

            row_count += 1

            if row_count % 1_000_000 == 0:
                print(f"{file_label} profilozott sorok: {row_count:,}")

    profile_rows = [{
        "file_label": file_label,
        "file_path": str(csv_path),
        "processed_row_count": row_count,
        "column_count": len(header),
        "max_rows_to_process": MAX_ROWS_TO_PROCESS if MAX_ROWS_TO_PROCESS is not None else "all",
    }]

    column_rows = []

    for column in header:
        column_rows.append({
            "file_label": file_label,
            "column_name": column,
            "nonempty_count": nonempty_counts[column],
            "empty_count": row_count - nonempty_counts[column],
            "sample_values": " | ".join(sample_values[column]),
        })

    return {
        "profile_rows": profile_rows,
        "column_rows": column_rows,
        "header": header,
        "row_count": row_count,
    }


# ============================================================
# 4. RÉGI WORKER FÁJL SZÁMLÁLÁSA
# ============================================================

def count_old_workers_by_match_key(old_workers_csv: Path) -> dict:
    counts_by_match_key = defaultdict(int)
    counts_by_county = defaultdict(int)
    counts_by_gender = defaultdict(int)
    counts_by_age_group = defaultdict(int)
    counts_by_education = defaultdict(int)

    workplace_id_count = 0
    workplace_id_missing = 0
    used_fallback_true = 0
    used_fallback_false = 0
    used_teaor_fallback_true = 0
    used_teaor_fallback_false = 0

    total_rows = 0

    with old_workers_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = [
            "county",
            "gender",
            "age_group",
            "education",
        ]

        for required_column in required_columns:
            if required_column not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop a régi worker fájlban: {required_column}"
                )

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            county = normalize_county_name(row["county"])
            gender = normalize_gender_old(row["gender"])
            age_group = normalize_age_group(row["age_group"])
            education = normalize_education(row["education"])

            key = make_match_key(
                county_name=county,
                sex=gender,
                age_group=age_group,
                education=education,
            )

            counts_by_match_key[key] += 1
            counts_by_county[county] += 1
            counts_by_gender[gender] += 1
            counts_by_age_group[age_group] += 1
            counts_by_education[education] += 1

            workplace_id = normalize_text(row.get("workplace_id", ""))

            if workplace_id == "":
                workplace_id_missing += 1
            else:
                workplace_id_count += 1

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
        "counts_by_match_key": dict(counts_by_match_key),
        "counts_by_county": dict(counts_by_county),
        "counts_by_gender": dict(counts_by_gender),
        "counts_by_age_group": dict(counts_by_age_group),
        "counts_by_education": dict(counts_by_education),
        "workplace_id_count": workplace_id_count,
        "workplace_id_missing": workplace_id_missing,
        "used_fallback_true": used_fallback_true,
        "used_fallback_false": used_fallback_false,
        "used_teaor_fallback_true": used_teaor_fallback_true,
        "used_teaor_fallback_false": used_teaor_fallback_false,
    }


# ============================================================
# 5. JELENLEGI FOGLALKOZTATOTT AGENTEK SZÁMLÁLÁSA
# ============================================================

def count_current_employed_agents_by_match_key(current_agents_csv: Path) -> dict:
    counts_by_match_key = defaultdict(int)
    counts_by_county = defaultdict(int)
    counts_by_sex = defaultdict(int)
    counts_by_age_group = defaultdict(int)
    counts_by_education = defaultdict(int)
    activity_counts = defaultdict(int)

    total_rows = 0
    employed_rows = 0

    with current_agents_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = [
            "county_name",
            "sex",
            "exact_age",
            "education_level_calibrated",
            "economic_activity_status_calibrated",
        ]

        for required_column in required_columns:
            if required_column not in reader.fieldnames:
                raise ValueError(
                    f"Hiányzó oszlop a jelenlegi 64-es agent fájlban: {required_column}"
                )

        for row in reader:
            if MAX_ROWS_TO_PROCESS is not None and total_rows >= MAX_ROWS_TO_PROCESS:
                break

            activity = normalize_activity(
                row["economic_activity_status_calibrated"]
            )

            activity_counts[activity] += 1

            if activity == "Foglalkoztatott":
                county = normalize_county_name(row["county_name"])
                sex = normalize_sex_current(row["sex"])
                exact_age = convert_count_to_integer(row["exact_age"])
                age_group = map_exact_age_to_worker_age_group(exact_age)
                education = normalize_education(row["education_level_calibrated"])

                key = make_match_key(
                    county_name=county,
                    sex=sex,
                    age_group=age_group,
                    education=education,
                )

                counts_by_match_key[key] += 1
                counts_by_county[county] += 1
                counts_by_sex[sex] += 1
                counts_by_age_group[age_group] += 1
                counts_by_education[education] += 1

                employed_rows += 1

            total_rows += 1

            if total_rows % 1_000_000 == 0:
                print(f"Jelenlegi agentek feldolgozva: {total_rows:,}")

    return {
        "total_rows": total_rows,
        "employed_rows": employed_rows,
        "counts_by_match_key": dict(counts_by_match_key),
        "counts_by_county": dict(counts_by_county),
        "counts_by_sex": dict(counts_by_sex),
        "counts_by_age_group": dict(counts_by_age_group),
        "counts_by_education": dict(counts_by_education),
        "activity_counts": dict(activity_counts),
    }


# ============================================================
# 6. DATAFRAME KONVERZIÓK ÉS FEASIBILITY
# ============================================================

def match_key_counts_to_dataframe(
    counter: dict,
    count_column_name: str,
) -> pd.DataFrame:
    rows = []

    for key, count in counter.items():
        county, sex, age_group, education = key

        rows.append({
            "county_name": county,
            "sex": sex,
            "age_group": age_group,
            "education": education,
            count_column_name: count,
        })

    return pd.DataFrame(rows)


def create_matching_feasibility_comparison(
    old_worker_counts: dict,
    current_employed_counts: dict,
) -> pd.DataFrame:
    old_df = match_key_counts_to_dataframe(
        old_worker_counts["counts_by_match_key"],
        "old_worker_count",
    )

    current_df = match_key_counts_to_dataframe(
        current_employed_counts["counts_by_match_key"],
        "current_employed_count",
    )

    comparison = old_df.merge(
        current_df,
        on=[
            "county_name",
            "sex",
            "age_group",
            "education",
        ],
        how="outer",
    )

    comparison["old_worker_count"] = pd.to_numeric(
        comparison["old_worker_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["current_employed_count"] = pd.to_numeric(
        comparison["current_employed_count"],
        errors="coerce",
    ).fillna(0).apply(convert_count_to_integer)

    comparison["difference_current_minus_old"] = (
        comparison["current_employed_count"] - comparison["old_worker_count"]
    )

    comparison["absolute_difference"] = comparison[
        "difference_current_minus_old"
    ].abs()

    comparison["old_surplus_worker_records"] = comparison.apply(
        lambda row: max(row["old_worker_count"] - row["current_employed_count"], 0),
        axis=1,
    )

    comparison["current_surplus_employed_agents"] = comparison.apply(
        lambda row: max(row["current_employed_count"] - row["old_worker_count"], 0),
        axis=1,
    )

    comparison["exact_match_assignable_count"] = comparison.apply(
        lambda row: min(row["old_worker_count"], row["current_employed_count"]),
        axis=1,
    )

    comparison["exact_match_possible"] = comparison["absolute_difference"] == 0

    comparison = comparison.sort_values(
        [
            "absolute_difference",
            "old_worker_count",
            "current_employed_count",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return comparison


def simple_counter_to_dataframe(counter: dict, key_name: str, count_name: str) -> pd.DataFrame:
    rows = []

    for key, count in counter.items():
        rows.append({
            key_name: key,
            count_name: count,
        })

    return pd.DataFrame(rows).sort_values(count_name, ascending=False).reset_index(drop=True)


# ============================================================
# 7. SUMMARY ÉS METHOD NOTES
# ============================================================

def create_summary(
    old_profile: dict,
    current_profile: dict,
    old_worker_counts: dict,
    current_employed_counts: dict,
    feasibility: pd.DataFrame,
) -> pd.DataFrame:
    old_total = old_worker_counts["total_rows"]
    current_employed_total = current_employed_counts["employed_rows"]

    exact_assignable = feasibility["exact_match_assignable_count"].sum()
    old_surplus_total = feasibility["old_surplus_worker_records"].sum()
    current_surplus_total = feasibility["current_surplus_employed_agents"].sum()

    rows = [
        {
            "metric": "old_worker_records_total",
            "value": old_total,
            "note": "Régi agents_with_workplaces.csv sorainak száma.",
        },
        {
            "metric": "current_agents_total",
            "value": current_employed_counts["total_rows"],
            "note": "Jelenlegi 64-es agentfájl összes sora.",
        },
        {
            "metric": "current_employed_agents_total",
            "value": current_employed_total,
            "note": "Jelenlegi 64-es fájlban Foglalkoztatott agentek száma.",
        },
        {
            "metric": "current_employed_minus_old_worker_total",
            "value": current_employed_total - old_total,
            "note": "Ideálisan 0, ha a régi worker recordok száma egyezik az új foglalkoztatott agentek számával.",
        },
        {
            "metric": "old_worker_workplace_id_count",
            "value": old_worker_counts["workplace_id_count"],
            "note": "Régi worker rekordok nem üres workplace_id-val.",
        },
        {
            "metric": "old_worker_missing_workplace_id_count",
            "value": old_worker_counts["workplace_id_missing"],
            "note": "Régi worker rekordok hiányzó workplace_id-val. Ideálisan 0.",
        },
        {
            "metric": "old_worker_used_fallback_true",
            "value": old_worker_counts["used_fallback_true"],
            "note": "Régi worker rekordok, ahol workplace assignment fallback történt.",
        },
        {
            "metric": "old_worker_used_fallback_false",
            "value": old_worker_counts["used_fallback_false"],
            "note": "Régi worker rekordok, ahol nem történt fallback.",
        },
        {
            "metric": "old_worker_used_teaor_fallback_true",
            "value": old_worker_counts["used_teaor_fallback_true"],
            "note": "Régi worker rekordok, ahol TEÁOR fallback történt.",
        },
        {
            "metric": "old_worker_used_teaor_fallback_false",
            "value": old_worker_counts["used_teaor_fallback_false"],
            "note": "Régi worker rekordok, ahol nem történt TEÁOR fallback.",
        },
        {
            "metric": "match_key_groups_total",
            "value": len(feasibility),
            "note": "county × sex × age_group × education matching kulcscsoportok száma.",
        },
        {
            "metric": "exact_match_assignable_total",
            "value": exact_assignable,
            "note": "Ennyi agent/worker record párosítható pontos kulcson, ha csak county+sex+age+education szinten matchingelünk.",
        },
        {
            "metric": "exact_match_assignable_share_of_current_employed",
            "value": exact_assignable / current_employed_total if current_employed_total else 0,
            "note": "Pontos kulcson kiosztható arány a jelenlegi foglalkoztatottakhoz képest.",
        },
        {
            "metric": "old_surplus_worker_records_total",
            "value": old_surplus_total,
            "note": "Régi worker rekord többlet pontos matching kulcson. Ezekhez fallback kellhet.",
        },
        {
            "metric": "current_surplus_employed_agents_total",
            "value": current_surplus_total,
            "note": "Új foglalkoztatott agent többlet pontos matching kulcson. Ezekhez fallback kellhet.",
        },
        {
            "metric": "max_match_key_absolute_difference",
            "value": feasibility["absolute_difference"].max() if len(feasibility) else 0,
            "note": "Legnagyobb eltérés egy county × sex × age_group × education csoportban.",
        },
    ]

    return pd.DataFrame(rows)


def write_method_notes() -> None:
    notes = [
        {
            "order": 1,
            "note": "Ez a script nem módosít agenteket és nem rendel workplace-et. Csak profilozza a régi worker/workplace fájlt és a jelenlegi foglalkoztatott agenteket.",
        },
        {
            "order": 2,
            "note": "A régi agents_with_workplaces.csv-t worker/workplace record poolként kezeljük.",
        },
        {
            "order": 3,
            "note": "A jelenlegi 64-es agentfájlból csak a Foglalkoztatott agentek vesznek részt a matching feasibility vizsgálatban.",
        },
        {
            "order": 4,
            "note": "Az elsődleges összehasonlító kulcs: county × sex/gender × 5 éves age_group × education.",
        },
        {
            "order": 5,
            "note": "A comparison output megmutatja, hány rekord párosítható pontosan, és hol lesz szükség fallback matchingre.",
        },
        {
            "order": 6,
            "note": "A régi workplace assignment fallback mezőit külön kell majd megőrizni az új matching fallback mezőitől.",
        },
        {
            "order": 7,
            "note": "A későbbi illesztő scriptben minden agent kap workplace/occupation oszlopokat; a nem foglalkoztatottaknál explicit NO_WORKPLACE / NO_OCCUPATION jelölés javasolt, nem üres cella.",
        },
    ]

    pd.DataFrame(notes).to_csv(
        MATCHING_METHOD_NOTES_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 8. FŐ FUTTATÁS
# ============================================================

def run_previous_workplace_layer_profile() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    print("Régi worker/workplace fájl profilozása...")
    old_profile = profile_csv_file(
        csv_path=PREVIOUS_WORKER_WORKPLACE_CSV,
        file_label="old_agents_with_workplaces",
    )

    print("Jelenlegi 64-es agentfájl profilozása...")
    current_profile = profile_csv_file(
        csv_path=CURRENT_AGENTS_CSV,
        file_label="current_agents_64",
    )

    print("Régi worker/workplace rekordok számlálása matching kulcs szerint...")
    old_worker_counts = count_old_workers_by_match_key(
        old_workers_csv=PREVIOUS_WORKER_WORKPLACE_CSV,
    )

    print("Jelenlegi foglalkoztatott agentek számlálása matching kulcs szerint...")
    current_employed_counts = count_current_employed_agents_by_match_key(
        current_agents_csv=CURRENT_AGENTS_CSV,
    )

    print("Matching feasibility comparison készítése...")
    feasibility = create_matching_feasibility_comparison(
        old_worker_counts=old_worker_counts,
        current_employed_counts=current_employed_counts,
    )

    print("Summary készítése...")
    summary = create_summary(
        old_profile=old_profile,
        current_profile=current_profile,
        old_worker_counts=old_worker_counts,
        current_employed_counts=current_employed_counts,
        feasibility=feasibility,
    )

    print("Method notes írása...")
    write_method_notes()

    print("Outputok mentése...")

    pd.DataFrame(old_profile["profile_rows"] + current_profile["profile_rows"]).to_csv(
        OLD_WORKER_FILE_PROFILE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(old_profile["column_rows"] + current_profile["column_rows"]).to_csv(
        OLD_WORKER_COLUMNS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    match_key_counts_to_dataframe(
        old_worker_counts["counts_by_match_key"],
        "old_worker_count",
    ).to_csv(
        OLD_WORKER_COUNTS_BY_MATCH_KEY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    match_key_counts_to_dataframe(
        current_employed_counts["counts_by_match_key"],
        "current_employed_count",
    ).to_csv(
        CURRENT_EMPLOYED_COUNTS_BY_MATCH_KEY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    feasibility.to_csv(
        MATCHING_FEASIBILITY_COMPARISON_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        MATCHING_PROFILE_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Kész.")
    print(f"Régi worker rekordok: {old_worker_counts['total_rows']:,}")
    print(f"Jelenlegi foglalkoztatott agentek: {current_employed_counts['employed_rows']:,}")
    print(f"Különbség: {current_employed_counts['employed_rows'] - old_worker_counts['total_rows']:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_previous_workplace_layer_profile()