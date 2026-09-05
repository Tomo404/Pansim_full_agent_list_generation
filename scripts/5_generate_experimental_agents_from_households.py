from pathlib import Path
import csv
import random
from collections import defaultdict


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

GENERATED_HOUSEHOLDS_CSV = Path(
    "../outputs/household_generation/05_generated_households.csv"
)

ALLOCATED_SETTLEMENT_COMPOSITION_TARGETS_CSV = Path(
    "../outputs/household_generation/12_allocated_settlement_composition_targets.csv"
)

OUTPUT_FOLDER = Path("../outputs/household_generation")

HOUSEHOLDS_WITH_COMPOSITION_CSV = OUTPUT_FOLDER / "17_generated_households_with_composition.csv"

EXPERIMENTAL_AGENTS_CSV = OUTPUT_FOLDER / "18_generated_agents_from_households_experimental.csv"

VALIDATION_SUMMARY_CSV = OUTPUT_FOLDER / "19_experimental_agent_generation_validation.csv"

AGENT_COUNTS_BY_SETTLEMENT_CSV = OUTPUT_FOLDER / "20_experimental_agent_counts_by_settlement.csv"

AGENT_COUNTS_BY_AGE_STATUS_CSV = OUTPUT_FOLDER / "21_experimental_agent_counts_by_age_status.csv"


AGENT_ID_PREFIX = "AG"

RANDOM_SEED = 42

# Ha tesztelni akarod gyorsan, állítsd például 100_000-re.
# Ha None, akkor minden householdot feldolgoz.
MAX_HOUSEHOLDS_TO_PROCESS = None


# ============================================================
# 2. ÁLTALÁNOS SEGÉDFÜGGVÉNYEK
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
    if raw_value is None:
        return 0

    text_value = str(raw_value).strip()

    if text_value == "":
        return 0

    return int(round(float(text_value)))


def normalize_text(raw_text: str) -> str:
    """
    Egyszerű kisbetűsített szöveg normalizálás kategóriafelismeréshez.
    """
    if raw_text is None:
        return ""

    return str(raw_text).strip().lower()


def map_household_size_numeric_to_label(household_size_numeric: int) -> str:
    """
    A generated household fájl numeric méretéből visszaadja
    a 12-es kompozíciós target fájlban használt size labelt.
    """
    if household_size_numeric <= 1:
        return "1_person"

    if household_size_numeric == 2:
        return "2_person"

    if household_size_numeric == 3:
        return "3_person"

    if household_size_numeric == 4:
        return "4_person"

    if household_size_numeric == 5:
        return "5_person"

    return "6plus_person"


# ============================================================
# 3. KOMPOZÍCIÓS TARGETEK BETÖLTÉSE
# ============================================================

def load_composition_targets_by_settlement_and_size(
    allocated_targets_csv: Path,
) -> dict:
    """
    Beolvassa a 12_allocated_settlement_composition_targets.csv fájlt.

    Output:
    {
        (settlement_key, household_size_label): [
            {
                age_composition,
                employment_composition,
                remaining_count
            },
            ...
        ]
    }

    A remaining_count alapján fogjuk kiosztani konkrét householdokra
    a kompozíciós címkéket.
    """
    targets_by_key = defaultdict(list)

    with allocated_targets_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            settlement_key = row["settlement_key"]
            household_size_label = row["household_size_label"]

            allocated_count = convert_count_to_integer(row["allocated_household_count"])

            if allocated_count <= 0:
                continue

            key = (settlement_key, household_size_label)

            targets_by_key[key].append({
                "household_age_composition": row["household_age_composition"],
                "household_employment_composition": row["household_employment_composition"],
                "remaining_count": allocated_count,
            })

    # Keverjük a kategóriák sorrendjét, hogy ne mindig ugyanaz a kompozíció
    # kerüljön előre az adott település + méret csoportban.
    random_generator = random.Random(RANDOM_SEED)

    for key in targets_by_key:
        random_generator.shuffle(targets_by_key[key])

    return targets_by_key


def pop_next_composition_for_household(
    targets_by_key: dict,
    settlement_key: str,
    household_size_label: str,
) -> dict:
    """
    Kiveszi a következő elérhető kompozíciós targetet
    egy adott település + household_size csoportból.

    Ha minden jól ment az előző scriptben, akkor mindig talál targetet.
    """
    key = (settlement_key, household_size_label)

    if key not in targets_by_key:
        raise ValueError(
            f"Nincs kompozíciós target ehhez a csoporthoz: {key}"
        )

    target_list = targets_by_key[key]

    for target in target_list:
        if target["remaining_count"] > 0:
            target["remaining_count"] -= 1
            return {
                "household_age_composition": target["household_age_composition"],
                "household_employment_composition": target["household_employment_composition"],
            }

    raise ValueError(
        f"Elfogytak a kompozíciós targetek ehhez a csoporthoz: {key}"
    )


# ============================================================
# 4. AGE COMPOSITION → KONKRÉT BROAD AGE GROUP LISTA
# ============================================================

def infer_required_age_groups_from_composition(
    household_age_composition: str,
) -> list[str]:
    """
    A KSH household korösszetétel szövegéből megállapítja,
    milyen broad age groupoknak kell jelen lenniük.

    Használt csoportok:
    - under_30
    - age_30_64
    - age_65_plus
    """
    text = normalize_text(household_age_composition)

    has_under_30 = "30 évesnél fiatalabb" in text
    has_30_64 = "30–64" in text or "30-64" in text
    has_65_plus = "65 éves" in text

    required_age_groups = []

    if has_under_30:
        required_age_groups.append("under_30")

    if has_30_64:
        required_age_groups.append("age_30_64")

    if has_65_plus:
        required_age_groups.append("age_65_plus")

    return required_age_groups


def create_age_group_list_for_household(
    household_size_numeric: int,
    household_age_composition: str,
) -> list[str]:
    """
    A household korösszetétel címkéből létrehoz egy személyszintű
    broad age group listát.

    Egyszerű szabály:
    - minden required age group kap legalább 1 személyt
    - a maradék személyeket egy domináns/default csoportba tesszük
    """
    required_age_groups = infer_required_age_groups_from_composition(
        household_age_composition
    )

    if not required_age_groups:
        # Ha valamiért nem ismerjük fel, középkorúként kezeljük.
        required_age_groups = ["age_30_64"]

    age_groups = []

    for age_group in required_age_groups:
        if len(age_groups) < household_size_numeric:
            age_groups.append(age_group)

    while len(age_groups) < household_size_numeric:
        # Ha van 30-64-es a háztartásban, az legyen a default kitöltés.
        # Ha nincs, akkor az első required kategóriával töltjük fel.
        if "age_30_64" in required_age_groups:
            age_groups.append("age_30_64")
        else:
            age_groups.append(required_age_groups[0])

    return age_groups


# ============================================================
# 5. EMPLOYMENT COMPOSITION → STÁTUSZ DARABSZÁMOK
# ============================================================

def choose_employed_count_for_three_plus_household(
    household_size_numeric: int,
    random_generator: random.Random,
) -> int:
    """
    A 'három vagy több foglalkoztatott' kategória nem mondja meg pontosan,
    hogy 3, 4, 5... foglalkoztatott van.

    Első kísérleti szabály:
    - minimum 3
    - maximum household size
    - a kisebb értékek valószínűbbek, mert a 3+ kategória aggregált.
    """
    minimum_employed = min(3, household_size_numeric)
    maximum_employed = household_size_numeric

    possible_counts = list(range(minimum_employed, maximum_employed + 1))

    # Súlyok: 3 foglalkoztatott legyen a legvalószínűbb,
    # aztán csökkenjen.
    weights = []

    for count in possible_counts:
        distance_from_minimum = count - minimum_employed
        weight = 1 / (1 + distance_from_minimum)
        weights.append(weight)

    return random_generator.choices(
        population=possible_counts,
        weights=weights,
        k=1,
    )[0]


def infer_status_counts_from_employment_composition(
    household_size_numeric: int,
    household_employment_composition: str,
    random_generator: random.Random,
) -> dict:
    """
    A household foglalkoztatottsági összetételéből személyszintű státusz darabszámokat becsül.

    Használt státuszok:
    - employed
    - unemployed
    - inactive_benefit
    - dependent
    """
    text = normalize_text(household_employment_composition)

    status_counts = {
        "employed": 0,
        "unemployed": 0,
        "inactive_benefit": 0,
        "dependent": 0,
    }

    if "egy foglalkoztatott" in text:
        status_counts["employed"] = min(1, household_size_numeric)

    elif "két foglalkoztatott" in text:
        status_counts["employed"] = min(2, household_size_numeric)

    elif "három vagy több foglalkoztatott" in text:
        status_counts["employed"] = choose_employed_count_for_three_plus_household(
            household_size_numeric=household_size_numeric,
            random_generator=random_generator,
        )

    elif "csak eltartott" in text:
        status_counts["dependent"] = household_size_numeric

    elif "nincs foglalkoztatott" in text and "munkanélküli" in text and "nincs munkanélküli" not in text:
        # Nincs dolgozó, de van munkanélküli és/vagy ellátott.
        # Első kísérletben biztosítunk legalább 1 munkanélkülit.
        status_counts["unemployed"] = 1

    elif "nincs foglalkoztatott" in text and "nincs munkanélküli" in text:
        # Nincs dolgozó és nincs munkanélküli, de van ellátásban részesülő inaktív.
        status_counts["inactive_benefit"] = 1

    else:
        # Fallback: ha nem ismerjük fel, legyen mindenki dependent.
        status_counts["dependent"] = household_size_numeric

    already_assigned = (
        status_counts["employed"]
        + status_counts["unemployed"]
        + status_counts["inactive_benefit"]
        + status_counts["dependent"]
    )

    remaining_people = household_size_numeric - already_assigned

    if remaining_people > 0:
        if status_counts["employed"] > 0:
            # Ha van foglalkoztatott, a többi első körben eltartott.
            status_counts["dependent"] += remaining_people

        elif status_counts["unemployed"] > 0:
            # Ha nincs dolgozó, de van munkanélküli, a maradék lehet eltartott.
            status_counts["dependent"] += remaining_people

        elif status_counts["inactive_benefit"] > 0:
            # Ha van ellátott inaktív, a maradékot életkor alapján később finomítanánk.
            # Most konzervatívan dependent.
            status_counts["dependent"] += remaining_people

        else:
            status_counts["dependent"] += remaining_people

    return status_counts


# ============================================================
# 6. AGE GROUP + STATUS ÖSSZERENDELÉSE SZEMÉLYEKRE
# ============================================================

def choose_agent_index_for_status(
    current_agents: list[dict],
    preferred_age_groups: list[str],
) -> int | None:
    """
    Kiválaszt egy még státusz nélküli agentet a preferált age groupok alapján.
    """
    for preferred_age_group in preferred_age_groups:
        for index, agent in enumerate(current_agents):
            if (
                agent["economic_activity_status"] == ""
                and agent["broad_age_group"] == preferred_age_group
            ):
                return index

    for index, agent in enumerate(current_agents):
        if agent["economic_activity_status"] == "":
            return index

    return None


def assign_statuses_to_age_groups(
    age_groups: list[str],
    status_counts: dict,
) -> list[dict]:
    """
    Az age group listához gazdasági státuszokat rendel.

    Preferenciák:
    - employed: főleg 30-64, utána under_30, végül 65+
    - unemployed: főleg 30-64, utána under_30
    - inactive_benefit: főleg 65+, utána 30-64
    - dependent: maradék
    """
    agents = []

    for age_group in age_groups:
        agents.append({
            "broad_age_group": age_group,
            "economic_activity_status": "",
        })

    status_assignment_order = [
        (
            "employed",
            ["age_30_64", "under_30", "age_65_plus"],
        ),
        (
            "unemployed",
            ["age_30_64", "under_30", "age_65_plus"],
        ),
        (
            "inactive_benefit",
            ["age_65_plus", "age_30_64", "under_30"],
        ),
        (
            "dependent",
            ["under_30", "age_30_64", "age_65_plus"],
        ),
    ]

    for status_name, preferred_age_groups in status_assignment_order:
        for _ in range(status_counts[status_name]):
            chosen_index = choose_agent_index_for_status(
                current_agents=agents,
                preferred_age_groups=preferred_age_groups,
            )

            if chosen_index is not None:
                agents[chosen_index]["economic_activity_status"] = status_name

    for agent in agents:
        if agent["economic_activity_status"] == "":
            agent["economic_activity_status"] = "dependent"

    return agents


def generate_agents_for_one_household(
    household_size_numeric: int,
    household_age_composition: str,
    household_employment_composition: str,
    random_generator: random.Random,
) -> list[dict]:
    """
    Egy household-kompozíciós címkéből konkrét agenteket generál.
    """
    age_groups = create_age_group_list_for_household(
        household_size_numeric=household_size_numeric,
        household_age_composition=household_age_composition,
    )

    status_counts = infer_status_counts_from_employment_composition(
        household_size_numeric=household_size_numeric,
        household_employment_composition=household_employment_composition,
        random_generator=random_generator,
    )

    agents = assign_statuses_to_age_groups(
        age_groups=age_groups,
        status_counts=status_counts,
    )

    return agents


# ============================================================
# 7. VALIDÁCIÓS SEGÉDFÜGGVÉNYEK
# ============================================================

def validate_household_age_composition(
    household_age_composition: str,
    agents: list[dict],
) -> bool:
    """
    Ellenőrzi, hogy a generált agentek broad age groupjai tartalmazzák-e
    a household age composition alapján elvárt csoportokat.
    """
    required_age_groups = set(
        infer_required_age_groups_from_composition(household_age_composition)
    )

    generated_age_groups = set(
        agent["broad_age_group"]
        for agent in agents
    )

    return required_age_groups.issubset(generated_age_groups)


def validate_household_employment_composition(
    household_employment_composition: str,
    agents: list[dict],
) -> bool:
    """
    Ellenőrzi, hogy a generált agentek economic_activity_status mezői
    teljesítik-e a household employment composition fő feltételét.
    """
    text = normalize_text(household_employment_composition)

    employed_count = sum(
        1
        for agent in agents
        if agent["economic_activity_status"] == "employed"
    )

    unemployed_count = sum(
        1
        for agent in agents
        if agent["economic_activity_status"] == "unemployed"
    )

    inactive_benefit_count = sum(
        1
        for agent in agents
        if agent["economic_activity_status"] == "inactive_benefit"
    )

    dependent_count = sum(
        1
        for agent in agents
        if agent["economic_activity_status"] == "dependent"
    )

    if "egy foglalkoztatott" in text:
        return employed_count == 1

    if "két foglalkoztatott" in text:
        return employed_count == 2

    if "három vagy több foglalkoztatott" in text:
        return employed_count >= 3

    if "csak eltartott" in text:
        return dependent_count == len(agents)

    if "nincs foglalkoztatott" in text and "munkanélküli" in text and "nincs munkanélküli" not in text:
        return employed_count == 0 and (unemployed_count + inactive_benefit_count) >= 1

    if "nincs foglalkoztatott" in text and "nincs munkanélküli" in text:
        return employed_count == 0 and unemployed_count == 0 and inactive_benefit_count >= 1

    return True


def update_nested_counter(counter: dict, key_tuple: tuple, increment: int = 1) -> None:
    """
    Egyszerű tuple-key alapú számláló frissítése.
    """
    counter[key_tuple] += increment


# ============================================================
# 8. FŐ GENERÁLÁSI FOLYAMAT
# ============================================================

def generate_households_with_composition_and_agents() -> None:
    """
    Fő folyamat:

    1. Betölti a településszintű household-kompozíciós targeteket.
    2. Végigmegy a 05_generated_households.csv sorain.
    3. Minden householdhoz rendel age/employment kompozíciót.
    4. Minden householdhoz generál személyszintű agenteket.
    5. Közben validációs számlálókat gyűjt.
    """
    create_output_folder_if_missing(OUTPUT_FOLDER)

    random_generator = random.Random(RANDOM_SEED)

    print("Kompozíciós targetek betöltése...")
    composition_targets_by_key = load_composition_targets_by_settlement_and_size(
        ALLOCATED_SETTLEMENT_COMPOSITION_TARGETS_CSV
    )

    print("Output fájlok előkészítése...")

    total_households_processed = 0
    total_agents_generated = 0
    total_expected_agents_from_household_size = 0

    age_composition_error_count = 0
    employment_composition_error_count = 0
    household_size_error_count = 0

    agent_counts_by_settlement = defaultdict(int)
    agent_counts_by_age_status = defaultdict(int)
    household_counts_by_size = defaultdict(int)
    household_counts_by_employment_composition = defaultdict(int)

    household_output_columns = [
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
        "household_size_label",
        "household_age_composition",
        "household_employment_composition",
    ]

    agent_output_columns = [
        "agent_id",
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
        "household_size_numeric",
        "agent_position_in_household",
        "broad_age_group",
        "economic_activity_status",
        "household_age_composition",
        "household_employment_composition",
        "generation_rule",
    ]

    with GENERATED_HOUSEHOLDS_CSV.open("r", encoding="utf-8-sig", newline="") as household_input_file, \
            HOUSEHOLDS_WITH_COMPOSITION_CSV.open("w", encoding="utf-8-sig", newline="") as household_output_file, \
            EXPERIMENTAL_AGENTS_CSV.open("w", encoding="utf-8-sig", newline="") as agent_output_file:

        household_reader = csv.DictReader(household_input_file)

        household_writer = csv.DictWriter(
            household_output_file,
            fieldnames=household_output_columns,
        )
        household_writer.writeheader()

        agent_writer = csv.DictWriter(
            agent_output_file,
            fieldnames=agent_output_columns,
        )
        agent_writer.writeheader()

        agent_counter = 1

        for household_row in household_reader:
            if (
                MAX_HOUSEHOLDS_TO_PROCESS is not None
                and total_households_processed >= MAX_HOUSEHOLDS_TO_PROCESS
            ):
                break

            household_size_numeric = convert_count_to_integer(
                household_row["household_size_numeric"]
            )

            household_size_label = map_household_size_numeric_to_label(
                household_size_numeric
            )

            composition = pop_next_composition_for_household(
                targets_by_key=composition_targets_by_key,
                settlement_key=household_row["settlement_key"],
                household_size_label=household_size_label,
            )

            household_age_composition = composition["household_age_composition"]
            household_employment_composition = composition[
                "household_employment_composition"
            ]

            household_with_composition = dict(household_row)
            household_with_composition["household_size_label"] = household_size_label
            household_with_composition["household_age_composition"] = household_age_composition
            household_with_composition["household_employment_composition"] = (
                household_employment_composition
            )

            household_writer.writerow({
                column_name: household_with_composition.get(column_name, "")
                for column_name in household_output_columns
            })

            agents_for_household = generate_agents_for_one_household(
                household_size_numeric=household_size_numeric,
                household_age_composition=household_age_composition,
                household_employment_composition=household_employment_composition,
                random_generator=random_generator,
            )

            if len(agents_for_household) != household_size_numeric:
                household_size_error_count += 1

            if not validate_household_age_composition(
                household_age_composition=household_age_composition,
                agents=agents_for_household,
            ):
                age_composition_error_count += 1

            if not validate_household_employment_composition(
                household_employment_composition=household_employment_composition,
                agents=agents_for_household,
            ):
                employment_composition_error_count += 1

            total_households_processed += 1
            total_expected_agents_from_household_size += household_size_numeric
            total_agents_generated += len(agents_for_household)

            update_nested_counter(
                household_counts_by_size,
                (household_size_label,),
                increment=1,
            )

            update_nested_counter(
                household_counts_by_employment_composition,
                (household_employment_composition,),
                increment=1,
            )

            for agent_position, generated_agent in enumerate(
                agents_for_household,
                start=1,
            ):
                agent_id = f"{AGENT_ID_PREFIX}_{agent_counter:010d}"

                agent_writer.writerow({
                    "agent_id": agent_id,
                    "household_id": household_row["household_id"],
                    "dwelling_id": household_row["dwelling_id"],
                    "settlement_key": household_row["settlement_key"],
                    "settlement_name": household_row["settlement_name"],
                    "settlement_ksh_code": household_row["settlement_ksh_code"],
                    "county_name": household_row["county_name"],
                    "county_code": household_row["county_code"],
                    "settlement_type": household_row["settlement_type"],
                    "district_code": household_row["district_code"],
                    "district_name": household_row["district_name"],
                    "household_size_numeric": household_size_numeric,
                    "agent_position_in_household": agent_position,
                    "broad_age_group": generated_agent["broad_age_group"],
                    "economic_activity_status": generated_agent[
                        "economic_activity_status"
                    ],
                    "household_age_composition": household_age_composition,
                    "household_employment_composition": household_employment_composition,
                    "generation_rule": "experimental_from_homesize_agegroup_employment",
                })

                update_nested_counter(
                    agent_counts_by_settlement,
                    (
                        household_row["settlement_key"],
                        household_row["settlement_name"],
                    ),
                    increment=1,
                )

                update_nested_counter(
                    agent_counts_by_age_status,
                    (
                        generated_agent["broad_age_group"],
                        generated_agent["economic_activity_status"],
                    ),
                    increment=1,
                )

                agent_counter += 1

            if total_households_processed % 100_000 == 0:
                print(f"Feldolgozott householdok: {total_households_processed:,}")

    print("Validációs outputok írása...")

    write_validation_summary(
        total_households_processed=total_households_processed,
        total_expected_agents_from_household_size=total_expected_agents_from_household_size,
        total_agents_generated=total_agents_generated,
        household_size_error_count=household_size_error_count,
        age_composition_error_count=age_composition_error_count,
        employment_composition_error_count=employment_composition_error_count,
        household_counts_by_size=household_counts_by_size,
        household_counts_by_employment_composition=household_counts_by_employment_composition,
    )

    write_agent_counts_by_settlement(agent_counts_by_settlement)
    write_agent_counts_by_age_status(agent_counts_by_age_status)

    print()
    print("Kész.")
    print(f"Feldolgozott householdok: {total_households_processed:,}")
    print(f"Generált agentek: {total_agents_generated:,}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


# ============================================================
# 9. VALIDÁCIÓS OUTPUTOK ÍRÁSA
# ============================================================

def write_validation_summary(
    total_households_processed: int,
    total_expected_agents_from_household_size: int,
    total_agents_generated: int,
    household_size_error_count: int,
    age_composition_error_count: int,
    employment_composition_error_count: int,
    household_counts_by_size: dict,
    household_counts_by_employment_composition: dict,
) -> None:
    """
    Rövid validációs összefoglalót ír.
    """
    rows = []

    rows.append({
        "metric": "total_households_processed",
        "value": total_households_processed,
        "note": "Feldolgozott household sorok száma.",
    })

    rows.append({
        "metric": "total_expected_agents_from_household_size",
        "value": total_expected_agents_from_household_size,
        "note": "Household_size_numeric összege. Ennyi agentet várunk.",
    })

    rows.append({
        "metric": "total_agents_generated",
        "value": total_agents_generated,
        "note": "Ténylegesen generált agentek száma.",
    })

    rows.append({
        "metric": "difference_agents_generated_vs_expected",
        "value": total_agents_generated - total_expected_agents_from_household_size,
        "note": "Ennek 0-nak kell lennie.",
    })

    rows.append({
        "metric": "household_size_error_count",
        "value": household_size_error_count,
        "note": "Olyan householdok száma, ahol agent_count != household_size_numeric.",
    })

    rows.append({
        "metric": "age_composition_error_count",
        "value": age_composition_error_count,
        "note": "Olyan householdok száma, ahol a broad age groupok nem teljesítik az age composition feltételt.",
    })

    rows.append({
        "metric": "employment_composition_error_count",
        "value": employment_composition_error_count,
        "note": "Olyan householdok száma, ahol a státuszok nem teljesítik az employment composition feltételt.",
    })

    for key_tuple, count in sorted(household_counts_by_size.items()):
        household_size_label = key_tuple[0]

        rows.append({
            "metric": f"household_count_{household_size_label}",
            "value": count,
            "note": "Household darabszám household_size_label szerint.",
        })

    for key_tuple, count in sorted(household_counts_by_employment_composition.items()):
        employment_composition = key_tuple[0]

        rows.append({
            "metric": f"household_count_employment_composition::{employment_composition}",
            "value": count,
            "note": "Household darabszám employment composition címke szerint.",
        })

    with VALIDATION_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_agent_counts_by_settlement(agent_counts_by_settlement: dict) -> None:
    """
    Településszintű agent darabszámokat ír.
    """
    with AGENT_COUNTS_BY_SETTLEMENT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "settlement_key",
                "settlement_name",
                "agent_count",
            ],
        )
        writer.writeheader()

        for key_tuple, count in sorted(agent_counts_by_settlement.items()):
            settlement_key, settlement_name = key_tuple

            writer.writerow({
                "settlement_key": settlement_key,
                "settlement_name": settlement_name,
                "agent_count": count,
            })


def write_agent_counts_by_age_status(agent_counts_by_age_status: dict) -> None:
    """
    Age group × economic activity status szintű agent darabszámokat ír.
    """
    with AGENT_COUNTS_BY_AGE_STATUS_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "broad_age_group",
                "economic_activity_status",
                "agent_count",
            ],
        )
        writer.writeheader()

        for key_tuple, count in sorted(agent_counts_by_age_status.items()):
            broad_age_group, economic_activity_status = key_tuple

            writer.writerow({
                "broad_age_group": broad_age_group,
                "economic_activity_status": economic_activity_status,
                "agent_count": count,
            })


# ============================================================
# 10. PROGRAM INDÍTÁSA
# ============================================================

if __name__ == "__main__":
    generate_households_with_composition_and_agents()