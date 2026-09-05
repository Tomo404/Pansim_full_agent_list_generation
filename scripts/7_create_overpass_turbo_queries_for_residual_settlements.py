from pathlib import Path
import csv
import re


# ============================================================
# 1. BEÁLLÍTÁSOK
# ============================================================

RESIDUAL_SETTLEMENTS_CSV = Path(
    "../outputs/household_generation/22_residual_population_without_6plus_households.csv"
)

OUTPUT_FOLDER = Path("../outputs/household_generation/osm_overpass_queries")

# Ennyi település legyen egy Overpass Turbo query-ben.
# Ha timeoutol, vedd le 10-re.
# Ha gyorsan fut, lehet 30-ra emelni.
SETTLEMENTS_PER_QUERY = 5

# Csak teszteléshez.
# Ha None, minden residual településre generál query-t.
MAX_SETTLEMENTS_TO_INCLUDE = None


# ============================================================
# 2. SEGÉDFÜGGVÉNYEK
# ============================================================

def create_output_folder_if_missing(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


def escape_regex_text(raw_text: str) -> str:
    """
    Településnév regexhez escape-elve.
    """
    return re.escape(str(raw_text).strip())


def load_residual_settlements(input_csv: Path) -> list[dict]:
    """
    Beolvassa a residual településlistát.
    """
    with input_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    rows = sorted(
        rows,
        key=lambda row: int(round(float(row["estimated_people_in_6plus_households"]))),
        reverse=True,
    )

    if MAX_SETTLEMENTS_TO_INCLUDE is not None:
        rows = rows[:MAX_SETTLEMENTS_TO_INCLUDE]

    return rows


def split_into_chunks(items: list, chunk_size: int) -> list[list]:
    """
    Listát darabokra vág.
    """
    return [
        items[index:index + chunk_size]
        for index in range(0, len(items), chunk_size)
    ]


def build_settlement_name_regex(settlement_rows: list[dict]) -> str:
    """
    Településnév regex készítése:
    ^(Település1|Település2|...)$
    """
    escaped_names = [
        escape_regex_text(row["settlement_name_flat"])
        for row in settlement_rows
    ]

    return "^(" + "|".join(escaped_names) + ")$"


def build_overpass_query(settlement_rows: list[dict], batch_index: int) -> str:
    """
    Overpass Turbo query készítése egy településcsomagra.

    Fontos:
    Ez nem Pythonból futtatja le a kérést.
    Ezt bemásolod https://overpass-turbo.eu/ oldalra.
    """
    settlement_regex = build_settlement_name_regex(settlement_rows)

    settlement_comment_lines = []

    for row in settlement_rows:
        settlement_comment_lines.append(
            f"// - {row['settlement_name_flat']} "
            f"({row['county_name']}), residual={row['estimated_people_in_6plus_households']}"
        )

    settlement_comment_block = "\n".join(settlement_comment_lines)

    query = f"""
/*
Residual settlement OSM institution search
Batch: {batch_index}

Települések:
{settlement_comment_block}

Cél:
Börtönök, idősotthonok, szociális intézmények, gyermekotthonok,
kollégiumok, munkásszállók és egyéb nagyobb intézmények keresése
a residual településeken.
*/

[out:json][timeout:180];

// Magyarország területe
area["ISO3166-1"="HU"][admin_level="2"]->.hungary;

// A batch-ben szereplő települések adminisztratív relációi.
// Magyarországon a települések jellemzően admin_level=8 alatt vannak.
rel(area.hungary)
  ["boundary"="administrative"]
  ["admin_level"="8"]
  ["name"~"{settlement_regex}"];

map_to_area -> .settlement_areas;

// Intézmény-jellegű objektumok keresése a települések területén.
(
  // ------------------------------------------------------------
  // DIREKT INTÉZMÉNYTALÁLATOK
  // ------------------------------------------------------------

  // Börtön / büntetés-végrehajtás
  nwr(area.settlement_areas)["amenity"="prison"];
  nwr(area.settlement_areas)
    ["name"~"börtön|fegyház|büntetés|büntetés-végrehajtás|BV", i]
    ["highway"!="bus_stop"]
    ["public_transport"!="platform"];

  // Szociális intézmények, idősotthonok
  nwr(area.settlement_areas)["amenity"="social_facility"];
  nwr(area.settlement_areas)["social_facility"];
  nwr(area.settlement_areas)["amenity"="nursing_home"];
  nwr(area.settlement_areas)["amenity"="retirement_home"];

  nwr(area.settlement_areas)
    ["name"~"idősotthon|idősek otthona|öregek otthona|nyugdíjasház|szociális otthon|szociális intézmény", i]
    ["highway"!="bus_stop"]
    ["public_transport"!="platform"];

  // Gyermekotthon / lakásotthon
  nwr(area.settlement_areas)
    ["name"~"gyermekotthon|nevelőotthon|lakásotthon|gyermekvédelmi", i]
    ["highway"!="bus_stop"]
    ["public_transport"!="platform"];

  // Kollégium / diákotthon
  nwr(area.settlement_areas)["building"="dormitory"];
  nwr(area.settlement_areas)["amenity"="dormitory"];

  nwr(area.settlement_areas)
    ["name"~"kollégium|diákotthon", i]
    ["highway"!="bus_stop"]
    ["public_transport"!="platform"];

  // Munkásszálló / szálló
  nwr(area.settlement_areas)
    ["name"~"munkásszálló|munkásszállás", i]
    ["highway"!="bus_stop"]
    ["public_transport"!="platform"];

  nwr(area.settlement_areas)["tourism"="hostel"];

  // Egyéb nagyobb intézmények
  nwr(area.settlement_areas)["amenity"="hospital"];
  nwr(area.settlement_areas)["amenity"="university"];
  nwr(area.settlement_areas)["amenity"="college"];

  // ------------------------------------------------------------
  // KÖZVETETT NYOMOK
  // Ezek nem maguk az intézmények, de utalhatnak rájuk.
  // Példa: buszmegálló neve "Patalom, szociális otthon".
  // ------------------------------------------------------------

  nwr(area.settlement_areas)
    ["name"~"börtön|fegyház|büntetés|büntetés-végrehajtás|idősotthon|idősek otthona|öregek otthona|szociális otthon|szociális intézmény|gyermekotthon|nevelőotthon|lakásotthon|kollégium|diákotthon|munkásszálló|munkásszállás", i]
    ["highway"="bus_stop"];

  nwr(area.settlement_areas)
    ["name"~"börtön|fegyház|büntetés|büntetés-végrehajtás|idősotthon|idősek otthona|öregek otthona|szociális otthon|szociális intézmény|gyermekotthon|nevelőotthon|lakásotthon|kollégium|diákotthon|munkásszálló|munkásszállás", i]
    ["public_transport"="platform"];
);

// Találatok kiírása.
out tags center;
"""
    return query.strip()


def write_batch_metadata(settlement_chunks: list[list[dict]]) -> None:
    """
    Rövid batch index CSV-t ír, hogy tudd, melyik query melyik településeket tartalmazza.
    """
    output_path = OUTPUT_FOLDER / "overpass_query_batches_index.csv"

    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        fieldnames = [
            "batch_index",
            "settlement_name",
            "county_name",
            "residual_population",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for batch_index, chunk in enumerate(settlement_chunks, start=1):
            for row in chunk:
                writer.writerow({
                    "batch_index": batch_index,
                    "settlement_name": row["settlement_name_flat"],
                    "county_name": row["county_name"],
                    "residual_population": row["estimated_people_in_6plus_households"],
                })


# ============================================================
# 3. FŐ FUTTATÁS
# ============================================================

def run_overpass_query_generation() -> None:
    create_output_folder_if_missing(OUTPUT_FOLDER)

    residual_settlements = load_residual_settlements(RESIDUAL_SETTLEMENTS_CSV)

    settlement_chunks = split_into_chunks(
        residual_settlements,
        SETTLEMENTS_PER_QUERY,
    )

    for batch_index, chunk in enumerate(settlement_chunks, start=1):
        query = build_overpass_query(
            settlement_rows=chunk,
            batch_index=batch_index,
        )

        output_path = OUTPUT_FOLDER / f"overpass_residual_institutions_batch_{batch_index:03d}.txt"

        with output_path.open("w", encoding="utf-8") as text_file:
            text_file.write(query)

    write_batch_metadata(settlement_chunks)

    print("Kész.")
    print(f"Residual települések száma: {len(residual_settlements)}")
    print(f"Query batch-ek száma: {len(settlement_chunks)}")
    print(f"Település / query: {SETTLEMENTS_PER_QUERY}")
    print(f"Output mappa: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    run_overpass_query_generation()