# Pansim_full_agent_list_generation
Repository containing all the scripts and starting excel tables used for agent generation and assignment to residents

Synthetic agentlista átadási összefoglaló
1. Az állomány célja

Ez az állomány egy szintetikus, agent-alapú magyarországi populáció aktuális generált verziója a PanSim / COVID-szimulációs modellhez.

Az állomány célja, hogy agent-szinten tartalmazzon olyan demográfiai, háztartási, munkahelyi, iskolai és egészségi attribútumokat, amelyek később felhasználhatók járványterjedési vagy COVID-kockázati szimulációkban.

A jelenlegi agentlista validálandó baseline, nem végleges lezárt adatállomány. A validáció célja annak ellenőrzése, hogy a generált agentek eloszlásai mennyire térnek el a KSH / egyéb külső referenciaadatoktól.

A jelenlegi fájl sok köztes/módszertani oszlopot is tartalmaz. Validációhoz elsősorban a README “Mely oszlopokat érdemes validációhoz használni?” része alapján érdemes dolgozni; az assignment_method, source és old_worker_* típusú oszlopok inkább audit/debug célúak.

2. Fő fájl

Jelenlegi aktuális agentlista:

144_agents_with_refined_health_status.csv

Jelenlegi agentek száma:

9 597 574

Ebből az egészségi állapot assignmentben:

5+ agentek:        9 134 438
5 év alattiak:       463 136

Fontos: a teljes agentlista kismértékben eltér a 2022-es népszámlálási teljes népességtől. Ennek oka főleg a household-generálásból maradt kb. 6 ezer fős residual eltérés, amely olyan településekhez kötődik, ahol a népesség alapján lenne 6+ fős háztartási maradék, de a flat household táblában nincs 6+ fős háztartás.

3. Fájlformátum

A fő agentlista CSV formátumú.

A fájl mérete miatt nem javasolt Excelben vagy WPS-ben közvetlenül megnyitni. A teljes agentlista túl nagy egy Excel munkalaphoz, ezért validációhoz inkább Python / pandas / DuckDB / R használata javasolt.

Táblázatkezelőben csak kisebb mintafájlokat vagy aggregált validációs táblákat érdemes megnyitni.

4. Oszlopcsoportok

Az agentlista sok oszlopot tartalmaz, mert több generálási lépés eredménye egyesítve szerepel benne. Validációhoz nem minden oszlop egyformán fontos. Az oszlopok nagyjából az alábbi csoportokba sorolhatók.

A. Azonosítók
agent_id

Egyedi agentazonosító.

Példa:

AG_0000000001

Ezt érdemes használni, ha egy konkrét agentet vissza kell keresni.

household_id

Annak a háztartásnak az azonosítója, amelyhez az agent tartozik.

Példa:

HH_0000000001

Háztartásszintű validációhoz hasznos.

dwelling_id

Lakás / dwelling azonosító.

A jelenlegi modellben a dwelling és household réteg a household-generálásból származik. Lakásszintű vagy háztartásszintű aggregációknál lehet hasznos.

B. Földrajzi oszlopok
settlement_key

Normalizált településkulcs, amelyet belső illesztésekhez használtunk.

Validációhoz inkább technikai mező, nem elsődleges riportolási oszlop.

settlement_name

Település neve.

Példa:

Abaliget

Településszintű validációhoz ez az egyik legfontosabb oszlop.

settlement_ksh_code

Település KSH-kódja.

Ez a legbiztonságosabb településszintű összekötő kulcs külső KSH táblákkal.

county_name

Vármegye neve.

Példa:

Baranya
Budapest
Pest

Vármegyei aggregációhoz használható.

county_code

Vármegye kódja.

Külső táblákkal való gépi illesztéshez lehet hasznos.

settlement_type

Településtípus.

Példák:

Főváros
Megyei jogú város(ok)
Egyéb város(ok)
Község(ek)

Sok hierarchikus KSH tábla vármegye × településtípus bontásban volt megadva, ezért ez validációs szempontból fontos.

district_code, district_name

Járási kód és járásnév.

Ezek egyelőre inkább földrajzi kiegészítő változók. Ha járásszintű térképezés vagy aggregáció kell, akkor hasznosak lehetnek.

C. Háztartási oszlopok
household_size_numeric

A háztartás mérete, vagyis hány fő tartozik az adott householdhoz.

Validációhoz fontos, ha a háztartásméret-eloszlást akarjuk ellenőrizni.

agent_position_in_household

Az agent sorszáma a saját háztartásán belül.

Inkább technikai mező. Validációhoz általában nem szükséges.

broad_age_group

A household-generálási lépésből származó tág életkori csoport.

Példák:

under_30
age_30_64
age_65_plus

Ez nem a végleges pontos életkor, hanem egy korábbi háztartási kompozíciós kategória.

household_age_composition

A háztartás korösszetételének kategóriája.

Például azt írja le, hogy csak 30–64 éves személyek vannak-e a háztartásban, vagy van-e 65+ személy, 30 év alatti stb.

Háztartási validációhoz hasznos, agent-szintű demográfiai validációhoz kevésbé.

household_employment_composition

A háztartás foglalkoztatottsági összetételének kategóriája.

Példák jellegében:

Egy foglalkoztatott van a háztartásban
Nincs foglalkoztatott, de van munkanélküli / inaktív
Csak eltartott személyek vannak

Ez household-szintű kompozíciós targetből származik.

generation_rule

A household / agent generálási szabály technikai jelölése.

Validációhoz általában nem elsődleges, inkább módszertani debug célra való.

D. Nem és életkor
sex

Az agent neme.

Példák:

male
female

vagy magyar címkével, ha a konkrét fájlban így szerepel.

Fontos validációs oszlop.

exact_age

Az agent végleges pontos életkora.

Ez az életkori validáció egyik fő oszlopa.

age_calibrated

Jelzi, hogy az életkor-kiosztás a későbbi age calibration után szerepel.

Ez a jelenlegi agentlista validált / kalibrált életkori állapotához tartozik.

original_exact_age_before_age_calibration

Az agent eredeti pontos életkora az age calibration előtt.

Technikai / módszertani oszlop. Validációhoz általában nem ezt kell használni.

original_broad_age_group_before_age_calibration

Az agent eredeti tág életkori csoportja az age calibration előtt.

Technikai oszlop.

original_clean_age_group_before_age_calibration

Az eredeti, tisztított életkori kategória az age calibration előtt.

Technikai / debug oszlop.

calibrated_clean_age_group

A kalibráció utáni tisztított életkori kategória.

Ez aggregált age-group validációhoz hasznos lehet.

E. Gazdasági aktivitás és végzettség
activity_education_age_group

Az economic activity + education assignment során használt életkori csoport.

Technikai célú csoportosító oszlop.

economic_activity_status

A korai / household-alapú gazdasági aktivitási státusz.

Ezt nem feltétlenül érdemes végső validációhoz használni, mert később készült kalibrált aktivitási változat is.

economic_activity_status_calibrated

A kalibrált gazdasági aktivitási státusz.

Validációhoz ezt érdemes használni.

Példák:

employed
unemployed
inactive_benefit
dependent
under15

vagy a fájlban szereplő magyar/angol kategóriák szerint.

education_level_calibrated

Az agent kalibrált iskolai végzettsége.

A végzettségi validációhoz ezt érdemes használni.

activity_education_assignment_source

Az activity + education assignment forrása / módszertani jelölése.

Inkább technikai mező.

F. Munkahelyi / foglalkoztatási réteg
workplace_status

Az agent munkahelyi státusza.

Főbb kategóriák:

EMP_DOM
EMP_FOREIGN
NOT_EMP

Jelentés:

EMP_DOM      belföldön foglalkoztatott
EMP_FOREIGN  külföldön dolgozó foglalkoztatott
NOT_EMP      nem foglalkoztatott

Munkahelyi validációhoz ez az egyik legfontosabb oszlop.

workplace_assignment_type

Munkahely-hozzárendelés típusa.

Például jelölheti, hogy az agent belföldi munkahelyet kapott, külföldi worker státuszt kapott, vagy nem foglalkoztatott.

workplace_assignment_source

A munkahely-hozzárendelés forrása.

Technikai / módszertani mező.

foreign_workplace_assignment_method

Külföldi munkavégzés assignment módszere.

Példák:

EXACT
FALLBACK
NAPP

Általában validációs/debug célra hasznos.

workplace_id

A belföldi munkahely azonosítója.

Belföldi foglalkoztatottaknál van értelmes értéke. Nem foglalkoztatottaknál vagy külföldön dolgozóknál tipikusan placeholder érték szerepelhet.

workplace_country

Munkahely országa.

Például:

Hungary
Foreign
NAPP
workplace_county

Munkahely vármegyéje.

Belföldi munkahelyek validációjához használható.

workplace_settlement

Munkahely települése.

workplace_settlement_type

Munkahely településtípusa.

workplace_teaor_code

Munkahely TEÁOR / ágazati kódja.

Munkahelyi ágazati validációhoz fontos.

workplace_size

Munkahely méretkategóriája.

previous_worker_record_id

A korábbi worker/workplace layerből átvett rekord azonosítója.

Technikai mező, az old worker poolból való hozzárendelés követésére.

domestic_worker_assignment_method

Belföldi munkahely-hozzárendelés módszere.

Példák:

EXACT
FALLBACK
NAPP

Ez validáció/debug szempontból fontos lehet, de szimulációs inputként általában nem szükséges.

old_worker_* oszlopok

Ezek a korábbi workplace layerből átvett eredeti worker-attribútumok.

Példák:

old_worker_county
old_worker_sector_code
old_worker_gender
old_worker_age_group
old_worker_education
old_worker_used_fallback
old_worker_fallback_level
old_worker_used_teaor_fallback

Ezek főleg audit/debug célú oszlopok. A doktorandusz validációjához hasznosak lehetnek, ha azt akarja ellenőrizni, hogy a domestic workplace assignment hogyan használta fel a régi munkahelyi réteget. Általános agent-validációhoz nem ezek az elsődleges oszlopok.

G. Iskolába járás
school_attendance_status

Jelzi, hogy az agent iskolába jár-e.

Fő kategóriák:

SCHOOL
NO_SCHOOL
school_attendance_type

Ha az agent iskolába jár, akkor a képzés típusa.

Példák:

Alapfokú képzésre jár
BA/BSc képzésre jár
MA/MSc / osztatlan képzésre jár
Doktori képzésre jár
school_attendance_assignment_method

Az iskolába járás assignment módszere.

Példák:

EXACT
FALLBACK
NAPP

Validáció/debug célra hasznos.

school_attendance_assignment_source

Az iskolába járás assignment forrása.

Technikai / módszertani oszlop.

H. Egészségi állapot

A health assignment három külön attribútumként szerepel, mert a forrástábla is három külön egészségi kérdést tartalmazott.

health_disability_status

Fogyatékosság / súlyos korlátozottság státusz.

Lehetséges kategóriák jellegében:

Fogyatékossága van vagy súlyosan korlátozott
Nincs fogyatékossága és nem súlyosan korlátozott
Nem válaszolt a fogyatékossági és a korlátozottsági kérdésekre
UNDER_5_NOT_IN_SOURCE
health_chronic_disease_status

Tartós betegség státusz.

Lehetséges kategóriák:

Van tartós betegsége
Nincs tartós betegsége
Nem válaszolt a tartós betegségre vonatkozó kérdésre
UNDER_5_NOT_IN_SOURCE
health_limitation_status

Korlátozottsági szint.

Lehetséges kategóriák:

Nem korlátozott
Mérsékelten korlátozott
Súlyosan korlátozott
Nem válaszolt a korlátozottsági kérdésekre
UNDER_5_NOT_IN_SOURCE
health_*_assignment_method

A health assignment módszertani jelölése.

A refined verzióban az 5+ agenteknél jellemzően:

REFINED_AGE_CELL_WEIGHTED

Az 5 év alattiaknál:

NAPP
health_refinement_source

A health refinement forrása / típusa.

Például:

AGE_CELL_FIXED_MULTIDIM_WEIGHTED
UNDER_5_NOT_IN_SOURCE

Jelentése: az 5+ agenteknél a korcsoportos health target kemény constraint volt, és ezen belül történt súlyozott finomítás a nem, végzettség és gazdasági aktivitás marginális eloszlásaihoz.

5. Mely oszlopokat érdemes validációhoz használni?
Demográfia

Elsődleges:

settlement_ksh_code
settlement_name
county_name
settlement_type
sex
exact_age
calibrated_clean_age_group

Másodlagos / technikai:

original_exact_age_before_age_calibration
original_broad_age_group_before_age_calibration
original_clean_age_group_before_age_calibration
Háztartás

Elsődleges:

household_id
dwelling_id
household_size_numeric
household_age_composition
household_employment_composition

Technikai:

agent_position_in_household
generation_rule
Gazdasági aktivitás és végzettség

Elsődleges:

economic_activity_status_calibrated
education_level_calibrated

Kevésbé ajánlott végső validációhoz:

economic_activity_status

Ez korábbi / household-alapú státusz, a kalibrált változat pontosabb validációs célra.

Munkahely

Elsődleges:

workplace_status
workplace_country
workplace_county
workplace_settlement
workplace_settlement_type
workplace_teaor_code
workplace_size

Technikai / audit:

previous_worker_record_id
domestic_worker_assignment_method
foreign_workplace_assignment_method
old_worker_*
Iskola

Elsődleges:

school_attendance_status
school_attendance_type

Technikai:

school_attendance_assignment_method
school_attendance_assignment_source
Health

Elsődleges:

health_disability_status
health_chronic_disease_status
health_limitation_status

Technikai:

health_disability_assignment_method
health_chronic_disease_assignment_method
health_limitation_assignment_method
health_refinement_source
6. Fontos módszertani megjegyzések
6.1. Skálázott targetek

Több assignment-lépésnél a KSH source tábla totalja nem pontosan egyezett a generated agentállomány totaljával. Ilyenkor a targeteket a tényleges generated agentállományra skáláztuk.

Ez azt jelenti, hogy az arányokat próbáltuk megtartani, nem minden esetben az eredeti abszolút KSH darabszámot kényszerítettük rá az agentlistára.

6.2. Health assignment

A health assignmentnél a fő constraint:

vármegye × településtípus × health_age_group × health_question × response

Ez pontosan megmaradt.

Ezen belül történt egy súlyozott finomítás, hogy a konkrét agentek health címkéi jobban illeszkedjenek:

sex
education
economic_activity

marginális eloszlásokhoz is.

6.3. 5 év alattiak

A health source tábla 5 éves kortól indul, ezért az 5 év alatti agentek nem kaptak normál health response kategóriát. Ők:

UNDER_5_NOT_IN_SOURCE

jelölést kaptak.

6.4. NAPP értékek

A NAPP placeholder azt jelzi, hogy az adott oszlop az adott agentre nem alkalmazható.

Például egy nem foglalkoztatott agentnél nincs értelmezhető belföldi munkahelyi TEÁOR-kód.
