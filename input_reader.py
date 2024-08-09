import sys
sys.path.append('/Users/michael.simantov/Documents/mu-sourcing/sourcing')
sys.path.append('/Users/michael.simantov/Documents/mu-sourcing/sourcing/utils')
sys.path.append('/Users/michael.simantov/Documents/mu-sourcing')
sys.path.append('/Users/michael.simantov/Documents/mu-placebo-api')
sys.path.append('/Users/michael.simantov/Documents/mu-placebo-api/placebo_api')
sys.path.append('/Users/michael.simantov/Documents/mu-placebo-api/placebo_api/utils')
sys.path.append('/Users/michael.simantov/Documents/mu-sourcing/sourcing/reading/commons')

from typing import NamedTuple, Optional, IO, List, Sequence, Dict
from enum import Enum
import logging
from datetime import date

# from sourcing.reading.commons import excel_commons as excel
import excel_commons as excel
from utils.date_utils import NaiveDateTime

_LOGGER = logging.getLogger(__name__)


class SectorName(Enum):
    ELECTRIC_UTILITY = "Electric Utility"
    NAICS_22_NON_COGEN = "NAICS-22 Non-Cogen"
    NAICS_22_COGEN = "NAICS-22 Cogen"
    INDUSTRIAL_NAICS_COGEN = "Industrial NAICS Cogen"
    COMMERCIAL_NAICS_COGEN = "Commercial NAICS Cogen"
    COMMERCIAL_NAICS_NON_COGEN = "Commercial NAICS Non-Cogen"
    INDUSTRIAL_NAICS_NON_COGEN = "Industrial NAICS Non-Cogen"

    @staticmethod
    def from_string(value: str) -> "SectorName":
        return {sn.value: sn for sn in SectorName}[value.strip()]


class PrimeMover(Enum):
    CA = "CA"
    CT = "CT"
    ST = "ST"
    GT = "GT"
    PS = "PS"
    CS = "CS"
    IC = "IC"
    BA = "BA"
    PV = "PV"
    HY = "HY"
    WT = "WT"
    OT = "OT"
    BT = "BT"
    CP = "CP"
    WS = "WS"
    FC = "FC"
    CE = "CE"
    FW = "FW"
    HA = "HA"


class FuelType(Enum):
    NG = "NG"
    BIT = "BIT"
    SUB = "SUB"
    DFO = "DFO"
    NUC = "NUC"
    RC = "RC"
    WAT = "WAT"
    RFO = "RFO"
    LFG = "LFG"
    PC = "PC"
    MWH = "MWH"
    SUN = "SUN"
    OBG = "OBG"
    GEO = "GEO"
    LIG = "LIG"
    OG = "OG"
    KER = "KER"
    WDS = "WDS"
    PG = "PG"
    ANT = "ANT"
    OTH = "OTH"
    WND = "WND"
    JF = "JF"
    SGC = "SGC"
    WO = "WO"
    TDF = "TDF"
    MSB = "MSB"
    MSN = "MSN"
    OBS = "OBS"
    WC = "WC"
    SC = "SC"
    AB = "AB"
    BLQ = "BLQ"
    WH = "WH"
    OBL = "OBL"
    SLW = "SLW"
    BFG = "BFG"
    PUR = "PUR"
    SGP = "SGP"
    H2 = "H2"
    WDL = "WDL"


class AerFuelCode(Enum):
    NG = "NG"
    COL = "COL"
    DFO = "DFO"
    NUC = "NUC"
    HPS = "HPS"
    RFO = "RFO"
    MLG = "MLG"
    PC = "PC"
    OTH = "OTH"
    SUN = "SUN"
    HYC = "HYC"
    ORW = "ORW"
    GEO = "GEO"
    OOG = "OOG"
    WOO = "WOO"
    WWW = "WWW"
    WND = "WND"
    WOC = "WOC"


class PhysicalUnitLabel(Enum):
    MCF = "mcf"
    SHORT_TONS = "short tons"
    BARRELS = "barrels"
    MEGAWATTHOURS = "megawatthours"


# pylint: disable=invalid-name
class PlantData(NamedTuple):
    plant_id: int
    combined_heat_and_power_plant: bool
    nuclear_unit_id: Optional[int]
    plant_name: str
    operator_name: Optional[str]
    operator_id: Optional[int]
    plant_state: str
    census_region: str
    nerc_region: Optional[str]
    naics_code: int
    eia_sector_number: int
    sector_name: SectorName
    reported_prime_mover: PrimeMover
    reported_fuel_type_code: FuelType
    aer_fuel_type_code: AerFuelCode
    balancing_authority_code: Optional[str]
    physical_unit_label: Optional[PhysicalUnitLabel]

    # Total Quantity Consumed In Physical Units (Consumed For Electric Generation And Useful Thermal Output)
    quantity_january: Optional[float]
    quantity_february: Optional[float]
    quantity_march: Optional[float]
    quantity_april: Optional[float]
    quantity_may: Optional[float]
    quantity_june: Optional[float]
    quantity_july: Optional[float]
    quantity_august: Optional[float]
    quantity_september: Optional[float]
    quantity_october: Optional[float]
    quantity_november: Optional[float]
    quantity_december: Optional[float]

    # Quantity Consumed In Physical Units For Electric Generation
    elec_quantity_january: Optional[float]
    elec_quantity_february: Optional[float]
    elec_quantity_march: Optional[float]
    elec_quantity_april: Optional[float]
    elec_quantity_may: Optional[float]
    elec_quantity_june: Optional[float]
    elec_quantity_july: Optional[float]
    elec_quantity_august: Optional[float]
    elec_quantity_september: Optional[float]
    elec_quantity_october: Optional[float]
    elec_quantity_november: Optional[float]
    elec_quantity_december: Optional[float]

    # Heat Content Of Fuels (MMBtu Per Unit)
    mmbtu_per_unit_january: Optional[float]
    mmbtu_per_unit_february: Optional[float]
    mmbtu_per_unit_march: Optional[float]
    mmbtu_per_unit_april: Optional[float]
    mmbtu_per_unit_may: Optional[float]
    mmbtu_per_unit_june: Optional[float]
    mmbtu_per_unit_july: Optional[float]
    mmbtu_per_unit_august: Optional[float]
    mmbtu_per_unit_september: Optional[float]
    mmbtu_per_unit_october: Optional[float]
    mmbtu_per_unit_november: Optional[float]
    mmbtu_per_unit_december: Optional[float]

    # Total Fuel Consumed (MMBtu)
    tot_mmbtu_january: Optional[float]
    tot_mmbtu_february: Optional[float]
    tot_mmbtu_march: Optional[float]
    tot_mmbtu_april: Optional[float]
    tot_mmbtu_may: Optional[float]
    tot_mmbtu_june: Optional[float]
    tot_mmbtu_july: Optional[float]
    tot_mmbtu_august: Optional[float]
    tot_mmbtu_september: Optional[float]
    tot_mmbtu_october: Optional[float]
    tot_mmbtu_november: Optional[float]
    tot_mmbtu_december: Optional[float]

    # Quantity Consumed For Electricity (MMBtu)
    elec_mmbtu_january: Optional[float]
    elec_mmbtu_february: Optional[float]
    elec_mmbtu_march: Optional[float]
    elec_mmbtu_april: Optional[float]
    elec_mmbtu_may: Optional[float]
    elec_mmbtu_june: Optional[float]
    elec_mmbtu_july: Optional[float]
    elec_mmbtu_august: Optional[float]
    elec_mmbtu_september: Optional[float]
    elec_mmbtu_october: Optional[float]
    elec_mmbtu_november: Optional[float]
    elec_mmbtu_december: Optional[float]

    # Electricity Net Generation (MWh)
    netgen_january: Optional[float]
    netgen_february: Optional[float]
    netgen_march: Optional[float]
    netgen_april: Optional[float]
    netgen_may: Optional[float]
    netgen_june: Optional[float]
    netgen_july: Optional[float]
    netgen_august: Optional[float]
    netgen_september: Optional[float]
    netgen_october: Optional[float]
    netgen_november: Optional[float]
    netgen_december: Optional[float]

    # Year-To-Date
    total_fuel_consumption_quantity: float
    electric_fuel_consumption_quantity: float
    total_fuel_consumption_mmbtu: float
    elec_fuel_consumption_mmbtu: float
    net_generation_mwh: float

    year: int


# pylint: enable=invalid-name

_EXPECTED_SHEETS = [
    "Page 1 Generation and Fuel Data",
    "Page 1 Puerto Rico",
    "Page 1 Energy Storage",
    "Page 2 Stocks Data",
    "Page 2 Oil Stocks Data",
    "Page 3 Boiler Fuel Data",
    "Page 4 Generator Data",
    "Page 5 Fuel Receipts and Costs",
    "Page 6 Plant Frame",
    "Page 6 Plant Frame Puerto Rico",
    "Page 7 File Layout",
]


def _parse_bool(value: str) -> bool:
    assert value in {"Y", "N"}, value
    return value == "Y"


def _parse_regulated(value: str) -> bool:
    assert value in {"REG", "UNR"}, value
    return value == "REG"


def _parse_float_or_none(cell: excel.Cell) -> Optional[float]:
    if cell.value() == "" or cell.value() == ".":
        return None

    return float(cell.float_parse())


def _parse_int_or_none(cell: excel.Cell) -> Optional[int]:
    if cell.value() == "" or cell.value() == ".":
        return None

    return int(cell.value())


def _parse_date_or_none(cell: excel.Cell) -> Optional[date]:
    if cell.value() == "" or cell.value() == ".":
        return None

    value = str(cell.ensure_int())

    year = int(value[-2:])
    month = int(value[:-2])

    return date(year, month, 1)


def parse_row_plant_data(
    row: Sequence[excel.Cell],
    header_mapper: Dict[str, int],
) -> PlantData:

    physical_unit_label = None
    raw_value = row[header_mapper["Physical\nUnit Label"]].value()
    if raw_value != "" and raw_value is not None:
        physical_unit_label = PhysicalUnitLabel[
            row[header_mapper["Physical\nUnit Label"]]
            .non_empty_string()
            .replace(" ", "_")
            .upper()
        ]

    return PlantData(
        plant_id=row[header_mapper["Plant Id"]].ensure_int(),
        combined_heat_and_power_plant=_parse_bool(
            row[header_mapper["Combined Heat And\nPower Plant"]].non_empty_string()
        ),
        nuclear_unit_id=_parse_int_or_none(row[header_mapper["Nuclear Unit Id"]]),
        plant_name=row[header_mapper["Plant Name"]].non_empty_string(),
        operator_name=row[header_mapper["Operator Name"]].str_parse_or_none(),
        operator_id=row[header_mapper["Operator Id"]].ensure_int_or_none(),
        plant_state=row[header_mapper["Plant State"]].non_empty_string(),
        census_region=row[header_mapper["Census Region"]].non_empty_string(),
        nerc_region=row[header_mapper["NERC Region"]].value(),
        naics_code=row[header_mapper["NAICS Code"]].ensure_int(),
        eia_sector_number=row[header_mapper["EIA Sector Number"]].ensure_int(),
        sector_name=SectorName.from_string(
            row[header_mapper["Sector Name"]].non_empty_string()
        ),
        reported_prime_mover=PrimeMover[
            row[header_mapper["Reported\nPrime Mover"]].non_empty_string()
        ],
        reported_fuel_type_code=FuelType[
            row[header_mapper["Reported\nFuel Type Code"]].non_empty_string()
        ],
        aer_fuel_type_code=AerFuelCode[
            row[header_mapper["MER\nFuel Type Code"]].non_empty_string()
        ],
        balancing_authority_code=row[
            header_mapper["Balancing\nAuthority Code"]
        ].str_parse_or_none(),
        physical_unit_label=physical_unit_label,
        quantity_january=_parse_float_or_none(row[header_mapper["Quantity\nJanuary"]]),
        quantity_february=_parse_float_or_none(
            row[header_mapper["Quantity\nFebruary"]]
        ),
        quantity_march=_parse_float_or_none(row[header_mapper["Quantity\nMarch"]]),
        quantity_april=_parse_float_or_none(row[header_mapper["Quantity\nApril"]]),
        quantity_may=_parse_float_or_none(row[header_mapper["Quantity\nMay"]]),
        quantity_june=_parse_float_or_none(row[header_mapper["Quantity\nJune"]]),
        quantity_july=_parse_float_or_none(row[header_mapper["Quantity\nJuly"]]),
        quantity_august=_parse_float_or_none(row[header_mapper["Quantity\nAugust"]]),
        quantity_september=_parse_float_or_none(
            row[header_mapper["Quantity\nSeptember"]]
        ),
        quantity_october=_parse_float_or_none(row[header_mapper["Quantity\nOctober"]]),
        quantity_november=_parse_float_or_none(
            row[header_mapper["Quantity\nNovember"]]
        ),
        quantity_december=_parse_float_or_none(
            row[header_mapper["Quantity\nDecember"]]
        ),
        elec_quantity_january=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nJanuary"]]
        ),
        elec_quantity_february=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nFebruary"]]
        ),
        elec_quantity_march=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nMarch"]]
        ),
        elec_quantity_april=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nApril"]]
        ),
        elec_quantity_may=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nMay"]]
        ),
        elec_quantity_june=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nJune"]]
        ),
        elec_quantity_july=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nJuly"]]
        ),
        elec_quantity_august=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nAugust"]]
        ),
        elec_quantity_september=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nSeptember"]]
        ),
        elec_quantity_october=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nOctober"]]
        ),
        elec_quantity_november=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nNovember"]]
        ),
        elec_quantity_december=_parse_float_or_none(
            row[header_mapper["Elec_Quantity\nDecember"]]
        ),
        mmbtu_per_unit_january=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nJanuary"]]
        ),
        mmbtu_per_unit_february=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nFebruary"]]
        ),
        mmbtu_per_unit_march=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nMarch"]]
        ),
        mmbtu_per_unit_april=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nApril"]]
        ),
        mmbtu_per_unit_may=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nMay"]]
        ),
        mmbtu_per_unit_june=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nJune"]]
        ),
        mmbtu_per_unit_july=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nJuly"]]
        ),
        mmbtu_per_unit_august=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nAugust"]]
        ),
        mmbtu_per_unit_september=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nSeptember"]]
        ),
        mmbtu_per_unit_october=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nOctober"]]
        ),
        mmbtu_per_unit_november=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nNovember"]]
        ),
        mmbtu_per_unit_december=_parse_float_or_none(
            row[header_mapper["MMBtuPer_Unit\nDecember"]]
        ),
        tot_mmbtu_january=_parse_float_or_none(
            row[header_mapper["Tot_MMBtu\nJanuary"]]
        ),
        tot_mmbtu_february=_parse_float_or_none(
            row[header_mapper["Tot_MMBtu\nFebruary"]]
        ),
        tot_mmbtu_march=_parse_float_or_none(row[header_mapper["Tot_MMBtu\nMarch"]]),
        tot_mmbtu_april=_parse_float_or_none(row[header_mapper["Tot_MMBtu\nApril"]]),
        tot_mmbtu_may=_parse_float_or_none(row[header_mapper["Tot_MMBtu\nMay"]]),
        tot_mmbtu_june=_parse_float_or_none(row[header_mapper["Tot_MMBtu\nJune"]]),
        tot_mmbtu_july=_parse_float_or_none(row[header_mapper["Tot_MMBtu\nJuly"]]),
        tot_mmbtu_august=_parse_float_or_none(row[header_mapper["Tot_MMBtu\nAugust"]]),
        tot_mmbtu_september=_parse_float_or_none(
            row[header_mapper["Tot_MMBtu\nSeptember"]]
        ),
        tot_mmbtu_october=_parse_float_or_none(
            row[header_mapper["Tot_MMBtu\nOctober"]]
        ),
        tot_mmbtu_november=_parse_float_or_none(
            row[header_mapper["Tot_MMBtu\nNovember"]]
        ),
        tot_mmbtu_december=_parse_float_or_none(
            row[header_mapper["Tot_MMBtu\nDecember"]]
        ),
        elec_mmbtu_january=_parse_float_or_none(
            row[header_mapper["Elec_MMBtu\nJanuary"]]
        ),
        elec_mmbtu_february=_parse_float_or_none(
            row[header_mapper["Elec_MMBtu\nFebruary"]]
        ),
        elec_mmbtu_march=_parse_float_or_none(row[header_mapper["Elec_MMBtu\nMarch"]]),
        elec_mmbtu_april=_parse_float_or_none(row[header_mapper["Elec_MMBtu\nApril"]]),
        elec_mmbtu_may=_parse_float_or_none(row[header_mapper["Elec_MMBtu\nMay"]]),
        elec_mmbtu_june=_parse_float_or_none(row[header_mapper["Elec_MMBtu\nJune"]]),
        elec_mmbtu_july=_parse_float_or_none(row[header_mapper["Elec_MMBtu\nJuly"]]),
        elec_mmbtu_august=_parse_float_or_none(
            row[header_mapper["Elec_MMBtu\nAugust"]]
        ),
        elec_mmbtu_september=_parse_float_or_none(
            row[header_mapper["Elec_MMBtu\nSeptember"]]
        ),
        elec_mmbtu_october=_parse_float_or_none(
            row[header_mapper["Elec_MMBtu\nOctober"]]
        ),
        elec_mmbtu_november=_parse_float_or_none(
            row[header_mapper["Elec_MMBtu\nNovember"]]
        ),
        elec_mmbtu_december=_parse_float_or_none(
            row[header_mapper["Elec_MMBtu\nDecember"]]
        ),
        netgen_january=_parse_float_or_none(row[header_mapper["Netgen\nJanuary"]]),
        netgen_february=_parse_float_or_none(row[header_mapper["Netgen\nFebruary"]]),
        netgen_march=_parse_float_or_none(row[header_mapper["Netgen\nMarch"]]),
        netgen_april=_parse_float_or_none(row[header_mapper["Netgen\nApril"]]),
        netgen_may=_parse_float_or_none(row[header_mapper["Netgen\nMay"]]),
        netgen_june=_parse_float_or_none(row[header_mapper["Netgen\nJune"]]),
        netgen_july=_parse_float_or_none(row[header_mapper["Netgen\nJuly"]]),
        netgen_august=_parse_float_or_none(row[header_mapper["Netgen\nAugust"]]),
        netgen_september=_parse_float_or_none(row[header_mapper["Netgen\nSeptember"]]),
        netgen_october=_parse_float_or_none(row[header_mapper["Netgen\nOctober"]]),
        netgen_november=_parse_float_or_none(row[header_mapper["Netgen\nNovember"]]),
        netgen_december=_parse_float_or_none(row[header_mapper["Netgen\nDecember"]]),
        total_fuel_consumption_quantity=row[
            header_mapper["Total Fuel Consumption\nQuantity"]
        ].float_parse(),
        electric_fuel_consumption_quantity=row[
            header_mapper["Electric Fuel Consumption\nQuantity"]
        ].float_parse(),
        total_fuel_consumption_mmbtu=row[
            header_mapper["Total Fuel Consumption\nMMBtu"]
        ].float_parse(),
        elec_fuel_consumption_mmbtu=row[
            header_mapper["Elec Fuel Consumption\nMMBtu"]
        ].float_parse(),
        net_generation_mwh=row[
            header_mapper["Net Generation\n(Megawatthours)"]
        ].float_parse(),
        year=row[header_mapper["YEAR"]].ensure_int(),
    )


def read_generation_and_fuel_data(bytesio: IO[bytes]) -> List[PlantData]:
    xlwb = excel.open_workbook(bytesio)

    # assert xlwb.sheet_names() == _EXPECTED_SHEETS

    sheet = xlwb.sheet_by_name(_EXPECTED_SHEETS[0])
    rows_iter = sheet.iter_rows(0, sheet.nrows())

    header = None
    while True:
        row = next(rows_iter)  # type: ignore    # zzz

        if row[1].value() == "Plant Id":
            header = row
            break

    assert header is not None
    header_mapper = {f.non_empty_string(): idx for idx, f in enumerate(header)}

    records = []
    for ind, xlrow in enumerate(rows_iter):
        try:
            records.append(parse_row_plant_data(xlrow, header_mapper))
        except Exception as exception:
            _LOGGER.error("Problem with row ", ind, ": %s'", str(xlrow))
            # raise exception

    return records


# pylint: disable=invalid-name
class FuelPurchaseData(NamedTuple):
    year: int
    month: int
    plant_id: int
    plant_name: str
    plant_state: str
    purchase_type: Optional[str]
    contract_expiration_date: Optional[date]
    energy_source: FuelType
    fuel_group: str
    coalmine_type: Optional[str]
    coalmine_state: Optional[str]
    coalmine_county: Optional[str]
    coalmine_msha_id: Optional[str]
    coalmine_name: Optional[str]
    supplier: str
    quantity: float
    average_heat_content: float
    average_sulfur_content: float
    average_ash_content: float
    average_mercury_content: float
    fuel_cost: Optional[float]
    regulated: bool
    operator_name: Optional[str]
    operator_id: int
    primary_transportation_mode: Optional[str]
    secondary_transportation_mode: Optional[str]
    natural_gas_supply_contract_type: Optional[str]
    natural_gas_delivery_contract_type: Optional[str]
    moisture_content: Optional[float]
    chlorine_content: Optional[float]
    ba_code: Optional[str]


# pylint: enable=invalid-name


def _parse_coalmine_county(cell: excel.Cell) -> Optional[str]:
    value = cell.value()
    if value == "" or value is None:
        return None

    return str(value)


def _parse_str(cell: excel.Cell) -> str:
    val = str(cell.value())
    assert val.strip() != ""

    return val


def parse_row_fuel_purchase_data(
    row: Sequence[excel.Cell],
    header_mapper: Dict[str, int],
) -> FuelPurchaseData:
    return FuelPurchaseData(
        year=row[header_mapper["YEAR"]].ensure_int(),
        month=row[header_mapper["MONTH"]].ensure_int(),
        plant_id=row[header_mapper["Plant Id"]].ensure_int(),
        plant_name=row[header_mapper["Plant Name"]].non_empty_string(),
        plant_state=row[header_mapper["Plant State"]].non_empty_string(),
        purchase_type=row[header_mapper["Purchase Type"]].str_parse_or_none(),
        contract_expiration_date=_parse_date_or_none(
            row[header_mapper["Contract\nExpiration Date"]]
        ),
        energy_source=FuelType[row[header_mapper["ENERGY_SOURCE"]].non_empty_string()],
        fuel_group=row[header_mapper["FUEL_GROUP"]].non_empty_string(),
        coalmine_type=row[header_mapper["Coalmine\nType"]].str_parse_or_none(),
        coalmine_state=row[header_mapper["Coalmine\nState"]].str_parse_or_none(),
        coalmine_county=_parse_coalmine_county(row[header_mapper["Coalmine\nCounty"]]),
        coalmine_msha_id=row[header_mapper["Coalmine\nMsha Id"]].str_parse_or_none(),
        coalmine_name=row[header_mapper["Coalmine\nName"]].str_parse_or_none(),
        supplier=_parse_str(row[header_mapper["SUPPLIER"]]),
        quantity=row[header_mapper["QUANTITY"]].float_parse(),
        average_heat_content=row[header_mapper["Average Heat\nContent"]].float_parse(),
        average_sulfur_content=row[
            header_mapper["Average Sulfur\nContent"]
        ].float_parse(),
        average_ash_content=row[header_mapper["Average Ash\nContent"]].float_parse(),
        average_mercury_content=row[
            header_mapper["Average Mercury\nContent"]
        ].float_parse(),
        fuel_cost=_parse_float_or_none(row[header_mapper["FUEL_COST"]]),
        regulated=_parse_regulated(row[header_mapper["Regulated"]].non_empty_string()),
        operator_name=row[header_mapper["Operator Name"]].str_parse_or_none(),
        operator_id=row[header_mapper["Operator Id"]].ensure_int(),
        primary_transportation_mode=row[
            header_mapper["Primary Transportation Mode"]
        ].str_parse_or_none(),
        secondary_transportation_mode=row[
            header_mapper["Secondary Transportation Mode"]
        ].str_parse_or_none(),
        natural_gas_supply_contract_type=row[
            header_mapper["Natural Gas Supply Contract Type"]
        ].str_parse_or_none(),
        natural_gas_delivery_contract_type=row[
            header_mapper["Natural Gas Delivery Contract Type"]
        ].str_parse_or_none(),
        moisture_content=_parse_float_or_none(row[header_mapper["Moisture\nContent"]]),
        chlorine_content=_parse_float_or_none(row[header_mapper["Chlorine\nContent"]]),
        # ba_code=row[header_mapper["BA_CODE"]].str_parse_or_none(),
        ba_code=row[header_mapper["Balancing\nAuthority Code"]].str_parse_or_none(),
        # ba_code=0,
    )


def read_fuel_receipts_and_costs(bytesio: IO[bytes]) -> List[FuelPurchaseData]:
    xlwb = excel.open_workbook(bytesio)

    assert xlwb.sheet_names() == _EXPECTED_SHEETS

    sheet = xlwb.sheet_by_name("Page 5 Fuel Receipts and Costs")
    rows_iter = sheet.iter_rows(0, sheet.nrows())

    header = None
    while True:
        row = next(rows_iter)  # type: ignore   # zzz

        if row[1].value() == "YEAR":
            header = row
            break

    assert header is not None
    header_mapper = {f.non_empty_string(): idx for idx, f in enumerate(header)}

    records = []
    for xlrow in rows_iter:
        try:
            records.append(parse_row_fuel_purchase_data(xlrow, header_mapper))
        except Exception as exception:
            _LOGGER.error("Problem with row '%s'", [v.value() for v in xlrow])
            raise exception

    return records


def _main() -> None:
    import sys

    for file in sys.argv[1:]:

        with open(file, "rb") as fdesc:
            result = read_generation_and_fuel_data(fdesc)
            print(result[:3])


if __name__ == "__main__":
    
    TESTING = True
    
    if TESTING:
        file = "EIA923_Schedules_2_3_4_5_M_12_2023_Early_Release.xlsx"

        if False:
            #  read generation_and_fuel_data
            with open(file, "rb") as fdesc:
                result = read_generation_and_fuel_data(fdesc)
                print(result[:3])

        #  read fuel receipts and costs
        with open(file, "rb") as fdesc:
            result = read_fuel_receipts_and_costs(fdesc)
            print(result[:3])

    # _main()