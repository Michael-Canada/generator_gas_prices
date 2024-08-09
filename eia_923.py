from typing import Optional, NamedTuple
from datetime import datetime
from collections import defaultdict

import pytz

from placebo.scenario_builder.model.supply_curve import (
    # MultiCoalTypePriceManager,
    Hub,
    Coordinates,
    CoalPrice,
    CoalType,
)
from placebo.scenario_builder.model.resource_universe import FuelType
from placebo.utils import storage_utils, arrow_reading, date_utils


class FuelPurchaseData(NamedTuple):
    year: int
    month: int
    plant_id: int
    plant_name: str
    plant_state: str

    energy_source: FuelType
    fuel_cost: Optional[float]


_STATE_CENTERS = {
    "AK": Coordinates(63.588753, -154.493062),
    "AL": Coordinates(32.318231, -86.902298),
    "AR": Coordinates(34.969704, -92.373123),
    "AZ": Coordinates(34.048928, -111.093731),
    "CA": Coordinates(36.778259, -119.417931),
    "CO": Coordinates(39.550051, -105.782067),
    "CT": Coordinates(41.603221, -73.087749),
    "DE": Coordinates(38.910832, -75.52767),
    "FL": Coordinates(27.664827, -81.515754),
    "GA": Coordinates(32.157435, -82.907123),
    "HI": Coordinates(19.896766, -155.582782),
    "IA": Coordinates(41.878003, -93.097702),
    "ID": Coordinates(44.068202, -114.742041),
    "IL": Coordinates(40.633125, -89.398528),
    "IN": Coordinates(40.551217, -85.602364),
    "KS": Coordinates(39.011902, -98.484246),
    "KY": Coordinates(37.839333, -84.270018),
    "LA": Coordinates(30.984298, -91.962333),
    "MA": Coordinates(42.407211, -71.382437),
    "MD": Coordinates(39.045755, -76.641271),
    "ME": Coordinates(45.253783, -69.445469),
    "MI": Coordinates(44.314844, -85.602364),
    "MN": Coordinates(46.729553, -94.6859),
    "MO": Coordinates(37.964253, -91.831833),
    "MS": Coordinates(32.354668, -89.398528),
    "MT": Coordinates(46.879682, -110.362566),
    "NC": Coordinates(35.759573, -79.0193),
    "ND": Coordinates(47.551493, -101.002012),
    "NE": Coordinates(41.492537, -99.901813),
    "NH": Coordinates(43.193852, -71.572395),
    "NJ": Coordinates(40.058324, -74.405661),
    "NM": Coordinates(34.97273, -105.032363),
    "NV": Coordinates(38.80261, -116.419389),
    "NY": Coordinates(43.299428, -74.217933),
    "OH": Coordinates(40.417287, -82.907123),
    "OK": Coordinates(35.007752, -97.092877),
    "OR": Coordinates(43.804133, -120.554201),
    "PA": Coordinates(41.203322, -77.194525),
    "RI": Coordinates(41.580095, -71.477429),
    "SC": Coordinates(33.836081, -81.163725),
    "SD": Coordinates(43.969515, -99.901813),
    "TN": Coordinates(35.517491, -86.580447),
    "TX": Coordinates(31.968599, -99.901813),
    "UT": Coordinates(39.32098, -111.093731),
    "VA": Coordinates(37.431573, -78.656894),
    "WA": Coordinates(47.751074, -120.740139),
    "WI": Coordinates(43.78444, -88.787868),
    "WV": Coordinates(38.597626, -80.454903),
    "WY": Coordinates(43.075968, -107.290284),
}


# def get_coal_price_manager(uri: str) -> MultiCoalTypePriceManager:
#     purschase_records = list(
#         arrow_reading.column_zip(
#             storage_utils.fetch_table(uri.strip()),
#             FuelPurchaseData,
#         )
#     )
#     hub_definitions = [
#         Hub(uid=state, coordinates=_STATE_CENTERS[state])
#         for state in {p.plant_state for p in purschase_records}
#     ]

#     prices = [
#         CoalPrice(
#             coal_type=CoalType[p.energy_source.name],
#             timestamp=date_utils.localized_dtm(
#                 datetime(p.year, p.month, 1).astimezone(pytz.UTC)
#             ),
#             # eia report express fuel cost in cents
#             price=p.fuel_cost / 100,
#             hub_uid=p.plant_state,
#         )
#         for p in purschase_records
#         if p.energy_source in {FuelType.BIT, FuelType.SUB, FuelType.LIG}
#         and p.fuel_cost is not None
#     ]

#     prices_by_key = defaultdict(list)
#     for price in prices:
#         prices_by_key[(price.coal_type, price.timestamp, price.hub_uid)].append(price)

#     aggregated_prices = []
#     for key, agg_prices in prices_by_key.items():
#         coal_type, timestamp, hub_uid = key

#         aggregated_prices.append(
#             CoalPrice(
#                 coal_type=coal_type,
#                 timestamp=timestamp,
#                 price=sum(p.price for p in agg_prices) / len(agg_prices),
#                 hub_uid=hub_uid,
#             )
#         )

#     manager = MultiCoalTypePriceManager(
#         prices=aggregated_prices, hub_definition=hub_definitions
#     )

#     return manager


# if __name__ == "__main__":
#     from placebo.utils import process_utils, date_utils

#     process_utils.init()

#     path = "/Users/adrienkergastel/Documents/draft/output_purchase.parquet"
#     manager = get_coal_price_manager(path)

#     price = manager.get_price(
#         dtm=date_utils.localized_from_isoformat("2024-06-20T00:00:00-00:00"),
#         coordinates=Coordinates(
#             latitude=41.803436,
#             longitude=-87.910928,
#         ),
#         coal_type=CoalType.LIG,
#     )