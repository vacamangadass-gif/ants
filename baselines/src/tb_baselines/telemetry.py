from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

# 不可达势能哨兵数值与无效梯度标记
FAR = int(np.int32(1 << 29))
INVALID_GRADIENT = FAR

# 势能源点类型数值编码
SOURCE_NONE = 0
SOURCE_FOOD_VISIBLE = 1
SOURCE_FOOD_MEMORY = 2
SOURCE_FRONTIER_UNSEEN = 3
SOURCE_HILL_FOE = 4

SOURCE_TYPE_NAMES: dict[int, str] = {
    SOURCE_NONE: "none",
    SOURCE_FOOD_VISIBLE: "food_visible",
    SOURCE_FOOD_MEMORY: "food_memory",
    SOURCE_FRONTIER_UNSEEN: "frontier_unseen",
    SOURCE_HILL_FOE: "hill_foe",
}

# 行优先四向移动偏移量与驻留动作编码
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOLD = 4


class RejectionCode(str, Enum):
    WATER = "WATER"
    FOOD = "FOOD"
    TAKEN = "TAKEN"
    GRADIENT_NON_DECREASING = "GRADIENT_NON_DECREASING"


class HoldReason(str, Enum):
    HARVEST_WAIT = "harvest_wait"
    DEFENSE = "defense"
    CONFLICT_WAIT = "conflict_wait"
    NO_VALID_ACTION = "no_valid_action"
    POTENTIAL_PLATEAU = "potential_plateau"
    NO_TARGET = "no_target"
    STALE_TARGET = "stale_target"


class SourceType(str, Enum):
    FOOD_VISIBLE = "food_visible"
    FOOD_MEMORY = "food_memory"
    FRONTIER_UNSEEN = "frontier_unseen"
    HILL_FOE = "hill_foe"
    NONE = "none"


@dataclass
class FieldMetadata:
    frontier_size: int
    frontier_nonempty_at_limit: bool
    source_grid: np.ndarray

    def source_type_at(self, r: int, c: int) -> str:
        code = int(self.source_grid[r, c])
        return SOURCE_TYPE_NAMES.get(code, "none")


def is_intentional_hold_reason(reason: HoldReason | str | None) -> bool:
    if reason is None:
        return False
    value = reason.value if isinstance(reason, HoldReason) else str(reason)
    return value in (HoldReason.HARVEST_WAIT.value, HoldReason.DEFENSE.value)


def compute_gradient_and_mask(
    potential_self: int,
    potentials_neighbors: list[int],
    water_neighbors: list[bool] | None = None,
) -> tuple[list[bool], list[int]]:
    valid_mask: list[bool] = []
    gradient: list[int] = []

    for k in range(len(potentials_neighbors)):
        p_neighbor = potentials_neighbors[k]
        is_water = water_neighbors[k] if water_neighbors is not None else False
        is_valid = (potential_self < FAR) and (p_neighbor < FAR) and (not is_water)
        valid_mask.append(is_valid)
        if is_valid:
            gradient.append(int(p_neighbor - potential_self))
        else:
            gradient.append(INVALID_GRADIENT)

    return valid_mask, gradient


def classify_hold_reason(
    is_holding: bool,
    pos: tuple[int, int],
    potential_self: int,
    potentials_neighbors: list[int],
    candidate_rejections: dict[int, str],
    visible_food_coords: set[tuple[int, int]],
    rows: int,
    cols: int,
    is_defending: bool = False,
    is_stale: bool = False,
) -> HoldReason | None:
    if not is_holding:
        return None

    if is_defending:
        return HoldReason.DEFENSE

    r, c = pos
    for dr, dc in DELTAS:
        nr = (r + dr) % rows
        nc = (c + dc) % cols
        if (nr, nc) in visible_food_coords:
            return HoldReason.HARVEST_WAIT

    if is_stale:
        return HoldReason.STALE_TARGET

    all_neighbors_far = all(p >= FAR for p in potentials_neighbors)
    if potential_self >= FAR and all_neighbors_far:
        return HoldReason.NO_TARGET

    has_taken = any(code == RejectionCode.TAKEN.value for code in candidate_rejections.values())
    has_strictly_lower_neighbor = any(
        potentials_neighbors[k] < potential_self
        for k in range(len(potentials_neighbors))
        if candidate_rejections.get(k) != RejectionCode.WATER.value
        and candidate_rejections.get(k) != RejectionCode.FOOD.value
    )

    if has_taken and has_strictly_lower_neighbor:
        return HoldReason.CONFLICT_WAIT

    terrain_rejection_count = sum(
        1
        for code in candidate_rejections.values()
        if code in (RejectionCode.WATER.value, RejectionCode.FOOD.value, RejectionCode.TAKEN.value)
    )
    if terrain_rejection_count == len(potentials_neighbors):
        return HoldReason.NO_VALID_ACTION

    if potential_self < FAR and not has_strictly_lower_neighbor:
        return HoldReason.POTENTIAL_PLATEAU

    return HoldReason.NO_VALID_ACTION


@dataclass
class TelemetryRecord:
    ant_id: int
    pos: tuple[int, int]
    potential_self: int
    potentials_neighbors: list[int]
    frontier_size: int
    frontier_nonempty_at_limit: bool
    nearest_source_type: str
    candidate_rejections: dict[int, str]
    hold_reason: str | None
    is_intentional_hold: bool
    valid_gradient_mask: list[bool]
    potential_gradient: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_ant_telemetry(
    ant_id: int,
    pos: tuple[int, int],
    dist: np.ndarray,
    field_meta: FieldMetadata,
    candidate_rejections: dict[int, str],
    action: int,
    visible_food: set[tuple[int, int]],
    friendly_hills: set[tuple[int, int]],
    is_stale: bool = False,
) -> TelemetryRecord:
    rows, cols = dist.shape
    ar, ac = pos
    pot_self = int(dist[ar, ac])
    pot_neighbors: list[int] = []
    water_neighbors: list[bool] = []

    for k in range(4):
        dr, dc = DELTAS[k]
        nr, nc = (ar + dr) % rows, (ac + dc) % cols
        pot_neighbors.append(int(dist[nr, nc]))
        water_neighbors.append(candidate_rejections.get(k) == RejectionCode.WATER.value)

    valid_mask, pot_gradient = compute_gradient_and_mask(
        pot_self, pot_neighbors, water_neighbors
    )

    is_holding = (action == HOLD)
    is_defending = ((ar, ac) in friendly_hills)

    hold_reason_enum = classify_hold_reason(
        is_holding=is_holding,
        pos=(ar, ac),
        potential_self=pot_self,
        potentials_neighbors=pot_neighbors,
        candidate_rejections=candidate_rejections,
        visible_food_coords=visible_food,
        rows=rows,
        cols=cols,
        is_defending=is_defending,
        is_stale=is_stale,
    )

    hold_reason_str = hold_reason_enum.value if hold_reason_enum is not None else None
    intentional = is_intentional_hold_reason(hold_reason_enum)

    return TelemetryRecord(
        ant_id=int(ant_id),
        pos=(ar, ac),
        potential_self=pot_self,
        potentials_neighbors=pot_neighbors,
        frontier_size=int(field_meta.frontier_size),
        frontier_nonempty_at_limit=bool(field_meta.frontier_nonempty_at_limit),
        nearest_source_type=field_meta.source_type_at(ar, ac),
        candidate_rejections={int(k): str(v) for k, v in candidate_rejections.items()},
        hold_reason=hold_reason_str,
        is_intentional_hold=intentional,
        valid_gradient_mask=valid_mask,
        potential_gradient=pot_gradient,
    )


class TelemetryCollector:
    def __init__(self) -> None:
        self.records: list[TelemetryRecord] = []

    def record_ant_decision(
        self,
        ant_id: int,
        pos: tuple[int, int],
        potential_self: int,
        potentials_neighbors: list[int],
        frontier_size: int,
        frontier_nonempty_at_limit: bool,
        nearest_source_type: str,
        candidate_rejections: dict[int, str],
        hold_reason: str | None,
        is_intentional_hold: bool,
        valid_gradient_mask: list[bool],
        potential_gradient: list[int],
    ) -> TelemetryRecord:
        record = TelemetryRecord(
            ant_id=int(ant_id),
            pos=(int(pos[0]), int(pos[1])),
            potential_self=int(potential_self),
            potentials_neighbors=[int(p) for p in potentials_neighbors],
            frontier_size=int(frontier_size),
            frontier_nonempty_at_limit=bool(frontier_nonempty_at_limit),
            nearest_source_type=str(nearest_source_type),
            candidate_rejections={int(k): str(v) for k, v in candidate_rejections.items()},
            hold_reason=str(hold_reason) if hold_reason is not None else None,
            is_intentional_hold=bool(is_intentional_hold),
            valid_gradient_mask=[bool(m) for m in valid_gradient_mask],
            potential_gradient=[int(g) for g in potential_gradient],
        )
        self.records.append(record)
        return record

    def export_turn_telemetry(self) -> list[TelemetryRecord]:
        return list(self.records)

    def clear(self) -> None:
        self.records.clear()

    def compute_turn_summary(self) -> dict[str, Any]:
        total = len(self.records)
        if total == 0:
            return {
                "total_ants": 0,
                "hold_count": 0,
                "intentional_hold_count": 0,
                "unintentional_hold_count": 0,
                "hold_reasons": {},
                "rejection_codes": {},
            }

        hold_count = 0
        intentional_hold_count = 0
        hold_reasons: dict[str, int] = {}
        rejection_codes: dict[str, int] = {}

        for rec in self.records:
            if rec.hold_reason is not None:
                hold_count += 1
                hold_reasons[rec.hold_reason] = hold_reasons.get(rec.hold_reason, 0) + 1
                if rec.is_intentional_hold:
                    intentional_hold_count += 1

            for code in rec.candidate_rejections.values():
                rejection_codes[code] = rejection_codes.get(code, 0) + 1

        return {
            "total_ants": total,
            "hold_count": hold_count,
            "intentional_hold_count": intentional_hold_count,
            "unintentional_hold_count": hold_count - intentional_hold_count,
            "hold_reasons": hold_reasons,
            "rejection_codes": rejection_codes,
        }
