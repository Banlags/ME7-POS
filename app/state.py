import csv
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# =========================
# Data models
# =========================

@dataclass
class POSItem:
    """Represents a single line item on the receipt."""
    item_id: str        # YOLO class_id as string
    name: str
    unit_price: float
    quantity: int = 1

    @property
    def subtotal(self) -> float:
        return self.unit_price * self.quantity


@dataclass
class POSSession:
    """Represents a single POS session (one customer)."""
    active: bool = False
    items: List[POSItem] = field(default_factory=list)

    @property
    def total_amount(self) -> float:
        return sum(item.subtotal for item in self.items)


@dataclass
class ItemDefinition:
    """Definition of an item from the catalog (items.csv)."""
    class_id: int
    class_name: str
    display_name: str
    price: float


# =========================
# Global state
# =========================

# Current POS session
pos_session = POSSession()

# Item catalog loaded from items.csv
ITEM_CATALOG: Dict[int, ItemDefinition] = {}

# For debouncing detections:
_last_detected_item_key: Optional[str] = None
_last_detection_time: float = 0.0
DETECTION_DEBOUNCE_SECONDS = 0.0  # using "enter frame" logic mainly
_last_frame_had_item: bool = False

# For session summary when ending a session
_last_session_total: float = 0.0
_last_session_item_count: int = 0


# =========================
# Catalog loading
# =========================

def _to_display_name(raw_name: str) -> str:
    """
    Convert class_name like 'coffee_nescafe' or 'lucky-me-pancit-canton'
    into a more human-friendly 'Coffee Nescafe' or 'Lucky Me Pancit Canton'.
    """
    name = raw_name.replace("_", " ").replace("-", " ")
    return name.title()


def load_item_catalog():
    """Load items.csv into ITEM_CATALOG."""
    global ITEM_CATALOG

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "items.csv")
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        print(f"[STATE] items.csv not found at {csv_path}. Catalog will be empty.")
        ITEM_CATALOG = {}
        return

    catalog: Dict[int, ItemDefinition] = {}

    # Use utf-8-sig to remove BOM (\ufeff)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                class_id = int(row["class_id"])
                class_name = row["class_name"]
                price = float(row["price"])
                display_name = _to_display_name(class_name)

                catalog[class_id] = ItemDefinition(
                    class_id=class_id,
                    class_name=class_name,
                    display_name=display_name,
                    price=price,
                )
            except Exception as e:
                print(f"[STATE] Skipping row in items.csv due to error: {e}. Row: {row}")

    ITEM_CATALOG = catalog
    print(f"[STATE] Loaded {len(ITEM_CATALOG)} items from items.csv.")


# Load catalog at import time
load_item_catalog()


# =========================
# Session control
# =========================

def start_session():
    """Start a new session: clear items and set active = True."""
    global pos_session
    pos_session = POSSession(active=True)
    print("[STATE] Session started.")


def end_session():
    """
    End the current session: keep items, mark inactive,
    and store summary totals for UI/announcement.
    """
    global _last_session_total, _last_session_item_count

    if not pos_session.active:
        return

    _last_session_total = pos_session.total_amount
    _last_session_item_count = sum(item.quantity for item in pos_session.items)
    pos_session.active = False

    print(
        f"[STATE] Session ended. "
        f"Total: {_last_session_total:.2f}, "
        f"Items: {_last_session_item_count}"
    )


def reset_session():
    """Alias for starting a fresh session."""
    start_session()


# =========================
# POS operations
# =========================

def _find_or_create_item(item_key: str, name: str, unit_price: float) -> POSItem:
    """
    If an item with this item_key already exists in the session, increase quantity.
    Otherwise, create a new line.
    """
    for item in pos_session.items:
        if item.item_id == item_key:
            item.quantity += 1
            return item

    new_item = POSItem(
        item_id=item_key,
        name=name,
        unit_price=unit_price,
        quantity=1,
    )
    pos_session.items.append(new_item)
    return new_item


def _add_item_from_catalog(class_id: int, yolo_class_name: str):
    """
    Given a YOLO detection (class_id, class_name),
    look up the item in the catalog and add it to the session.
    """
    if not pos_session.active:
        # If session is not active, ignore detections
        return

    item_def = ITEM_CATALOG.get(class_id)
    if item_def is None:
        print(f"[STATE] Detected class_id {class_id} ('{yolo_class_name}') not in catalog.")
        return

    item_key = str(class_id)
    name = item_def.display_name
    unit_price = item_def.price

    added_item = _find_or_create_item(item_key, name, unit_price)
    print(
        f"[STATE] Added item: {added_item.name} "
        f"(qty={added_item.quantity}, price={added_item.unit_price})"
    )


# =========================
# Detection processing (called from vision.py)
# =========================

def process_detection(detected_class_id: Optional[int], detected_class_name: Optional[str]):
    """
    Process the detection result from a single frame.

    - If detected_class_id is None, we mark that the frame had no item.
    - If not None, we only add the item when transitioning from
      'no item' in the previous frame to 'some item' in this frame.
    """
    global _last_detected_item_key, _last_frame_had_item, _last_detection_time

    now = time.time()

    if detected_class_id is None:
        _last_frame_had_item = False
        _last_detected_item_key = None
        return

    item_key = str(detected_class_id)

    # Only add when we go from "no item" → "some item"
    if not _last_frame_had_item:
        if DETECTION_DEBOUNCE_SECONDS > 0:
            if now - _last_detection_time < DETECTION_DEBOUNCE_SECONDS:
                return

        _last_detection_time = now
        _last_detected_item_key = item_key
        _last_frame_had_item = True

        _add_item_from_catalog(detected_class_id, detected_class_name or "")
    else:
        # Already had an item; do nothing (same item still in frame)
        pass


# =========================
# State export for API
# =========================

def get_pos_state_dict() -> Dict:
    """
    Return a JSON-serializable dict representing the current POS state.
    Used by the FastAPI /pos_state endpoint.
    """
    if pos_session.active:
        session_status = "SESSION ACTIVE"
    else:
        if pos_session.items:
            session_status = "SESSION ENDED"
        else:
            session_status = "WAITING FOR SESSION"

    items_list = [
        {
            "item_id": item.item_id,
            "name": item.name,
            "unit_price": item.unit_price,
            "quantity": item.quantity,
            "subtotal": item.subtotal,
        }
        for item in pos_session.items
    ]

    return {
        "session_active": pos_session.active,
        "session_status": session_status,
        "items": items_list,
        "total_amount": pos_session.total_amount,
        "last_session_total": _last_session_total,
        "last_session_item_count": _last_session_item_count,
    }