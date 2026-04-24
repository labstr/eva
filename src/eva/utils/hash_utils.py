"""Utilities for computing hashes and diffs of scenario database states."""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ORDER_INDEPENDENT_LIST_FIELDS: set[str] = {
    "standby_list",
    "bookings",
    "system_accounts",
    "group_memberships",
    "asset_recoveries",
}

# Counter-generated IDs follow one of a handful of prefixes:
#   REQ-HW-048271, REQ-SW-…, REQ-FAC-…, REQ-ACCT-…
#   INC0048271
#   CASE-ACCT-048271
#   SEC-048271
#   CAL-048271
# These differ across runs based on tool-call ordering. Scoring should treat
# them as content-addressed identifiers, not counter-addressed.
_COUNTER_ID_RE = re.compile(r"^(REQ-[A-Z]+-|INC|CASE-[A-Z]+-|SEC-|CAL-)\d{5,}$")

_COUNTER_KEYS_EXCLUDED_FROM_HASH: set[str] = {
    "_request_counter",
    "_ticket_counter",
    "_case_counter",
    "_security_case_counter",
}

# Top-level tables whose dict keys are counter-generated identifiers.
_ID_KEYED_TABLES: set[str] = {
    "requests",
    "tickets",
    "cases",
    "security_cases",
    "calendar_events",
}


def _is_counter_id(value: Any) -> bool:
    return isinstance(value, str) and _COUNTER_ID_RE.match(value) is not None


def _counter_id_prefix(value: str) -> str:
    m = _COUNTER_ID_RE.match(value)
    return m.group(1) if m else ""


def _strip_counter_ids(obj: Any) -> Any:
    """Recursively replace counter-ID strings with None so they don't
    contribute to content hashes."""
    if isinstance(obj, dict):
        return {k: _strip_counter_ids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_counter_ids(item) for item in obj]
    if _is_counter_id(obj):
        return None
    return obj


def _content_hash(record: Any) -> str:
    """Compute a short content hash for a record, ignoring any counter-ID
    fields it contains. This yields identical hashes for records that differ
    only by counter-derived IDs."""
    stripped = _strip_counter_ids(record)
    serialized = json.dumps(stripped, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(serialized.encode()).hexdigest()[:12]


def canonicalize_counter_ids(obj: Any) -> Any:
    """Replace counter-generated IDs with content-deterministic identifiers
    and drop the corresponding counter keys.

    Two-pass process:
      1. Visit every identity-bearing record (keys of ID-keyed top-level
         tables plus list entries that carry a counter-ID field) and build
         a mapping from the old counter-ID to a new content-hash ID.
      2. Rewrite every dict key and every string value in the DB that
         matches a mapped old ID.

    After rewriting, remove `_*_counter` top-level keys entirely — they
    encode sequence information that the content-hash IDs already capture.

    Input is deep-copied; the argument is not mutated.
    """
    if not isinstance(obj, dict):
        return obj

    # Deep copy via JSON round-trip to isolate from caller.
    result: dict[str, Any] = json.loads(json.dumps(obj, default=str))

    id_map: dict[str, str] = {}

    # Pass 1a: identity-bearing records at well-known table paths.
    for table_name in _ID_KEYED_TABLES:
        table = result.get(table_name)
        if not isinstance(table, dict):
            continue
        for old_id, record in table.items():
            if _is_counter_id(old_id) and old_id not in id_map:
                prefix = _counter_id_prefix(old_id)
                id_map[old_id] = f"{prefix}<{_content_hash(record)}>"

    # Pass 1b: list-of-records where each element carries a counter-ID field
    # (e.g. facilities.conference_rooms.<r>.bookings[].booking_id,
    # employees.<e>.system_accounts[].case_id, asset_recoveries[].recovery_id).
    _id_fields = {
        "request_id",
        "ticket_number",
        "case_id",
        "security_case_id",
        "calendar_event_id",
        "booking_id",
        "removal_case_id",
        "pending_request_id",
        "recovery_id",
    }

    def _scan(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                _scan(v)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    for fld in _id_fields:
                        val = item.get(fld)
                        if _is_counter_id(val) and val not in id_map:
                            prefix = _counter_id_prefix(val)
                            id_map[val] = f"{prefix}<{_content_hash(item)}>"
                _scan(item)

    _scan(result)

    # Pass 2: rewrite the DB. Dict keys and string values matching an id_map
    # entry are substituted; all other structure is preserved.
    def _rewrite(node: Any) -> Any:
        if isinstance(node, dict):
            return {id_map.get(k, k): _rewrite(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_rewrite(item) for item in node]
        if isinstance(node, str) and node in id_map:
            return id_map[node]
        return node

    rewritten = _rewrite(result)

    # Drop counter keys — their values are order-artifacts, not content.
    for k in _COUNTER_KEYS_EXCLUDED_FROM_HASH:
        rewritten.pop(k, None)

    return rewritten


def hash_file(path: Path) -> str:
    """SHA-256 of file contents.

    Args:
        path: Path to the file

    Returns:
        Hexadecimal SHA-256 hash string
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_directory(dir_path: Path, glob_pattern: str = "*.json") -> str:
    """Deterministic SHA-256 of all matching files in a directory.

    Sorts files by name, hashes each, then hashes the combined (name, hash) pairs.

    Args:
        dir_path: Path to the directory
        glob_pattern: Glob pattern for files to include (default: "*.json")

    Returns:
        Hexadecimal SHA-256 hash string
    """
    dir_path = Path(dir_path)
    file_hashes: list[tuple[str, str]] = []
    for file_path in sorted(dir_path.rglob(glob_pattern)):
        if file_path.is_file():
            rel_name = str(file_path.relative_to(dir_path))
            file_hashes.append((rel_name, hash_file(file_path)))

    combined = json.dumps(file_hashes, separators=(",", ":"))
    return hashlib.sha256(combined.encode()).hexdigest()


def normalize_for_comparison(obj: Any) -> Any:
    """Recursively normalize values for consistent comparison and hashing.

    Normalizations applied:
    - float → int when the value is a whole number and finite (e.g., 1.0 → 1)
    - str → None when the value is "none" or "null" (case-insensitive, stripped)
    - Recurse into dict values and list elements

    Args:
        obj: Any JSON-compatible value (dict, list, str, int, float, bool, None)

    Returns:
        Normalized copy of the input
    """
    if isinstance(obj, dict):
        normalized = {}
        for k, v in obj.items():
            norm_v = normalize_for_comparison(v)
            if k in ORDER_INDEPENDENT_LIST_FIELDS and isinstance(norm_v, list):
                norm_v = sorted(norm_v, key=lambda x: json.dumps(x, sort_keys=True, default=str))
            normalized[k] = norm_v
        return normalized
    if isinstance(obj, list):
        return [normalize_for_comparison(item) for item in obj]
    if isinstance(obj, float):
        if math.isfinite(obj) and obj.is_integer():
            return int(obj)
        return obj
    if isinstance(obj, str):
        if obj.strip().lower() in ("none", "null"):
            return None
        return obj
    return obj


def get_dict_hash(obj: dict) -> str:
    """Compute SHA-256 hash of a dictionary.

    Follows tau-2 bench's approach:
    - Serialize with sort_keys=True for deterministic ordering
    - Use default=str for non-JSON-serializable types
    - Compute SHA-256 hash of the serialized string

    The 'session' key is always excluded from hashing — auth success is
    tracked separately via the authentication_success metric.

    Args:
        obj: Dictionary to hash

    Returns:
        Hexadecimal SHA-256 hash string
    """
    obj_for_hash = {k: v for k, v in obj.items() if k != "session"} if isinstance(obj, dict) else obj
    obj_for_hash = canonicalize_counter_ids(obj_for_hash)
    normalized = normalize_for_comparison(obj_for_hash)
    serialized = json.dumps(normalized, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_db_diff(expected_db: dict, actual_db: dict) -> dict:
    """Compute structured diff between expected and actual database states.

    Args:
        expected_db: Expected final database state
        actual_db: Actual final database state

    Returns:
        Dictionary containing:
        - tables_added: Tables in actual but not expected
        - tables_removed: Tables in expected but not actual
        - tables_modified: Tables present in both with differences
        - For each modified table:
          - records_added: Records in actual but not expected
          - records_removed: Records in expected but not actual
          - records_modified: Records with field differences
          - For each modified record: field-level changes
    """
    expected_db = canonicalize_counter_ids(expected_db)
    actual_db = canonicalize_counter_ids(actual_db)

    diff: dict[str, Any] = {"tables_added": [], "tables_removed": [], "tables_modified": {}}

    expected_tables = set(expected_db.keys())
    actual_tables = set(actual_db.keys())

    # Tables only in actual (unexpected additions)
    diff["tables_added"] = sorted(actual_tables - expected_tables)

    # Tables only in expected (missing from actual)
    diff["tables_removed"] = sorted(expected_tables - actual_tables)

    # Tables in both - check for modifications
    common_tables = expected_tables & actual_tables
    for table in sorted(common_tables):
        expected_table = expected_db[table]
        actual_table = actual_db[table]

        # Handle non-dict tables (e.g., error_injections which is a list)
        if not isinstance(expected_table, dict) or not isinstance(actual_table, dict):
            if expected_table != actual_table:
                diff["tables_modified"][table] = {
                    "type": "non_dict_table",
                    "expected": expected_table,
                    "actual": actual_table,
                }
            continue

        # Compare records within the table
        table_diff = _compute_table_diff(expected_table, actual_table)
        if table_diff:  # Only include if there are differences
            diff["tables_modified"][table] = table_diff

    return diff


def _compute_table_diff(expected_table: dict, actual_table: dict) -> dict[str, Any] | None:
    """Compute diff between two table dictionaries.

    Args:
        expected_table: Expected table records
        actual_table: Actual table records

    Returns:
        Diff dictionary or None if tables are identical
    """
    table_diff: dict[str, Any] = {"records_added": [], "records_removed": [], "records_modified": {}}

    expected_keys = set(expected_table.keys())
    actual_keys = set(actual_table.keys())

    # Records only in actual
    table_diff["records_added"] = sorted(actual_keys - expected_keys)

    # Records only in expected
    table_diff["records_removed"] = sorted(expected_keys - actual_keys)

    # Records in both - check for field differences
    common_keys = expected_keys & actual_keys
    for key in sorted(common_keys):
        expected_record = expected_table[key]
        actual_record = actual_table[key]

        record_diff = _compute_record_diff(expected_record, actual_record)
        if record_diff:  # Only include if there are differences
            table_diff["records_modified"][str(key)] = record_diff

    # Return None if no differences found
    if not table_diff["records_added"] and not table_diff["records_removed"] and not table_diff["records_modified"]:
        return None

    return table_diff


def _compute_record_diff(
    expected_record: Any, actual_record: Any, path: str = "", field_name: str = ""
) -> dict[str, Any] | None:
    """Recursively compute diff between two records.

    Args:
        expected_record: Expected record value
        actual_record: Actual record value
        path: Current path in the nested structure (for debugging)
        field_name: Name of the field being compared (for order-independent handling)

    Returns:
        Diff dictionary or None if records are identical
    """
    # Normalize both sides to avoid false mismatches (e.g., 1 vs 1.0)
    expected_record = normalize_for_comparison(expected_record)
    actual_record = normalize_for_comparison(actual_record)

    # If values are identical, no diff
    if expected_record == actual_record:
        return None

    # If types differ, report the difference
    if type(expected_record) is not type(actual_record):
        return {
            "type": "type_mismatch",
            "expected": expected_record,
            "actual": actual_record,
            "expected_type": type(expected_record).__name__,
            "actual_type": type(actual_record).__name__,
        }

    # Handle dictionaries recursively
    if isinstance(expected_record, dict):
        field_diff: dict[str, Any] = {"fields_added": [], "fields_removed": [], "fields_modified": {}}

        expected_keys = set(expected_record.keys())
        actual_keys = set(actual_record.keys())

        field_diff["fields_added"] = sorted(actual_keys - expected_keys)
        field_diff["fields_removed"] = sorted(expected_keys - actual_keys)

        # Compare common fields
        common_keys = expected_keys & actual_keys
        for key in sorted(common_keys):
            nested_path = f"{path}.{key}" if path else key
            nested_diff = _compute_record_diff(expected_record[key], actual_record[key], nested_path, field_name=key)
            if nested_diff:
                field_diff["fields_modified"][key] = nested_diff

        # Return None if no differences
        if not field_diff["fields_added"] and not field_diff["fields_removed"] and not field_diff["fields_modified"]:
            return None

        return field_diff

    # Handle lists
    if isinstance(expected_record, list):
        # Sort order-independent lists before comparison
        if field_name in ORDER_INDEPENDENT_LIST_FIELDS:

            def sort_key(x):
                return json.dumps(x, sort_keys=True, default=str)

            expected_record = sorted(expected_record, key=sort_key)
            actual_record = sorted(actual_record, key=sort_key)

        if len(expected_record) != len(actual_record):
            return {
                "type": "list_length_mismatch",
                "expected": expected_record,
                "actual": actual_record,
                "expected_length": len(expected_record),
                "actual_length": len(actual_record),
            }

        # Compare elements
        list_diffs = []
        for i, (exp_item, act_item) in enumerate(zip(expected_record, actual_record)):
            item_diff = _compute_record_diff(exp_item, act_item, f"{path}[{i}]")
            if item_diff:
                list_diffs.append({"index": i, "diff": item_diff})

        if not list_diffs:
            return None

        return {"type": "list_differences", "differences": list_diffs}

    # Primitive type difference
    return {"type": "value_mismatch", "expected": expected_record, "actual": actual_record}
