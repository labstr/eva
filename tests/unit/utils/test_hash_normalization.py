"""Tests for hash normalization in eva.utils.hash_utils."""

from eva.utils.hash_utils import compute_db_diff, get_dict_hash, normalize_for_comparison

DB_A = {
    "reservations": {
        "ABC123": {
            "confirmation_number": "ABC123",
            "last_name": "Smith",
            "fare_paid": 350,
            "meal_preference": None,
            "passengers": [
                {"name": "John Smith", "seat": "12A", "bags": 2},
            ],
        },
    },
    "flights": {
        "UA100": {
            "flight_number": "UA100",
            "gate": None,
            "capacity": 180,
            "available_seats": 12,
        },
    },
}

# Variant form: floats for whole numbers, "none"/"null" strings for None
DB_B = {
    "reservations": {
        "ABC123": {
            "confirmation_number": "ABC123",
            "last_name": "Smith",
            "fare_paid": 350.0,
            "meal_preference": "none",
            "passengers": [
                {"name": "John Smith", "seat": "12A", "bags": 2.0},
            ],
        },
    },
    "flights": {
        "UA100": {
            "flight_number": "UA100",
            "gate": "Null",
            "capacity": 180.0,
            "available_seats": 12.0,
        },
    },
}


# Genuinely different database
DB_C = {
    "reservations": {
        "ABC123": {
            "confirmation_number": "ABC123",
            "last_name": "Smith",
            "fare_paid": 400,
            "meal_preference": None,
            "passengers": [
                {"name": "John Smith", "seat": "14B", "bags": 2},
            ],
        },
    },
    "flights": {
        "UA100": {
            "flight_number": "UA100",
            "gate": None,
            "capacity": 180,
            "available_seats": 10,
        },
    },
}

# Variant form: floats for whole numbers, lowercase "none"/"null" strings for None
DB_D = {
    "reservations": {
        "ABC123": {
            "confirmation_number": "ABC123",
            "last_name": "Smith",
            "fare_paid": 350.0,
            "meal_preference": "null",
            "passengers": [
                {"name": "John Smith", "seat": "12A", "bags": 2.0},
            ],
        },
    },
    "flights": {
        "UA100": {
            "flight_number": "UA100",
            "gate": "none",
            "capacity": 180.0,
            "available_seats": 12.0,
        },
    },
}


class TestNormalizeForComparison:
    def test_normalize_float_to_int(self):
        assert normalize_for_comparison(1.0) == 1
        assert normalize_for_comparison(0.0) == 0
        assert normalize_for_comparison(-3.0) == -3
        # Non-whole floats stay as floats
        assert normalize_for_comparison(1.5) == 1.5
        assert normalize_for_comparison(0.1) == 0.1

    def test_normalize_none_strings(self):
        assert normalize_for_comparison("none") is None
        assert normalize_for_comparison("None") is None
        assert normalize_for_comparison("null") is None
        assert normalize_for_comparison("NULL") is None
        assert normalize_for_comparison(" none ") is None
        assert normalize_for_comparison("  Null  ") is None

    def test_normalize_preserves_normal_strings(self):
        assert normalize_for_comparison("hello") == "hello"
        assert normalize_for_comparison("nonevent") == "nonevent"
        assert normalize_for_comparison("nullable") == "nullable"
        assert normalize_for_comparison("announcer") == "announcer"
        assert normalize_for_comparison("") == ""

    def test_normalize_nested_structures(self):
        nested = {
            "a": 1.0,
            "b": "none",
            "c": [2.0, "NULL", "keep"],
            "d": {"x": 3.0, "y": "null"},
        }
        expected = {
            "a": 1,
            "b": None,
            "c": [2, None, "keep"],
            "d": {"x": 3, "y": None},
        }
        assert normalize_for_comparison(nested) == expected

    def test_normalize_preserves_bools_and_none(self):
        assert normalize_for_comparison(True) is True
        assert normalize_for_comparison(False) is False
        assert normalize_for_comparison(None) is None

    def test_normalize_preserves_ints(self):
        assert normalize_for_comparison(42) == 42
        assert normalize_for_comparison(0) == 0


class TestGetDictHash:
    def test_hash_identical_for_normalized_equivalent_dbs(self):
        assert get_dict_hash(DB_A) == get_dict_hash(DB_B) == get_dict_hash(DB_D)

    def test_hash_differs_for_genuinely_different_dbs(self):
        assert get_dict_hash(DB_A) != get_dict_hash(DB_C)

    def test_hash_is_deterministic(self):
        h1 = get_dict_hash(DB_A)
        h2 = get_dict_hash(DB_A)
        assert h1 == h2

    def test_hash_differs_for_key_order_irrelevant(self):
        """Dict key order shouldn't matter (sort_keys=True)."""
        d1 = {"b": 1, "a": 2}
        d2 = {"a": 2, "b": 1}
        assert get_dict_hash(d1) == get_dict_hash(d2)

    def test_session_key_excluded_from_hash(self):
        """Session key should not affect the hash."""
        db_without_session = {"reservations": {"ABC": {"status": "confirmed"}}}
        db_with_session = {**db_without_session, "session": {"confirmation_number": "ABC", "last_name": "doe"}}
        assert get_dict_hash(db_without_session) == get_dict_hash(db_with_session)

    def test_different_sessions_produce_same_hash(self):
        """Two DBs identical except for session content should hash the same."""
        db_session_a = {"reservations": {}, "session": {"confirmation_number": "AAA", "last_name": "smith"}}
        db_session_b = {"reservations": {}, "session": {"confirmation_number": "BBB", "last_name": "jones"}}
        assert get_dict_hash(db_session_a) == get_dict_hash(db_session_b)


class TestComputeDbDiff:
    def test_no_diff_after_normalization(self):
        diff = compute_db_diff(DB_A, DB_B)
        assert diff["tables_added"] == []
        assert diff["tables_removed"] == []
        assert diff["tables_modified"] == {}

    def test_diff_for_genuinely_different_dbs(self):
        diff = compute_db_diff(DB_A, DB_C)
        assert diff["tables_modified"] != {}


DB_STANDBY_ORDER_1 = {
    "reservations": {
        "ABC123": {
            "confirmation_number": "ABC123",
            "standby_list": [
                {"flight": "UA100", "passenger": "Smith"},
                {"flight": "UA200", "passenger": "Smith"},
            ],
        },
    },
    "flights": {
        "UA100": {
            "flight_number": "UA100",
            "standby_list": [
                {"name": "Smith", "priority": 1},
                {"name": "Jones", "priority": 2},
            ],
        },
    },
}

DB_STANDBY_ORDER_2 = {
    "reservations": {
        "ABC123": {
            "confirmation_number": "ABC123",
            "standby_list": [
                {"flight": "UA200", "passenger": "Smith"},
                {"flight": "UA100", "passenger": "Smith"},
            ],
        },
    },
    "flights": {
        "UA100": {
            "flight_number": "UA100",
            "standby_list": [
                {"name": "Jones", "priority": 2},
                {"name": "Smith", "priority": 1},
            ],
        },
    },
}

DB_DIFFERENT_SEGMENTS = {
    "reservations": {
        "ABC123": {
            "confirmation_number": "ABC123",
            "segments": [
                {"flight": "UA100", "class": "economy"},
                {"flight": "UA200", "class": "business"},
            ],
        },
    },
}

DB_DIFFERENT_SEGMENTS_REVERSED = {
    "reservations": {
        "ABC123": {
            "confirmation_number": "ABC123",
            "segments": [
                {"flight": "UA200", "class": "business"},
                {"flight": "UA100", "class": "economy"},
            ],
        },
    },
}


class TestStandbyListOrderIndependence:
    def test_hash_matches_with_different_standby_order(self):
        """standby_list order should not affect hash."""
        assert get_dict_hash(DB_STANDBY_ORDER_1) == get_dict_hash(DB_STANDBY_ORDER_2)

    def test_diff_empty_with_different_standby_order(self):
        """standby_list order should not produce a diff."""
        diff = compute_db_diff(DB_STANDBY_ORDER_1, DB_STANDBY_ORDER_2)
        assert diff["tables_added"] == []
        assert diff["tables_removed"] == []
        assert diff["tables_modified"] == {}

    def test_hash_differs_for_different_segment_order(self):
        """Segments (not in ORDER_INDEPENDENT_LIST_FIELDS) should still be order-sensitive."""
        assert get_dict_hash(DB_DIFFERENT_SEGMENTS) != get_dict_hash(DB_DIFFERENT_SEGMENTS_REVERSED)

    def test_diff_reports_different_segment_order(self):
        """Segments reordering should produce a diff."""
        diff = compute_db_diff(DB_DIFFERENT_SEGMENTS, DB_DIFFERENT_SEGMENTS_REVERSED)
        assert diff["tables_modified"] != {}


# ---------------------------------------------------------------------------
# Counter-generated ID canonicalization
# ---------------------------------------------------------------------------


class TestCounterIdCanonicalization:
    """Counter-generated IDs (REQ-*, INC*, CASE-*, SEC-*, CAL-*) should be
    replaced with content-deterministic identifiers at comparison time, so
    scoring ignores tool-call order and counter values."""

    def _itsm_db(self, req_hw_id: str, req_fac_id: str, cal_id: str, counter: int) -> dict:
        return {
            "_request_counter": counter,
            "requests": {
                req_hw_id: {
                    "request_id": req_hw_id,
                    "employee_id": "EMP001",
                    "request_type": "laptop_replacement",
                    "laptop_size": "14_inch",
                },
                req_fac_id: {
                    "request_id": req_fac_id,
                    "employee_id": "EMP001",
                    "equipment_type": "ergonomic_chair",
                },
            },
            "calendar_events": {
                cal_id: {
                    "calendar_event_id": cal_id,
                    "request_id": req_fac_id,
                    "date": "2026-07-06",
                },
            },
        }

    def test_hash_matches_when_counter_order_differs(self):
        """Same content, different tool-call order → same hash."""
        expected = self._itsm_db("REQ-HW-048271", "REQ-FAC-048272", "CAL-048272", 48272)
        actual = self._itsm_db("REQ-HW-048272", "REQ-FAC-048271", "CAL-048271", 48272)
        assert get_dict_hash(expected) == get_dict_hash(actual)

    def test_hash_differs_when_content_differs(self):
        """Different content → different hash, even after canonicalization."""
        expected = self._itsm_db("REQ-HW-048271", "REQ-FAC-048272", "CAL-048272", 48272)
        actual = self._itsm_db("REQ-HW-048271", "REQ-FAC-048272", "CAL-048272", 48272)
        actual["requests"]["REQ-HW-048271"]["laptop_size"] = "16_inch"
        assert get_dict_hash(expected) != get_dict_hash(actual)

    def test_hash_differs_when_record_missing(self):
        """A missing request → different hash (not collapsed to same content)."""
        expected = self._itsm_db("REQ-HW-048271", "REQ-FAC-048272", "CAL-048272", 48272)
        actual = self._itsm_db("REQ-HW-048271", "REQ-FAC-048272", "CAL-048272", 48272)
        del actual["requests"]["REQ-FAC-048272"]
        assert get_dict_hash(expected) != get_dict_hash(actual)

    def test_diff_clean_when_only_counters_differ(self):
        """compute_db_diff should be empty when only counter/ID values differ."""
        expected = self._itsm_db("REQ-HW-048271", "REQ-FAC-048272", "CAL-048272", 48272)
        actual = self._itsm_db("REQ-HW-048275", "REQ-FAC-048276", "CAL-048276", 48276)
        diff = compute_db_diff(expected, actual)
        assert diff["tables_modified"] == {}
        assert diff["tables_added"] == []
        assert diff["tables_removed"] == []

    def test_cross_reference_preserved(self):
        """calendar_events.*.request_id should still tie to the right request
        after canonicalization — mis-wired references should be caught."""
        expected = self._itsm_db("REQ-HW-048271", "REQ-FAC-048272", "CAL-048272", 48272)
        # Point calendar at the HW request instead of the chair request.
        actual = self._itsm_db("REQ-HW-048271", "REQ-FAC-048272", "CAL-048272", 48272)
        actual["calendar_events"]["CAL-048272"]["request_id"] = "REQ-HW-048271"
        assert get_dict_hash(expected) != get_dict_hash(actual)

    def test_bookings_list_order_independent(self):
        """Bookings list ordering should not affect the hash."""
        db_a = {
            "facilities": {
                "conference_rooms": {
                    "RM-1": {
                        "bookings": [
                            {"booking_id": "REQ-FAC-048271", "date": "2026-07-06", "start_time": "10:00"},
                            {"booking_id": "REQ-FAC-048273", "date": "2026-07-07", "start_time": "14:00"},
                        ]
                    }
                }
            }
        }
        db_b = {
            "facilities": {
                "conference_rooms": {
                    "RM-1": {
                        "bookings": [
                            {"booking_id": "REQ-FAC-999999", "date": "2026-07-07", "start_time": "14:00"},
                            {"booking_id": "REQ-FAC-000001", "date": "2026-07-06", "start_time": "10:00"},
                        ]
                    }
                }
            }
        }
        assert get_dict_hash(db_a) == get_dict_hash(db_b)

    def test_stable_ids_untouched(self):
        """GRP-*, APP-*, EMP-* etc. should NOT be canonicalized — only
        counter-generated prefixes (REQ/INC/CASE/SEC/CAL)."""
        db_a = {"access_groups": {"GRP-ENGCORE": {"name": "Eng Core"}}}
        db_b = {"access_groups": {"GRP-DIFFERENT": {"name": "Eng Core"}}}
        assert get_dict_hash(db_a) != get_dict_hash(db_b)

    def test_session_still_excluded(self):
        """session subtree must remain excluded from the hash (pre-existing behavior)."""
        a = {"session": {"otp_auth": True}, "requests": {}}
        b = {"session": {"otp_auth": False}, "requests": {}}
        assert get_dict_hash(a) == get_dict_hash(b)
