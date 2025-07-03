import pytest
from robodsl.utils.typecheck_bridge import check_ros2_type

def test_typecheck_vector_float():
    ok, err, _ = check_ros2_type("std::vector<float>", [1.0, 2.0, 3.0])
    assert ok
    ok, err, _ = check_ros2_type("std::vector<float>", [1.0, "bad", 3.0])
    assert not ok and "float" in err

def test_typecheck_string():
    ok, err, _ = check_ros2_type("std::string", "hello")
    assert ok
    ok, err, _ = check_ros2_type("std::string", 123)
    assert not ok and "string" in err

def test_typecheck_float():
    ok, err, _ = check_ros2_type("float", 1.23)
    assert ok
    ok, err, _ = check_ros2_type("float", "not a float")
    assert not ok and "number" in err

def test_typecheck_int():
    ok, err, _ = check_ros2_type("int", 42)
    assert ok
    ok, err, _ = check_ros2_type("int", 3.14)
    assert not ok and "integer" in err

def test_typecheck_bool():
    assert check_ros2_type("bool", True)[0]
    assert check_ros2_type("bool", False)[0]
    assert not check_ros2_type("bool", 1)[0]

def test_typecheck_unsupported():
    ok, err, _ = check_ros2_type("unsupported_type", 123)
    assert not ok and "not supported" in err

def test_typecheck_int_types():
    for t in ["int8_t", "int16_t", "int32_t", "int64_t"]:
        ok, err, debug = check_ros2_type(t, 42)
        if not ok:
            print(f"[TYPECHECK DEBUG] {debug}")
        assert ok
        ok, err, debug = check_ros2_type(t, 3.14)
        if ok:
            print(f"[TYPECHECK DEBUG] {debug}")
        assert not ok

def test_typecheck_uint_types():
    for t in ["uint8_t", "uint16_t", "uint32_t", "uint64_t", "size_t"]:
        ok, err, debug = check_ros2_type(t, 42)
        if not ok:
            print(f"[TYPECHECK DEBUG] {debug}")
        assert ok
        ok, err, debug = check_ros2_type(t, -1)
        if ok:
            print(f"[TYPECHECK DEBUG] {debug}")
        assert not ok

def test_typecheck_float_types():
    for t in ["float32", "float64"]:
        ok, err, debug = check_ros2_type(t, 1.23)
        if not ok:
            print(f"[TYPECHECK DEBUG] {debug}")
        assert ok
        ok, err, debug = check_ros2_type(t, "not a float")
        if ok:
            print(f"[TYPECHECK DEBUG] {debug}")
        assert not ok

def test_typecheck_char():
    ok, err, debug = check_ros2_type("char", "a")
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("char", "abc")
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_array():
    ok, err, debug = check_ros2_type("std::array<int, 3>", [1, 2, 3])
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("std::array<int, 3>", [1, 2])
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("std::array<float, 2>", [1.0, "bad"])
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_map():
    ok, err, debug = check_ros2_type("std::map<string, int>", {"a": 1, "b": 2})
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("std::map<string, int>", {"a": "bad"})
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("std::map<int, float>", {"1": 1.0, "2": 2.0})
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("std::map<int, float>", {"a": 1.0})
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok  # JSON keys are always strings, so int keys are accepted as strings

def test_typecheck_unordered_map():
    ok, err, debug = check_ros2_type("std::unordered_map<string, float>", {"x": 1.0, "y": 2.0})
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("std::unordered_map<string, float>", {"x": "bad"})
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_custom_msg():
    ok, err, debug = check_ros2_type("std_msgs/msg/String", {"data": "hello"})
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("std_msgs/msg/String", 123)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_char_types():
    ok, err, debug = check_ros2_type("char", "A")
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("char", 65)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("char", "AB")
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("char", 128)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned char", 255)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("unsigned char", -1)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned char", 256)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_short_types():
    ok, err, debug = check_ros2_type("short", -32768)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("short", 32767)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("short", -32769)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("short", 32768)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned short", 0)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("unsigned short", 65535)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("unsigned short", -1)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned short", 65536)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_int_types():
    ok, err, debug = check_ros2_type("int", 0)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("int32_t", 2147483647)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("int", 2147483648)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("int32_t", -2147483649)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned int", 4294967295)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("unsigned int", -1)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned int", 4294967296)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_long_types():
    ok, err, debug = check_ros2_type("long", 9223372036854775807)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("long", 9223372036854775808)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned long", 18446744073709551615)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("unsigned long", -1)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned long", 18446744073709551616)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_long_long_types():
    ok, err, debug = check_ros2_type("long long", 9223372036854775807)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("long long", 9223372036854775808)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned long long", 18446744073709551615)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("unsigned long long", -1)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("unsigned long long", 18446744073709551616)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("int64_t", -9223372036854775808)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("uint64_t", 18446744073709551615)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("int64_t", -9223372036854775809)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("uint64_t", 18446744073709551616)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("size_t", 18446744073709551615)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("size_t", -1)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_int8_16_types():
    ok, err, debug = check_ros2_type("int8_t", -128)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("int8_t", 127)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("int8_t", -129)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("int8_t", 128)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("uint8_t", 0)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("uint8_t", 255)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("uint8_t", -1)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("uint8_t", 256)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("int16_t", -32768)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("int16_t", 32767)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("int16_t", -32769)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("int16_t", 32768)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("uint16_t", 0)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("uint16_t", 65535)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("uint16_t", -1)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("uint16_t", 65536)
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok

def test_typecheck_float_types():
    ok, err, debug = check_ros2_type("float", 1.23)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("float32", 1.23)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("double", 1.23)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("float64", 1.23)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("long double", 1.23)
    if not ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert ok
    ok, err, debug = check_ros2_type("float", "not a float")
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("double", "not a double")
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok
    ok, err, debug = check_ros2_type("long double", "not a long double")
    if ok:
        print(f"[TYPECHECK DEBUG] {debug}")
    assert not ok 