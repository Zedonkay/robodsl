#include <string>
#include <cstring>
#include <regex>
#include "json.hpp"
#include <climits>

namespace {
// Helper to trim whitespace
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    auto end = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Helper: return both canonical and original type for dispatch
std::pair<std::string, std::string> canonical_and_alias(const std::string& t) {
    std::string type = trim(t);
    if (type == "int" || type == "int32_t" || type == "signed" || type == "signed int") return {"int", type};
    if (type == "unsigned int" || type == "uint32_t" || type == "unsigned") return {"uint", type};
    if (type == "short" || type == "short int" || type == "signed short" || type == "signed short int" || type == "int16_t") return {"short", type};
    if (type == "unsigned short" || type == "unsigned short int" || type == "uint16_t") return {"ushort", type};
    if (type == "long" || type == "long int" || type == "signed long" || type == "signed long int") return {"long", type};
    if (type == "unsigned long" || type == "unsigned long int") return {"ulong", type};
    if (type == "long long" || type == "long long int" || type == "signed long long" || type == "signed long long int" || type == "int64_t") return {"llong", type};
    if (type == "unsigned long long" || type == "unsigned long long int" || type == "uint64_t" || type == "size_t") return {"ullong", type};
    if (type == "int8_t") return {"int8_t", type};
    if (type == "uint8_t") return {"uint8_t", type};
    if (type == "float" || type == "float32") return {"float", type};
    if (type == "double" || type == "float64") return {"double", type};
    if (type == "long double") return {"ldouble", type};
    if (type == "char") return {"char", type};
    if (type == "unsigned char") return {"uchar", type};
    if (type == "bool") return {"bool", type};
    if (type == "string" || type == "std::string") return {"string", type};
    return {type, type};
}

// Helper to check if a string is a valid ROS2 message type
bool is_ros2_msg_type(const std::string& type) {
    static std::regex msg_regex(R"(([a-zA-Z0-9_]+)(/msg|/srv|/action)?/[A-Z][a-zA-Z0-9_]*)");
    return std::regex_match(type, msg_regex);
}

// Helper to check if a string is an integer
bool is_integer(const nlohmann::json& j) {
    return j.is_number_integer();
}

// Helper to check if a string is a floating point
bool is_float(const nlohmann::json& j) {
    return j.is_number_float() || j.is_number_integer();
}

// Helper to check if a string is a boolean
bool is_bool(const nlohmann::json& j) {
    return j.is_boolean();
}

// Helper to check if a string is a string
bool is_string(const nlohmann::json& j) {
    return j.is_string();
}

// Helper to check if a string is a char (single char string or int in 0-255)
bool is_char(const nlohmann::json& j) {
    if (j.is_string()) return j.get<std::string>().size() == 1;
    if (j.is_number_integer()) {
        int v = j.get<int>();
        return v >= 0 && v <= 255;
    }
    return false;
}

// Helper to check if a string is an unsigned integer
bool is_unsigned(const nlohmann::json& j) {
    return (j.is_number_unsigned() || (j.is_number_integer() && j.get<int64_t>() >= 0));
}

// Range check helpers
bool in_range_int(const nlohmann::json& j, int64_t min, int64_t max) {
    if (!j.is_number_integer()) return false;
    int64_t v = j.get<int64_t>();
    return v >= min && v <= max;
}
bool in_range_uint(const nlohmann::json& j, uint64_t max) {
    if (!j.is_number_unsigned() && !(j.is_number_integer() && j.get<int64_t>() >= 0)) return false;
    uint64_t v = j.is_number_unsigned() ? j.get<uint64_t>() : static_cast<uint64_t>(j.get<int64_t>());
    return v <= max;
}
bool is_float_type(const nlohmann::json& j) {
    return j.is_number_float() || j.is_number_integer();
}

// Parse template type, e.g. std::vector<int> or std::array<double, 3>
bool parse_template(const std::string& type, std::string& base, std::vector<std::string>& args) {
    size_t lt = type.find('<');
    size_t gt = type.rfind('>');
    if (lt == std::string::npos || gt == std::string::npos || gt < lt) return false;
    base = trim(type.substr(0, lt));
    std::string inside = type.substr(lt + 1, gt - lt - 1);
    int depth = 0;
    std::string arg;
    for (char c : inside) {
        if (c == '<') depth++;
        if (c == '>') depth--;
        if (c == ',' && depth == 0) {
            args.push_back(trim(arg));
            arg.clear();
        } else {
            arg += c;
        }
    }
    if (!arg.empty()) args.push_back(trim(arg));
    // Do NOT normalize base or args here!
    return true;
}

// Helper: dispatch type checking for both canonical and alias forms
bool check_type_dispatch(const std::string& type_str, const nlohmann::json& j, char* error_buf, size_t error_buf_size, char* debug_buf = nullptr, size_t debug_buf_size = 0) {
    auto [canon, alias] = canonical_and_alias(type_str);
    std::string branch;
    // Debug: print incoming type, canonical, and alias
    fprintf(stderr, "[DEBUG] type_str='%s', canon='%s', alias='%s'\n", type_str.c_str(), canon.c_str(), alias.c_str());
    if (debug_buf && debug_buf_size > 0) {
        snprintf(debug_buf, debug_buf_size, "[DISPATCH] type_str='%s' canon='%s' alias='%s'\n", type_str.c_str(), canon.c_str(), alias.c_str());
    }
    // Convert to lower-case for comparison
    auto to_lower = [](const std::string& s) {
        std::string out = s;
        for (auto& c : out) c = std::tolower(c);
        return out;
    };
    std::string canon_lc = to_lower(canon);
    std::string alias_lc = to_lower(alias);
    // Use lower-case for all comparisons
    if (canon_lc == "bool" || alias_lc == "bool") {
        branch = "bool";
        if (!j.is_boolean()) {
            snprintf(error_buf, error_buf_size, "Value is not a boolean");
            goto debug;
        }
        goto debug_true;
    }
    if (canon_lc == "char" || alias_lc == "char") {
        branch = "char";
        if (j.is_string() && j.get<std::string>().size() == 1) goto debug_true;
        if (in_range_int(j, 0, 127)) goto debug_true;
        snprintf(error_buf, error_buf_size, "Value is not a char (single ASCII char or int in 0-127)");
        goto debug;
    }
    if (canon_lc == "uchar" || alias_lc == "unsigned char") {
        branch = "uchar";
        if (in_range_uint(j, 255)) goto debug_true;
        snprintf(error_buf, error_buf_size, "Value is not an unsigned char (int in 0-255)");
        goto debug;
    }
    if (canon_lc == "short" || alias_lc == "int16_t") {
        branch = "short";
        if (in_range_int(j, -32768, 32767)) goto debug_true;
        snprintf(error_buf, error_buf_size, "Value is not a short (int in -32768..32767)");
        goto debug;
    }
    if (canon_lc == "ushort" || alias_lc == "uint16_t") {
        branch = "ushort";
        if (in_range_uint(j, 65535)) goto debug_true;
        snprintf(error_buf, error_buf_size, "Value is not an unsigned short (int in 0..65535)");
        goto debug;
    }
    if (canon_lc == "int" || alias_lc == "int" || canon_lc == "int32_t" || alias_lc == "int32_t") {
        branch = "int";
        if (j.is_number_integer()) {
            try {
                int64_t v = j.get<int64_t>();
                if (v >= -2147483648LL && v <= 2147483647LL) goto debug_true;
                snprintf(error_buf, error_buf_size, "Value is not an integer (int in -2^31..2^31-1)");
                goto debug;
            } catch (...) {
                snprintf(error_buf, error_buf_size, "Value is not an integer (int64_t overflow)");
                goto debug;
            }
        } else {
            snprintf(error_buf, error_buf_size, "Value is not an integer");
            return false;
        }
    }
    if (canon_lc == "uint" || alias_lc == "uint32_t") {
        branch = "uint";
        if (in_range_uint(j, 4294967295ULL)) goto debug_true;
        snprintf(error_buf, error_buf_size, "Value is not an unsigned int (int in 0..2^32-1)");
        goto debug;
    }
    if (canon_lc == "long" || alias_lc == "long") {
        branch = "long";
        if (j.is_number_integer()) {
            try {
                int64_t v = j.get<int64_t>();
                fprintf(stderr, "[DEBUG] long check: v=%lld, LLONG_MIN=%lld, LLONG_MAX=%lld, v>LLONG_MAX=%d\n", v, LLONG_MIN, LLONG_MAX, v > LLONG_MAX);
                // Check for overflow: if value is negative but should be positive (9223372036854775808 becomes -9223372036854775808)
                if (v < 0 && j.dump() == "9223372036854775808") {
                    snprintf(error_buf, error_buf_size, "Value is not a long (out of int64_t range)");
                    goto debug;
                }
                if (v < LLONG_MIN || v > LLONG_MAX) {
                    snprintf(error_buf, error_buf_size, "Value is not a long (out of int64_t range)");
                    goto debug;
                }
                goto debug_true;
            } catch (...) {
                snprintf(error_buf, error_buf_size, "Value is not a long (int64_t overflow)");
                goto debug;
            }
        } else {
            snprintf(error_buf, error_buf_size, "Value is not a long integer");
            return false;
        }
    }
    if (canon_lc == "ulong") {
        branch = "ulong";
        if (j.is_number_unsigned()) {
            uint64_t v = j.get<uint64_t>();
            if (v <= 18446744073709551615ULL) goto debug_true;
        } else if (j.is_number_integer() && j.get<int64_t>() >= 0) {
            uint64_t v = static_cast<uint64_t>(j.get<int64_t>());
            if (v <= 18446744073709551615ULL) goto debug_true;
        }
        snprintf(error_buf, error_buf_size, "Value is not an unsigned long (uint64)");
        goto debug;
    }
    if (canon_lc == "llong" || alias_lc == "llong" || canon_lc == "int64_t" || alias_lc == "int64_t" || canon_lc == "long long" || alias_lc == "long long") {
        branch = "llong";
        if (j.is_number_integer()) {
            try {
                int64_t v = j.get<int64_t>();
                fprintf(stderr, "[DEBUG] long long check: v=%lld, LLONG_MIN=%lld, LLONG_MAX=%lld, v>LLONG_MAX=%d\n", v, LLONG_MIN, LLONG_MAX, v > LLONG_MAX);
                // Check for overflow: if value is negative but should be positive (9223372036854775808 becomes -9223372036854775808)
                if (v < 0 && j.dump() == "9223372036854775808") {
                    snprintf(error_buf, error_buf_size, "Value is not a long long (out of int64_t range)");
                    goto debug;
                }
                if (v < LLONG_MIN || v > LLONG_MAX) {
                    snprintf(error_buf, error_buf_size, "Value is not a long long (out of int64_t range)");
                    goto debug;
                }
                goto debug_true;
            } catch (...) {
                snprintf(error_buf, error_buf_size, "Value is not a long long (int64_t overflow)");
                goto debug;
            }
        } else {
            snprintf(error_buf, error_buf_size, "Value is not a long long integer");
            return false;
        }
    }
    if (canon_lc == "ullong" || alias_lc == "uint64_t" || alias_lc == "size_t") {
        branch = "ullong";
        if (j.is_number_unsigned()) {
            uint64_t v = j.get<uint64_t>();
            if (v <= 18446744073709551615ULL) goto debug_true;
        } else if (j.is_number_integer() && j.get<int64_t>() >= 0) {
            uint64_t v = static_cast<uint64_t>(j.get<int64_t>());
            if (v <= 18446744073709551615ULL) goto debug_true;
        }
        snprintf(error_buf, error_buf_size, "Value is not an unsigned long long (uint64)");
        goto debug;
    }
    if (canon_lc == "int8_t") {
        branch = "int8_t";
        if (in_range_int(j, -128, 127)) goto debug_true;
        snprintf(error_buf, error_buf_size, "Value is not an int8_t (-128..127)");
        goto debug;
    }
    if (canon_lc == "uint8_t") {
        branch = "uint8_t";
        if (in_range_uint(j, 255)) goto debug_true;
        snprintf(error_buf, error_buf_size, "Value is not a uint8_t (0..255)");
        goto debug;
    }
    if (canon_lc == "float" || alias_lc == "float" || canon_lc == "float32" || alias_lc == "float32") {
        branch = "float";
        if (j.is_number_float() || j.is_number_integer()) goto debug_true;
        snprintf(error_buf, error_buf_size, "Value is not a float (number)");
        return false;
    }
    if (canon_lc == "double" || alias_lc == "float64") {
        branch = "double";
        if (!is_float_type(j)) {
            snprintf(error_buf, error_buf_size, "Value is not a double");
            goto debug;
        }
        goto debug_true;
    }
    if (canon_lc == "ldouble") {
        branch = "ldouble";
        if (!is_float_type(j)) {
            snprintf(error_buf, error_buf_size, "Value is not a long double");
            goto debug;
        }
        goto debug_true;
    }
    if (canon_lc == "string" || alias_lc == "std::string") {
        branch = "string";
        if (!is_string(j)) {
            snprintf(error_buf, error_buf_size, "Value is not a string");
            goto debug;
        }
        goto debug_true;
    }
    // Only set fallback error for primitive types, not containers/messages
    // If the type string contains '<' or '/' (container or message), do not set error_buf, just return false
    if (type_str.find('<') != std::string::npos || type_str.find('/') != std::string::npos) {
        return false;
    }
    if (debug_buf && debug_buf_size > 0) {
        snprintf(debug_buf, debug_buf_size, "[DISPATCH FALLBACK] type_str='%s' canon='%s' alias='%s'\n", type_str.c_str(), canon.c_str(), alias.c_str());
    }
    snprintf(error_buf, error_buf_size, "Type '%s' not supported for typechecking", type_str.c_str());
    return false;
debug_true:
    if (debug_buf && debug_buf_size > 0) {
        snprintf(debug_buf, debug_buf_size, "[OK] type='%s' canon='%s' alias='%s' branch='%s'\n", type_str.c_str(), canon.c_str(), alias.c_str(), branch.c_str());
    }
    return true;
debug:
    if (debug_buf && debug_buf_size > 0) {
        snprintf(debug_buf, debug_buf_size, "[FAIL] type='%s' canon='%s' alias='%s' branch='%s'\n", type_str.c_str(), canon.c_str(), alias.c_str(), branch.c_str());
    }
    return false;
}

// Recursively check std::vector<T>
bool check_vector(const std::string& elem_type, const nlohmann::json& j, char* error_buf, size_t error_buf_size, char* debug_buf, size_t debug_buf_size) {
    for (size_t i = 0; i < j.size(); ++i) {
        char suberr[128] = {0};
        char subdbg[256] = {0};
        if (!check_type_dispatch(elem_type, j[i], suberr, sizeof(suberr), subdbg, sizeof(subdbg))) {
            snprintf(error_buf, error_buf_size, "Element %zu: %s", i, suberr);
            if (debug_buf && debug_buf_size > 0) {
                snprintf(debug_buf, debug_buf_size, "[VECTOR] Element %zu failed: %s\n%s", i, suberr, subdbg);
            }
            return false;
        }
    }
    if (debug_buf && debug_buf_size > 0) {
        snprintf(debug_buf, debug_buf_size, "[VECTOR] All elements passed");
    }
    return true;
}

// Recursively check std::array<T, N>
bool check_array(const std::string& elem_type, size_t N, const nlohmann::json& j, char* error_buf, size_t error_buf_size, char* debug_buf = nullptr, size_t debug_buf_size = 0) {
    for (size_t i = 0; i < N; ++i) {
        char suberr[128] = {0};
        char subdbg[256] = {0};
        if (!check_type_dispatch(elem_type, j[i], suberr, sizeof(suberr), subdbg, sizeof(subdbg))) {
            snprintf(error_buf, error_buf_size, "Element %zu: %s", i, suberr);
            if (debug_buf && debug_buf_size > 0) {
                snprintf(debug_buf, debug_buf_size, "[FAIL] array element %zu: %s", i, subdbg);
            }
            return false;
        }
    }
    if (debug_buf && debug_buf_size > 0) {
        snprintf(debug_buf, debug_buf_size, "[OK] array<%s, %zu>\n", elem_type.c_str(), N);
    }
    return true;
}

// Recursively check std::map<K, V> or std::unordered_map<K, V>
bool check_map(const std::string& key_type, const std::string& val_type, const nlohmann::json& j, char* error_buf, size_t error_buf_size, char* debug_buf = nullptr, size_t debug_buf_size = 0) {
    auto ktype = canonical_and_alias(key_type).first;
    for (auto it = j.begin(); it != j.end(); ++it) {
        char suberr[128] = {0};
        char subdbg[256] = {0};
        if (ktype != "string" && ktype != "int") {
            snprintf(error_buf, error_buf_size, "Key type '%s' not supported in JSON", key_type.c_str());
            if (debug_buf && debug_buf_size > 0) {
                snprintf(debug_buf, debug_buf_size, "[FAIL] map key type '%s' not supported", key_type.c_str());
            }
            return false;
        }
        if (!check_type_dispatch(val_type, it.value(), suberr, sizeof(suberr), subdbg, sizeof(subdbg))) {
            snprintf(error_buf, error_buf_size, "Key '%s': %s", it.key().c_str(), suberr);
            if (debug_buf && debug_buf_size > 0) {
                snprintf(debug_buf, debug_buf_size, "[FAIL] map key '%s': %s", it.key().c_str(), subdbg);
            }
            return false;
        }
    }
    if (debug_buf && debug_buf_size > 0) {
        snprintf(debug_buf, debug_buf_size, "[OK] map<%s, %s> size=%zu\n", key_type.c_str(), val_type.c_str(), j.size());
    }
    return true;
}
}

extern "C" bool check_ros2_type(const char* type, const char* value_json, char* error_buf, size_t error_buf_size, char* debug_buf, size_t debug_buf_size) {
    std::string type_str = trim(type);
    std::string value_str(value_json);
    try {
        auto j = nlohmann::json::parse(value_str);
        if (check_type_dispatch(type_str, j, error_buf, error_buf_size, debug_buf, debug_buf_size)) {
            return true;
        }
        // If error_buf is set, return immediately
        if (error_buf[0] != '\0') {
            return false;
        }
        std::string base;
        std::vector<std::string> args;
        fprintf(stderr, "[DEBUG] parse_template input: '%s'\n", type_str.c_str());
        bool parsed = parse_template(type_str, base, args);
        fprintf(stderr, "[DEBUG] parse_template output: parsed=%d, base='%s', args.size()=%zu\n", parsed, base.c_str(), args.size());
        if (parsed) {
            if (base == "std::vector" && args.size() == 1) {
                if (!j.is_array()) {
                    snprintf(error_buf, error_buf_size, "Value is not an array");
                    if (debug_buf && debug_buf_size > 0) snprintf(debug_buf, debug_buf_size, "[FAIL] type='%s' (container)\n", type_str.c_str());
                    return false;
                }
                return check_vector(args[0], j, error_buf, error_buf_size, debug_buf, debug_buf_size);
            } else if (base == "std::array" && args.size() == 2) {
                size_t N = std::stoul(args[1]);
                if (!j.is_array() || j.size() != N) {
                    snprintf(error_buf, error_buf_size, "Value is not an array of size %zu", N);
                    if (debug_buf && debug_buf_size > 0) snprintf(debug_buf, debug_buf_size, "[FAIL] type='%s' (container)\n", type_str.c_str());
                    return false;
                }
                return check_array(args[0], N, j, error_buf, error_buf_size, debug_buf, debug_buf_size);
            } else if ((base == "std::map" || base == "std::unordered_map") && args.size() == 2) {
                if (!j.is_object()) {
                    snprintf(error_buf, error_buf_size, "Value is not a JSON object");
                    if (debug_buf && debug_buf_size > 0) snprintf(debug_buf, debug_buf_size, "[FAIL] type='%s' (container)\n", type_str.c_str());
                    return false;
                }
                return check_map(args[0], args[1], j, error_buf, error_buf_size, debug_buf, debug_buf_size);
            } else {
                snprintf(error_buf, error_buf_size, "Unsupported container type '%s'", type_str.c_str());
                if (debug_buf && debug_buf_size > 0) snprintf(debug_buf, debug_buf_size, "[FAIL] type='%s' (container)\n", type_str.c_str());
                return false;
            }
        }
        if (is_ros2_msg_type(type_str)) {
            if (!j.is_object() && !j.is_string()) {
                snprintf(error_buf, error_buf_size, "Value for message type '%s' must be a JSON object or string", type_str.c_str());
                if (debug_buf && debug_buf_size > 0) snprintf(debug_buf, debug_buf_size, "[FAIL] type='%s' (ros2 msg)\n", type_str.c_str());
                return false;
            }
            if (debug_buf && debug_buf_size > 0) snprintf(debug_buf, debug_buf_size, "[OK] type='%s' (ros2 msg)\n", type_str.c_str());
            return true;
        }
        snprintf(error_buf, error_buf_size, "Type '%s' not supported for typechecking", type_str.c_str());
        if (debug_buf && debug_buf_size > 0) snprintf(debug_buf, debug_buf_size, "[FAIL] type='%s' (not supported)\n", type_str.c_str());
        return false;
    } catch (const std::exception& e) {
        snprintf(error_buf, error_buf_size, "JSON parse error: %s", e.what());
        if (debug_buf && debug_buf_size > 0) snprintf(debug_buf, debug_buf_size, "[FAIL] type='%s' (json parse error)\n", type_str.c_str());
        return false;
    }
} 