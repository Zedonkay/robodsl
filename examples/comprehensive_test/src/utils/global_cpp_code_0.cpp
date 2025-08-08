
    // Additional C++ code that gets included in the generated files
    namespace robot_utils {
        // Utility functions
        template<typename T>
        T clamp(T value, T min, T max) {
            return std::max(min, std::min(max, value));
        }
        
        double radians_to_degrees(double radians) {
            return radians * 180.0 / M_PI;
        }
        
        double degrees_to_radians(double degrees) {
            return degrees * M_PI / 180.0;
        }
    }
