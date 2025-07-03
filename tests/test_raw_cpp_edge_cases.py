#!/usr/bin/env python3

import pytest
from pathlib import Path
from robodsl.parsers.lark_parser import RoboDSLParser
from robodsl.generators.main_generator import MainGenerator


class TestRawCppEdgeCases:
    """Test edge cases and complex C++ syntax in raw C++ code blocks."""
    
    def test_empty_cpp_block(self):
        """Test empty C++ code blocks."""
        robodsl_code = """
        cpp: {
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        assert ast.raw_cpp_code[0].code.strip() == "{}"
        assert len(ast.nodes) == 1
    
    def test_cpp_block_with_only_whitespace(self):
        """Test C++ blocks with only whitespace and comments."""
        robodsl_code = """
        cpp: {
            // This is a comment
            /* Multi-line comment */
            
            
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "// This is a comment" in code
        assert "/* Multi-line comment */" in code
    
    def test_nested_braces(self):
        """Test deeply nested braces in C++ code."""
        robodsl_code = """
        cpp: {
            namespace outer {
                namespace inner {
                    class DeeplyNested {
                    public:
                        void method() {
                            if (condition) {
                                for (int i = 0; i < 10; i++) {
                                    while (true) {
                                        switch (value) {
                                            case 1: {
                                                do {
                                                    // Nested code
                                                } while (false);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    };
                }
            }
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "namespace outer" in code
        assert "namespace inner" in code
        assert "class DeeplyNested" in code
        assert "switch (value)" in code
    
    def test_template_specialization(self):
        """Test complex template specializations."""
        robodsl_code = """
        cpp: {
            template<typename T>
            class Container {
            public:
                T value;
            };
            
            template<>
            class Container<int> {
            public:
                int value;
                void special_method() {}
            };
            
            template<typename T, typename U = std::vector<T>>
            class AdvancedContainer {
            private:
                U data;
            public:
                template<typename V>
                void process(V&& item) {
                    data.push_back(std::forward<V>(item));
                }
            };
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "template<typename T>" in code
        assert "template<>" in code
        assert "template<typename T, typename U = std::vector<T>>" in code
        assert "template<typename V>" in code
    
    def test_lambda_expressions(self):
        """Test lambda expressions and captures."""
        robodsl_code = """
        cpp: {
            auto lambda1 = []() { return 42; };
            auto lambda2 = [](int x) -> int { return x * 2; };
            auto lambda3 = [&](const auto& item) { process(item); };
            auto lambda4 = [=, &result](int value) mutable { result += value; };
            
            std::function<void(int)> callback = [this](int value) {
                this->process_value(value);
            };
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "auto lambda1 = []()" in code
        assert "auto lambda2 = [](int x) -> int" in code
        assert "auto lambda3 = [&](const auto& item)" in code
        assert "auto lambda4 = [=, &result](int value) mutable" in code
    
    def test_preprocessor_directives(self):
        """Test preprocessor directives and macros."""
        robodsl_code = """
        cpp: {
            #ifdef DEBUG
            #define LOG(msg) std::cout << "[DEBUG] " << msg << std::endl
            #else
            #define LOG(msg) ((void)0)
            #endif
            
            #if __cplusplus >= 201703L
            #define USE_CXX17 1
            #else
            #define USE_CXX17 0
            #endif
            
            #pragma once
            #pragma GCC optimize("O3")
            
            #include <iostream>
            #include <vector>
            
            #undef LOG
            #define LOG(msg) std::cerr << "[ERROR] " << msg << std::endl
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "#ifdef DEBUG" in code
        assert "#define LOG(msg)" in code
        assert "#pragma once" in code
        assert "#include <iostream>" in code
    
    def test_string_literals(self):
        """Test various string literal types."""
        robodsl_code = r"""
        cpp: {
            const char* str1 = "Hello World";
            const wchar_t* str2 = L"Wide String";
            const char16_t* str3 = u"UTF-16 String";
            const char32_t* str4 = U"UTF-32 String";
    
            std::string str5 = R"(Raw string with "quotes" and \backslashes)";
            std::wstring str6 = LR"(Wide raw string)";
    
            auto str7 = "Auto string literal";
            auto str8 = u8"UTF-8 string";
        }
    
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert 'const char* str1 = "Hello World"' in code
        assert 'const wchar_t* str2 = L"Wide String"' in code
        assert 'const char16_t* str3 = u"UTF-16 String"' in code
        assert 'R"(Raw string with "quotes" and \\backslashes)"' in code
    
    def test_operator_overloading(self):
        """Test operator overloading."""
        robodsl_code = """
        cpp: {
            class Complex {
            private:
                double real, imag;
            public:
                Complex(double r, double i) : real(r), imag(i) {}
                
                Complex operator+(const Complex& other) const {
                    return Complex(real + other.real, imag + other.imag);
                }
                
                Complex& operator+=(const Complex& other) {
                    real += other.real;
                    imag += other.imag;
                    return *this;
                }
                
                bool operator==(const Complex& other) const {
                    return real == other.real && imag == other.imag;
                }
                
                double operator[](int index) const {
                    return index == 0 ? real : imag;
                }
                
                operator double() const {
                    return std::sqrt(real*real + imag*imag);
                }
            };
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "Complex operator+(const Complex& other) const" in code
        assert "Complex& operator+=" in code
        assert "bool operator==" in code
        assert "double operator[](int index) const" in code
        assert "operator double() const" in code
    
    def test_variadic_templates(self):
        """Test variadic templates and fold expressions."""
        robodsl_code = """
        cpp: {
            template<typename... Args>
            void print_all(Args&&... args) {
                (std::cout << ... << args) << std::endl;
            }
            
            template<typename T, typename... Rest>
            class Tuple {
            private:
                T first;
                Tuple<Rest...> rest;
            public:
                Tuple(T&& f, Rest&&... r) 
                    : first(std::forward<T>(f)), rest(std::forward<Rest>(r)...) {}
            };
            
            template<typename T>
            class Tuple<T> {
            private:
                T first;
            public:
                Tuple(T&& f) : first(std::forward<T>(f)) {}
            };
            
            template<typename... Types>
            auto sum_all(Types... values) {
                return (... + values);
            }
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "template<typename... Args>" in code
        assert "(std::cout << ... << args)" in code
        assert "template<typename T, typename... Rest>" in code
        assert "return (... + values);" in code
    
    def test_concepts_and_constraints(self):
        """Test C++20 concepts and constraints."""
        robodsl_code = """
        cpp: {
            #if __cplusplus >= 202002L
            template<typename T>
            concept Numeric = std::is_arithmetic_v<T>;
            
            template<typename T>
            concept Container = requires(T t) {
                { t.begin() } -> std::input_iterator;
                { t.end() } -> std::input_iterator;
                { t.size() } -> std::convertible_to<std::size_t>;
            };
            
            template<Numeric T>
            T add(T a, T b) {
                return a + b;
            }
            
            template<Container C>
            auto process_container(const C& container) {
                return std::accumulate(container.begin(), container.end(), 
                                     typename C::value_type{});
            }
            
            template<typename T>
            requires Numeric<T> && std::is_integral_v<T>
            T multiply(T a, T b) {
                return a * b;
            }
            #endif
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "#if __cplusplus >= 202002L" in code
        assert "concept Numeric" in code
        assert "concept Container" in code
        assert "template<Numeric T>" in code
        assert "requires Numeric<T>" in code
    
    def test_coroutines(self):
        """Test C++20 coroutines."""
        robodsl_code = """
        cpp: {
            #if __cplusplus >= 202002L
            #include <coroutine>
            #include <future>
            
            template<typename T>
            struct Generator {
                struct promise_type {
                    T current_value;
                    
                    Generator get_return_object() {
                        return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
                    }
                    
                    std::suspend_always initial_suspend() { return {}; }
                    std::suspend_always final_suspend() noexcept { return {}; }
                    void return_void() {}
                    std::suspend_always yield_value(T value) {
                        current_value = value;
                        return {};
                    }
                    void unhandled_exception() { std::terminate(); }
                };
                
                bool next() {
                    if (coro) {
                        coro.resume();
                        return !coro.done();
                    }
                    return false;
                }
                
                T current_value() { return coro.promise().current_value; }
                
                Generator(std::coroutine_handle<promise_type> h) : coro(h) {}
                ~Generator() { if (coro) coro.destroy(); }
                
            private:
                std::coroutine_handle<promise_type> coro;
            };
            
            Generator<int> fibonacci(int n) {
                int a = 0, b = 1;
                for (int i = 0; i < n; ++i) {
                    co_yield a;
                    int temp = a;
                    a = b;
                    b = temp + b;
                }
            }
            #endif
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "#include <coroutine>" in code
        assert "struct promise_type" in code
        assert "co_yield a;" in code
        assert "std::suspend_always" in code
    
    def test_attributes(self):
        """Test C++ attributes."""
        robodsl_code = """
        cpp: {
            [[nodiscard]] int important_function() {
                return 42;
            }
            
            [[deprecated("Use new_function instead")]]
            void old_function() {}
            
            [[maybe_unused]] void debug_function() {}
            
            [[noreturn]] void terminate_program() {
                std::exit(1);
            }
            
            class [[nodiscard]] ImportantClass {
            public:
                [[deprecated]] void old_method() {}
            };
            
            enum class [[nodiscard]] Status {
                Success,
                Failure
            };
        }
        
        node test_node {
            parameter int value = 42
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "[[nodiscard]] int important_function()" in code
        assert '[[deprecated("Use new_function instead")]]' in code
        assert "[[maybe_unused]] void debug_function()" in code
        assert "[[noreturn]] void terminate_program()" in code
        assert "class [[nodiscard]] ImportantClass" in code
    
    def test_multiple_cpp_blocks_with_complex_content(self):
        """Test multiple C++ blocks with very complex content."""
        robodsl_code = """
        cpp: {
            // First block: Complex template metaprogramming
            template<int N>
            struct Factorial {
                static constexpr int value = N * Factorial<N-1>::value;
            };
            
            template<>
            struct Factorial<0> {
                static constexpr int value = 1;
            };
            
            template<typename... Args>
            struct TypeList {};
            
            template<typename List>
            struct Length;
            
            template<typename... Args>
            struct Length<TypeList<Args...>> {
                static constexpr int value = sizeof...(Args);
            };
        }
        
        cpp: {
            // Second block: Advanced SFINAE
            template<typename T, typename = void>
            struct has_size : std::false_type {};
            
            template<typename T>
            struct has_size<T, std::void_t<decltype(std::declval<T>().size())>> 
                : std::true_type {};
            
            template<typename T>
            constexpr bool has_size_v = has_size<T>::value;
            
            // CRTP example
            template<typename Derived>
            class Base {
            public:
                void interface() {
                    static_cast<Derived*>(this)->implementation();
                }
            };
            
            class Derived : public Base<Derived> {
            public:
                void implementation() {
                    // Implementation
                }
            };
        }
        
        cpp: {
            // Third block: Modern C++ features
            #if __cplusplus >= 201703L
            template<typename... Args>
            auto make_tuple(Args&&... args) {
                return std::tuple<Args...>(std::forward<Args>(args)...);
            }
            
            template<typename... Args>
            auto sum(Args... args) {
                return (... + args);
            }
            
            template<typename T>
            concept Addable = requires(T a, T b) {
                { a + b } -> std::convertible_to<T>;
            };
            #endif
        }
        
        node test_node {
            parameter int value = 42
            cpp: {
                // Node-level complex code
                template<typename T>
                class NodeHelper {
                private:
                    T data_;
                    std::mutex mutex_;
                    
                public:
                    template<typename U>
                    void process(U&& item) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        data_ = std::forward<U>(item);
                    }
                    
                    [[nodiscard]] T get() const {
                        std::lock_guard<std::mutex> lock(mutex_);
                        return data_;
                    }
                };
            }
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        assert len(ast.raw_cpp_code) == 3
        assert len(ast.nodes) == 1
        assert len(ast.nodes[0].content.raw_cpp_code) == 1
        
        # Check global blocks
        global_code = " ".join(block.code for block in ast.raw_cpp_code)
        assert "struct Factorial" in global_code
        assert "struct TypeList" in global_code
        assert "struct has_size" in global_code
        assert "class Base" in global_code
        assert "concept Addable" in global_code
        
        # Check node-level block
        node_code = ast.nodes[0].content.raw_cpp_code[0].code
        assert "class NodeHelper" in node_code
        assert "std::mutex mutex_" in node_code
        assert "[[nodiscard]] T get()" in node_code
    
    def test_generation_with_complex_cpp(self):
        """Test that complex C++ code gets properly generated."""
        robodsl_code = """
        cpp: {
            namespace complex_cpp {
                template<typename T>
                class AdvancedProcessor {
                private:
                    std::vector<T> data_;
                    std::mutex mutex_;
                    
                public:
                    template<typename U>
                    void add_data(U&& item) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        data_.emplace_back(std::forward<U>(item));
                    }
                    
                    [[nodiscard]] auto process_data() const {
                        std::lock_guard<std::mutex> lock(mutex_);
                        return std::accumulate(data_.begin(), data_.end(), T{});
                    }
                    
                    template<typename Predicate>
                    auto filter_data(Predicate&& pred) const {
                        std::lock_guard<std::mutex> lock(mutex_);
                        std::vector<T> result;
                        std::copy_if(data_.begin(), data_.end(), 
                                   std::back_inserter(result), 
                                   std::forward<Predicate>(pred));
                        return result;
                    }
                };
            }
        }
        
        node complex_node {
            parameter int buffer_size = 1024
            parameter double threshold = 0.5
            
            publisher /processed_data: "std_msgs/msg/Float64MultiArray"
            subscriber /raw_data: "std_msgs/msg/Float64MultiArray"
            
            cpp: {
                class NodeProcessor {
                private:
                    complex_cpp::AdvancedProcessor<double> processor_;
                    int buffer_size_;
                    double threshold_;
                    
                public:
                    NodeProcessor(int buffer_size, double threshold)
                        : buffer_size_(buffer_size), threshold_(threshold) {}
                    
                    void process_input(const std::vector<double>& input) {
                        for (double value : input) {
                            if (value > threshold_) {
                                processor_.add_data(value * 2.0);
                            } else {
                                processor_.add_data(value);
                            }
                        }
                    }
                    
                    [[nodiscard]] auto get_processed_data() const {
                        return processor_.process_data();
                    }
                    
                    auto get_filtered_data() const {
                        return processor_.filter_data([](double x) { return x > 0; });
                    }
                };
            }
        }
        """
        
        # Parse and generate
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        generator = MainGenerator(output_dir="test_output")
        generated_files = generator.generate(ast)
        
        # Check that files were generated
        assert len(generated_files) > 0
        
        # Find the node files
        node_files = [f for f in generated_files if "complex_node" in str(f)]
        assert len(node_files) >= 2  # Header and source
        
        # Check header file
        header_file = next(f for f in node_files if f.suffix == '.hpp')
        header_content = header_file.read_text()
        
        # Check source file
        source_file = next(f for f in node_files if f.suffix == '.cpp')
        source_content = source_file.read_text()
        
        # Verify complex C++ code is present
        assert "namespace complex_cpp" in header_content
        assert "class AdvancedProcessor" in header_content
        assert "template<typename T>" in header_content
        assert "[[nodiscard]]" in header_content
        assert "class NodeProcessor" in header_content
        
        assert "namespace complex_cpp" in source_content
        assert "class AdvancedProcessor" in source_content
        assert "class NodeProcessor" in source_content
        assert "std::lock_guard<std::mutex>" in source_content

    def test_cpp_block_invalid_syntax(self):
        robodsl_code = "cpp: { int x = ; } node n { parameter int x = 1 }"
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        assert "int x = ;" in ast.raw_cpp_code[0].code

    def test_cpp_block_deeply_nested_templates(self):
        robodsl_code = """
        cpp: {
            template<typename T> struct A { template<typename U> struct B { template<typename V> struct C {}; }; };
        }
        node n { parameter int x = 1 }
        """
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        assert "struct C" in ast.raw_cpp_code[0].code

    def test_cpp_block_injection(self):
        robodsl_code = 'cpp: { int x = 0; /* malicious */ system("rm -rf /"); } node n { parameter int x = 1 }'
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        assert "system(\"rm -rf /\");" in ast.raw_cpp_code[0].code 