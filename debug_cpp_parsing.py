#!/usr/bin/env python3

from robodsl.parsers.lark_parser import RoboDSLParser

# Test string literals case
robodsl_code = """
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

# Test empty C++ block case
robodsl_code_empty = """
        cpp: {
        }
        
        node test_node {
            parameter int value = 42
        }
        """

parser = RoboDSLParser(debug=True)
try:
    ast = parser.parse(robodsl_code)
    print("Success!")
    print(f"Raw C++ code blocks: {len(ast.raw_cpp_code)}")
    for i, block in enumerate(ast.raw_cpp_code):
        print(f"Block {i}: {repr(block.code)}")
        print(f"Block {i} content:")
        print(block.code)
        
        # Check for the raw string literal
        if 'R"(' in block.code:
            print("Raw string literal found!")
            # Find the raw string
            start = block.code.find('R"(')
            end = block.code.find(')"', start)
            if end != -1:
                raw_string = block.code[start:end+2]
                print(f"Raw string: {repr(raw_string)}")
        
    # Test without debug mode
    print("\n" + "="*50 + "\n")
    print("Testing without debug mode:")
    parser2 = RoboDSLParser(debug=False)
    ast2 = parser2.parse(robodsl_code)
    print(f"Raw C++ code blocks: {len(ast2.raw_cpp_code)}")
    for i, block in enumerate(ast2.raw_cpp_code):
        print(f"Block {i}: {repr(block.code)}")
        print(f"Block {i} content:")
        print(block.code)
        
    # Test empty C++ block case
    print("\n" + "="*50 + "\n")
    print("Testing empty C++ block case:")
    ast_empty = parser.parse(robodsl_code_empty)
    print(f"Raw C++ code blocks: {len(ast_empty.raw_cpp_code)}")
    for i, block in enumerate(ast_empty.raw_cpp_code):
        print(f"Block {i}: {repr(block.code)}")
        print(f"Block {i} content:")
        print(block.code)
        print(f"Block {i} stripped: {repr(block.code.strip())}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 