#!/usr/bin/env python3

from robodsl.parsers.lark_parser import RoboDSLParser
from lark import Lark

# Test the grammar directly
grammar_file = "src/robodsl/grammar/robodsl.lark"
with open(grammar_file, 'r') as f:
    grammar_content = f.read()

parser = Lark(grammar_content, parser='lalr', start='start')

# Test ONNX model within node
test_code = '''
node image_classifier {
    subscriber /camera/image_raw: "sensor_msgs/msg/Image"
    publisher /classification/result: "std_msgs/msg/Float32MultiArray"
    parameter string model_path = "resnet50.onnx"
    
    onnx_model resnet50 {
        input: "input" -> "float32[1,3,224,224]"
        output: "output" -> "float32[1,1000]"
        device: cuda
        optimization: tensorrt
    }
}
'''

try:
    parse_tree = parser.parse(test_code)
    print("Parse tree:")
    print(parse_tree.pretty())
except Exception as e:
    print(f"Parse error: {e}")

# Test with RoboDSL parser
try:
    robodsl_parser = RoboDSLParser()
    ast = robodsl_parser.parse(test_code)
    print(f"\nAST nodes: {len(ast.nodes)}")
    if ast.nodes:
        node = ast.nodes[0]
        print(f"Node name: {node.name}")
        print(f"Node subscribers: {len(node.content.subscribers)}")
        print(f"Node publishers: {len(node.content.publishers)}")
        print(f"Node parameters: {len(node.content.parameters)}")
        print(f"Node ONNX models: {len(node.content.onnx_models)}")
        
        if node.content.onnx_models:
            model = node.content.onnx_models[0]
            print(f"Model name: {model.name}")
            print(f"Device: {model.config.device.device if model.config.device else 'None'}")
            print(f"Optimizations: {[opt.optimization for opt in model.config.optimizations]}")
except Exception as e:
    print(f"RoboDSL parse error: {e}") 