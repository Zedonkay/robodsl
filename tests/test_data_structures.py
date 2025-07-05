import pytest
from pathlib import Path
from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.core.ast import (
    StructNode, ClassNode, EnumNode, TypedefNode, UsingNode, NodeNode
)

def test_data_structures_parsing():
    test_file = Path(__file__).parent / "data_structures_test.robodsl"
    ast = parse_robodsl(test_file.read_text(), debug=True)

    # Check typedef
    typedefs = [ds for ds in ast.data_structures if isinstance(ds, TypedefNode)]
    assert any(td.new_name == "FloatVector" for td in typedefs)

    # Check using
    usings = [ds for ds in ast.data_structures if isinstance(ds, UsingNode)]
    assert any(u.new_name == "Point3D" for u in usings)

    # Check enums
    enums = [ds for ds in ast.data_structures if isinstance(ds, EnumNode)]
    assert any(e.name == "SensorType" and e.enum_type == "class" for e in enums)
    assert any(e.name == "ProcessingMode" and e.enum_type is None for e in enums)

    # Check structs
    structs = [ds for ds in ast.data_structures if isinstance(ds, StructNode)]
    assert any(s.name == "SensorConfig" for s in structs)
    assert any(s.name == "ArrayHolder" for s in structs)

    # Check class
    classes = [ds for ds in ast.data_structures if isinstance(ds, ClassNode)]
    assert any(c.name == "DataProcessor" for c in classes)
    processor = next(c for c in classes if c.name == "DataProcessor")
    assert processor.inheritance is not None
    assert any(base[1] == "BaseProcessor" for base in processor.inheritance.base_classes)
    # Check public and private sections
    assert any(sec.access_specifier == "public" for sec in processor.content.access_sections)
    assert any(sec.access_specifier == "private" for sec in processor.content.access_sections)

    # Check node
    nodes = ast.nodes
    assert any(n.name == "test_node" for n in nodes) 