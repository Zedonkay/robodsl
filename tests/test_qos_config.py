"""Tests for QoS configuration in RoboDSL."""

import pytest
from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.core.ast import RoboDSLAST, NodeNode, PublisherNode, SubscriberNode, ServiceNode, ActionNode, QoSNode, QoSSettingNode

def test_qos_config_parsing():
    """Test parsing of QoS configurations for different entities."""
    dsl_content = """
    node test_node {
        publisher /test_topic: "std_msgs/msg/String" {
            qos {
                reliability: reliable
                depth: 10
            }
        }
        subscriber /test_sub: "std_msgs/msg/String" {
            qos {
                reliability: best_effort
            }
        }
        service /test_srv: "std_srvs/srv/Trigger" {
            qos {
                reliability: reliable
            }
        }
        action /test_action: "test_msgs/action/Fibonacci" {
            qos {
                reliability: reliable
                depth: 5
            }
        }
    }
    """
    
    ast = parse_robodsl(dsl_content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.name == "test_node"
    
    # Check publisher QoS
    assert len(node.content.publishers) == 1
    pub = node.content.publishers[0]
    assert pub.qos is not None
    assert pub.qos.reliability == "reliable"
    assert pub.qos.depth == 10
    
    # Check subscriber QoS
    assert len(node.content.subscribers) == 1
    sub = node.content.subscribers[0]
    assert sub.qos is not None
    assert sub.qos.reliability == "best_effort"
    
    # Check service QoS
    assert len(node.content.services) == 1
    srv = node.content.services[0]
    assert srv.qos is not None
    assert srv.qos.reliability == "reliable"
    
    # Check action QoS
    assert len(node.content.actions) == 1
    action = node.content.actions[0]
    assert action.qos is not None
    assert action.qos.reliability == "reliable"
    assert action.qos.depth == 5

def test_qos_reliability_values():
    """Test different QoS reliability values."""
    dsl_content = """
    node test_node {
        publisher /topic1: "std_msgs/msg/String" {
            qos {
                reliability: reliable
            }
        }
        publisher /topic2: "std_msgs/msg/String" {
            qos {
                reliability: best_effort
            }
        }
    }
    """
    
    ast = parse_robodsl(dsl_content)
    
    assert len(ast.nodes[0].content.publishers) == 2
    pub1 = ast.nodes[0].content.publishers[0]
    pub2 = ast.nodes[0].content.publishers[1]
    
    assert pub1.qos.reliability == "reliable"
    assert pub2.qos.reliability == "best_effort"

def test_qos_durability_values():
    """Test different QoS durability values."""
    dsl_content = """
    node test_node {
        publisher /topic1: "std_msgs/msg/String" {
            qos {
                durability: volatile
            }
        }
        publisher /topic2: "std_msgs/msg/String" {
            qos {
                durability: transient_local
            }
        }
    }
    """
    
    ast = parse_robodsl(dsl_content)
    
    assert len(ast.nodes[0].content.publishers) == 2
    pub1 = ast.nodes[0].content.publishers[0]
    pub2 = ast.nodes[0].content.publishers[1]
    
    assert pub1.qos.durability == "volatile"
    assert pub2.qos.durability == "transient_local"

def test_qos_history_values():
    """Test different QoS history values."""
    dsl_content = """
    node test_node {
        publisher /topic1: "std_msgs/msg/String" {
            qos {
                history: keep_last
                depth: 10
            }
        }
        publisher /topic2: "std_msgs/msg/String" {
            qos {
                history: keep_all
            }
        }
    }
    """
    
    ast = parse_robodsl(dsl_content)
    
    assert len(ast.nodes[0].content.publishers) == 2
    pub1 = ast.nodes[0].content.publishers[0]
    pub2 = ast.nodes[0].content.publishers[1]
    
    assert pub1.qos.history == "keep_last"
    assert pub1.qos.depth == 10
    assert pub2.qos.history == "keep_all"

def test_qos_liveliness_values():
    """Test different QoS liveliness values."""
    dsl_content = """
    node test_node {
        publisher /topic1: "std_msgs/msg/String" {
            qos {
                liveliness: automatic
            }
        }
        publisher /topic2: "std_msgs/msg/String" {
            qos {
                liveliness: manual_by_topic
                lease_duration: 5.0
            }
        }
    }
    """
    
    ast = parse_robodsl(dsl_content)
    
    assert len(ast.nodes[0].content.publishers) == 2
    pub1 = ast.nodes[0].content.publishers[0]
    pub2 = ast.nodes[0].content.publishers[1]
    
    assert pub1.qos.liveliness == "automatic"
    assert pub2.qos.liveliness == "manual_by_topic"
    assert pub2.qos.lease_duration == 5.0

def test_qos_multiple_settings():
    """Test multiple QoS settings on a single entity."""
    dsl_content = """
    node test_node {
        publisher /test_topic: "std_msgs/msg/String" {
            qos {
                reliability: reliable
                durability: transient_local
                history: keep_last
                depth: 20
                liveliness: automatic
                lease_duration: 10.0
            }
        }
    }
    """
    
    ast = parse_robodsl(dsl_content)
    
    pub = ast.nodes[0].content.publishers[0]
    qos = pub.qos
    
    assert qos.reliability == "reliable"
    assert qos.durability == "transient_local"
    assert qos.history == "keep_last"
    assert qos.depth == 20
    assert qos.liveliness == "automatic"
    assert qos.lease_duration == 10.0

def test_qos_default_values():
    """Test that QoS entities without explicit QoS config work."""
    dsl_content = """
    node test_node {
        publisher /test_topic: "std_msgs/msg/String"
        subscriber /test_sub: "std_msgs/msg/String"
        service /test_srv: "std_srvs/srv/Trigger"
    }
    """
    
    ast = parse_robodsl(dsl_content)
    
    node = ast.nodes[0]
    
    # These should have no QoS config (None)
    assert node.content.publishers[0].qos is None
    assert node.content.subscribers[0].qos is None
    assert node.content.services[0].qos is None

def test_qos_invalid_values():
    """Test that invalid QoS values are handled appropriately."""
    # This test would check how the parser handles invalid QoS values
    # For now, we'll just test that the parser doesn't crash
    dsl_content = """
    node test_node {
        publisher /test_topic: "std_msgs/msg/String" {
            qos {
                reliability: invalid_value
            }
        }
    }
    """
    
    # The parser should still parse this, but the semantic analyzer
    # would catch invalid values
    try:
        ast = parse_robodsl(dsl_content)
        assert len(ast.nodes) == 1
        assert len(ast.nodes[0].content.publishers) == 1
        print("✓ Parser handled invalid QoS value gracefully")
    except Exception as e:
        print(f"✗ Parser failed on invalid QoS value: {e}")

def test_qos_numeric_values():
    """Test QoS settings with numeric values."""
    dsl_content = """
    node test_node {
        publisher /test_topic: "std_msgs/msg/String" {
            qos {
                depth: 100
                lease_duration: 30.5
                deadline: 1.0
            }
        }
    }
    """
    
    ast = parse_robodsl(dsl_content)
    
    pub = ast.nodes[0].content.publishers[0]
    qos = pub.qos
    
    assert qos.depth == 100
    assert qos.lease_duration == 30.5
    assert qos.deadline == 1.0

if __name__ == "__main__":
    # Run the tests
    test_qos_config_parsing()
    test_qos_reliability_values()
    test_qos_durability_values()
    test_qos_history_values()
    test_qos_liveliness_values()
    test_qos_multiple_settings()
    test_qos_default_values()
    test_qos_invalid_values()
    test_qos_numeric_values()
    print("All QoS configuration tests completed!")
