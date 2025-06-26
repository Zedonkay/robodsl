"""Tests for QoS configuration in RoboDSL."""

import pytest
from robodsl.parser import parse_robodsl, QoSConfig, PublisherConfig, SubscriberConfig, ServiceConfig, ActionConfig, NodeConfig

def test_qos_config_parsing():
    """Test parsing of QoS configurations for different entities."""
    dsl_content = """
    node test_node {
        publisher /test_topic std_msgs/msg/String qos reliability=reliable depth=10
        subscriber /test_sub std_msgs/msg/String qos reliability=best_effort
        service /test_srv std_srvs/srv/SetBool qos reliability=reliable
        action /test_action test_msgs/action/Fibonacci qos reliability=best_effort
    }
    """
    
    config = parse_robodsl(dsl_content)
    node = config.nodes[0]
    
    # Check publisher QoS
    assert len(node.publishers) == 1
    assert isinstance(node.publishers[0], PublisherConfig)
    assert node.publishers[0].qos is not None
    assert node.publishers[0].qos.reliability == 'reliable'
    assert node.publishers[0].qos.depth == 10
    
    # Check subscriber QoS
    assert len(node.subscribers) == 1
    assert isinstance(node.subscribers[0], SubscriberConfig)
    assert node.subscribers[0].qos is not None
    assert node.subscribers[0].qos.reliability == 'best_effort'
    
    # Check service QoS
    assert len(node.services) == 1
    assert isinstance(node.services[0], ServiceConfig)
    assert node.services[0].qos is not None
    assert node.services[0].qos.reliability == 'reliable'
    
    # Check action QoS
    assert len(node.actions) == 1
    assert isinstance(node.actions[0], ActionConfig)
    assert node.actions[0].qos is not None
    assert node.actions[0].qos.reliability == 'best_effort'

def test_qos_config_structure():
    """Test the structure of QoS configuration objects."""
    # Test creating a QoS config with valid values
    qos = QoSConfig(reliability='reliable', depth=10)
    assert qos.reliability == 'reliable'
    assert qos.depth == 10
    
    # Test default values
    qos_default = QoSConfig()
    assert qos_default.reliability is None
    assert qos_default.depth == 10  # Default value from class definition

if __name__ == '__main__':
    pytest.main()
