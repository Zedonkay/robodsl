"""Package Generator for RoboDSL.

This generator creates package.xml and other package-related files.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..core.ast import RoboDSLAST


class PackageGenerator(BaseGenerator):
    """Generates package.xml and other package files."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate package files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Generate package.xml
        package_xml_path = self._generate_package_xml(ast)
        generated_files.append(package_xml_path)
        
        return generated_files
    
    def _generate_package_xml(self, ast: RoboDSLAST) -> Path:
        """Generate package.xml file."""
        context = self._prepare_package_context(ast)
        
        try:
            content = self.render_template('package.xml.jinja2', context)
            package_xml_path = self.output_dir / 'package.xml'
            return self.write_file(package_xml_path, content)
        except Exception as e:
            print(f"Template error for package.xml: {e}")
            # Fallback to simple package.xml
            content = self._generate_fallback_package_xml(ast)
            package_xml_path = self.output_dir / 'package.xml'
            return self.write_file(package_xml_path, content)
    
    def _prepare_package_context(self, ast: RoboDSLAST) -> Dict[str, Any]:
        """Prepare context for package template rendering."""
        # Get package information
        package_info = getattr(ast, 'package', {})
        package_name = package_info.get('name', 'robodsl_package')
        version = package_info.get('version', '0.1.0')
        description = package_info.get('description', 'Generated ROS2 package from RoboDSL specification')
        maintainer = package_info.get('maintainer', 'robot_team@example.com')
        license = package_info.get('license', 'Apache-2.0')
        
        # Collect all dependencies
        dependencies = set()
        optional_dependencies = set()
        system_dependencies = set()
        
        # Add basic ROS2 dependencies
        dependencies.update([
            'rclcpp',
            'std_msgs',
            'geometry_msgs',
            'sensor_msgs'
        ])
        
        # Add dependencies based on node content
        for node in ast.nodes:
            # Check for lifecycle nodes
            if getattr(node, 'lifecycle', False) or 'lifecycle' in node.name.lower():
                dependencies.add('rclcpp_lifecycle')
            
            # Check for action nodes
            if hasattr(node, 'actions') and node.actions:
                dependencies.add('rclcpp_action')
            
            # Check for service nodes
            if hasattr(node, 'services') and node.services:
                dependencies.add('std_srvs')
            
            # Check for message package usage (only add packages that appear)
            for pub in getattr(node, 'publishers', []):
                if 'nav_msgs' in pub.msg_type:
                    dependencies.add('nav_msgs')
                if 'visualization_msgs' in pub.msg_type:
                    dependencies.add('visualization_msgs')
                if 'tf2_msgs' in pub.msg_type:
                    dependencies.add('tf2_msgs')
                # vision_msgs removed unless explicitly present
                parts = pub.msg_type.split('/')
                if len(parts) >= 3:
                    dependencies.add(parts[0])
                if 'trajectory_msgs' in pub.msg_type:
                    dependencies.add('trajectory_msgs')
            
            for sub in getattr(node, 'subscribers', []):
                if 'nav_msgs' in sub.msg_type:
                    dependencies.add('nav_msgs')
                if 'visualization_msgs' in sub.msg_type:
                    dependencies.add('visualization_msgs')
                if 'tf2_msgs' in sub.msg_type:
                    dependencies.add('tf2_msgs')
                # vision_msgs removed unless explicitly present
                parts = sub.msg_type.split('/')
                if len(parts) >= 3:
                    dependencies.add(parts[0])
                if 'trajectory_msgs' in sub.msg_type:
                    dependencies.add('trajectory_msgs')

            # Services and actions: add their packages
            for srv in getattr(node, 'services', []):
                parts = srv.srv_type.split('/')
                if len(parts) >= 3:
                    dependencies.add(parts[0])
            for act in getattr(node, 'actions', []):
                parts = act.action_type.split('/')
                if len(parts) >= 3:
                    dependencies.add(parts[0])
        
        # Check for CUDA usage
        if hasattr(ast, 'cuda_kernels') and ast.cuda_kernels:
            system_dependencies.add('cuda')
        
        # Check for OpenCV usage
        opencv_needed = False
        if hasattr(ast, 'global_cpp_code'):
            for code_block in ast.global_cpp_code:
                if 'cv_bridge' in str(code_block) or 'opencv' in str(code_block) or 'cv::' in str(code_block):
                    opencv_needed = True
                    break
        
        if opencv_needed:
            dependencies.add('cv_bridge')
            system_dependencies.add('opencv')
        
        # Check for ONNX usage
        if hasattr(ast, 'onnx_models') and ast.onnx_models:
            system_dependencies.add('onnxruntime')
            system_dependencies.add('tensorrt')
        
        # Add test dependencies
        optional_dependencies.update([
            'ament_lint_auto',
            'ament_lint_common',
            'ament_cmake_gtest',
            'ament_cmake_pytest'
        ])
        
        return {
            'package_name': package_name,
            'version': version,
            'description': description,
            'maintainer': maintainer,
            'license': license,
            'dependencies': sorted(list(dependencies)),
            'optional_dependencies': sorted(list(optional_dependencies)),
            'system_dependencies': sorted(list(system_dependencies))
        }
    
    def _generate_fallback_package_xml(self, ast: RoboDSLAST) -> str:
        """Generate a fallback package.xml if template fails."""
        return f"""<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robodsl_package</name>
  <version>0.1.0</version>
  <description>Generated ROS2 package from RoboDSL specification</description>
  <maintainer email="robot_team@example.com">Robot Team</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>ament_cmake_python</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>cv_bridge</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>
  <test_depend>ament_cmake_gtest</test_depend>
  <test_depend>ament_cmake_pytest</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>""" 