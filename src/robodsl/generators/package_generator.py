"""Package Generator for RoboDSL.

This generator creates package.xml files for ROS2 packages.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..core.ast import RoboDSLAST, NodeNode


class PackageGenerator(BaseGenerator):
    """Generates ROS2 package.xml files."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate package.xml files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Generate package.xml
        package_path = self._generate_package_xml(ast)
        generated_files.append(package_path)
        
        return generated_files
    
    def _generate_package_xml(self, ast: RoboDSLAST) -> Path:
        """Generate the package.xml file."""
        context = self._prepare_package_context(ast)
        
        try:
            content = self.render_template('package.xml.jinja2', context)
            package_path = self.get_output_path('package.xml')
            return self.write_file(package_path, content)
        except Exception as e:
            print(f"Template error for package.xml: {e}")
            # Fallback to simple package.xml
            content = self._generate_fallback_package_xml(ast)
            package_path = self.get_output_path('package.xml')
            return self.write_file(package_path, content)
    
    def _prepare_package_context(self, ast: RoboDSLAST) -> Dict[str, Any]:
        """Prepare context for package.xml template rendering."""
        # Determine package name
        package_name = getattr(ast, 'package_name', 'robodsl_package')
        
        # Collect all dependencies
        dependencies = set()
        build_dependencies = set()
        exec_dependencies = set()
        test_dependencies = set()
        
        # Add basic ROS2 dependencies
        build_dependencies.update([
            'ament_cmake',
            'ament_cmake_python'
        ])
        
        exec_dependencies.update([
            'rclcpp',
            'std_msgs'
        ])
        
        # Check if we have any CUDA kernels
        has_cuda_kernels = bool(ast.cuda_kernels and ast.cuda_kernels.kernels)
        
        # Check nodes for CUDA kernels
        for node in ast.nodes:
            if node.content.cuda_kernels:
                has_cuda_kernels = True
        
        # Add CUDA dependencies if needed
        if has_cuda_kernels:
            build_dependencies.add('cuda')
            exec_dependencies.add('cuda_runtime')
        
        # Add lifecycle dependencies if any node is lifecycle
        for node in ast.nodes:
            if node.content.lifecycle:
                exec_dependencies.add('rclcpp_lifecycle')
                break
        
        # Collect message dependencies from publishers/subscribers
        for node in ast.nodes:
            for pub in node.content.publishers:
                if '.' in pub.msg_type:
                    package = pub.msg_type.split('.')[0]
                    exec_dependencies.add(package)
                else:
                    exec_dependencies.add('std_msgs')
            
            for sub in node.content.subscribers:
                if '.' in sub.msg_type:
                    package = sub.msg_type.split('.')[0]
                    exec_dependencies.add(package)
                else:
                    exec_dependencies.add('std_msgs')
            
            for srv in node.content.services:
                if '.' in srv.srv_type:
                    package = srv.srv_type.split('.')[0]
                    exec_dependencies.add(package)
                else:
                    exec_dependencies.add('std_srvs')
            
            for action in node.content.actions:
                if '.' in action.action_type:
                    package = action.action_type.split('.')[0]
                    exec_dependencies.add(package)
                else:
                    exec_dependencies.add('std_msgs')
        
        # Add test dependencies
        test_dependencies.update([
            'ament_lint_auto',
            'ament_lint_common'
        ])
        
        return {
            'package_name': package_name,
            'version': '0.1.0',
            'description': 'Generated ROS2 package from RoboDSL specification',
            'maintainer': 'robodsl',
            'maintainer_email': 'robodsl@example.com',
            'license': 'Apache-2.0',
            'build_dependencies': sorted(list(build_dependencies)),
            'exec_dependencies': sorted(list(exec_dependencies)),
            'test_dependencies': sorted(list(test_dependencies)),
            'has_cuda': has_cuda_kernels
        }
    
    def _generate_fallback_package_xml(self, ast: RoboDSLAST) -> str:
        """Generate a fallback package.xml if template fails."""
        package_name = getattr(ast, 'package_name', 'robodsl_package')
        
        content = f"""<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>0.1.0</version>
  <description>Generated ROS2 package from RoboDSL specification</description>
  <maintainer email="robodsl@example.com">robodsl</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>ament_cmake_python</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
"""
        
        # Add lifecycle dependency if any node is lifecycle
        for node in ast.nodes:
            if node.content.lifecycle:
                content += "  <depend>rclcpp_lifecycle</depend>\n"
                break
        
        # Add message dependencies
        message_packages = set()
        for node in ast.nodes:
            for pub in node.content.publishers:
                if '.' in pub.msg_type:
                    package = pub.msg_type.split('.')[0]
                    message_packages.add(package)
                else:
                    message_packages.add('std_msgs')
            
            for sub in node.content.subscribers:
                if '.' in sub.msg_type:
                    package = sub.msg_type.split('.')[0]
                    message_packages.add(package)
                else:
                    message_packages.add('std_msgs')
            
            for srv in node.content.services:
                if '.' in srv.srv_type:
                    package = srv.srv_type.split('.')[0]
                    message_packages.add(package)
                else:
                    message_packages.add('std_srvs')
            
            for action in node.content.actions:
                if '.' in action.action_type:
                    package = action.action_type.split('.')[0]
                    message_packages.add(package)
                else:
                    message_packages.add('std_msgs')
        
        for package in sorted(message_packages):
            content += f"  <depend>{package}</depend>\n"
        
        content += """
  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
"""
        
        return content 