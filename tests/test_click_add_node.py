"""Test Click add-node command."""

import click
from click.testing import CliRunner
from pathlib import Path

@click.group()
def cli():
    """A CLI group for testing add-node command."""
    pass

@cli.command(name='add-node')
@click.argument('node_name')
@click.option('--language', '-l', default='python', help='Programming language for the node')
@click.option('--project-dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), 
              default=Path.cwd(), help='Project directory')
def add_node(node_name, language, project_dir):
    """Add a new node to the project."""
    click.echo(f"Adding node '{node_name}' with language '{language}' to project at '{project_dir}'")
    click.echo("Node created successfully!")

def test_add_node():
    """Test the add-node command."""
    runner = CliRunner()
    with runner.isolated_filesystem() as temp_dir:
        project_dir = Path(temp_dir) / "test_project"
        project_dir.mkdir()
        
        result = runner.invoke(
            cli,
            ["add-node", "camera", "--language", "python", "--project-dir", str(project_dir)],
            catch_exceptions=False
        )
        
        print("\n=== OUTPUT ===")
        print(result.output)
        print("=== EXCEPTION ===")
        print(result.exception)
        print("==============")
        
        assert result.exit_code == 0
        assert "Adding node 'camera' with language 'python'" in result.output
        assert "Node created successfully!" in result.output
