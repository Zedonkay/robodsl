"""Minimal test of Click command groups."""

import click
from click.testing import CliRunner

@click.group()
def cli():
    """A simple CLI group."""
    pass

@cli.command()
@click.argument('name')
def hello(name):
    """Say hello to someone."""
    click.echo(f"Hello, {name}!")

def test_hello():
    """Test the hello command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["hello", "world"], catch_exceptions=False)
    print("\n=== OUTPUT ===")
    print(result.output)
    print("=== EXCEPTION ===")
    print(result.exception)
    print("==============")
    assert result.exit_code == 0
    assert "Hello, world!" in result.output
