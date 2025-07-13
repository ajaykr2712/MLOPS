"""
Command-line interface for MDT Dashboard.
Provides commands for running services and managing the platform.
"""

import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from pathlib import Path
import subprocess
import sys

app = typer.Typer(help="MDT Dashboard CLI - Model Drift Detection & Telemetry Platform")
console = Console()


@app.command()
def run_server(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server"),
    port: int = typer.Option(8000, help="Port to bind the server"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Run the FastAPI server."""
    console.print("🚀 Starting MDT Dashboard API Server...", style="bold green")
    
    try:
        uvicorn.run(
            "mdt_dashboard.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info"
        )
    except ImportError:
        console.print("❌ FastAPI dependencies not installed. Please install with:", style="bold red")
        console.print("pip install fastapi uvicorn", style="cyan")
        raise typer.Exit(1)


@app.command()
def run_dashboard(
    port: int = typer.Option(8501, help="Port to run Streamlit dashboard"),
    server_name: str = typer.Option("localhost", help="Server name for Streamlit"),
):
    """Run the Streamlit dashboard."""
    console.print("🎨 Starting MDT Dashboard UI...", style="bold blue")
    
    try:
        # Run streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "src/mdt_dashboard/dashboard/main.py",
            "--server.port", str(port),
            "--server.address", server_name,
            "--server.headless", "true"
        ]
        
        subprocess.run(cmd, check=True)
        
    except ImportError:
        console.print("❌ Streamlit not installed. Please install with:", style="bold red")
        console.print("pip install streamlit plotly", style="cyan")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Failed to start dashboard: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def run_worker(
    concurrency: int = typer.Option(4, help="Number of concurrent workers"),
    queue: str = typer.Option("default", help="Queue name to process"),
):
    """Run background worker for processing tasks."""
    console.print("⚙️ Starting MDT Background Worker...", style="bold yellow")
    
    try:
        # This would start a Celery worker or similar
        console.print(f"Worker starting with {concurrency} concurrent processes")
        console.print(f"Processing queue: {queue}")
        console.print("Worker functionality not implemented yet.", style="yellow")
        
    except Exception as e:
        console.print(f"❌ Failed to start worker: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def run_migration():
    """Run database migrations."""
    console.print("🗃️ Running database migrations...", style="bold cyan")
    
    try:
        # This would run Alembic migrations
        console.print("Migration functionality not implemented yet.", style="yellow")
        
    except Exception as e:
        console.print(f"❌ Migration failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def status():
    """Show status of MDT Dashboard services."""
    console.print("📊 MDT Dashboard Status", style="bold magenta")
    
    table = Table(title="Service Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Port", style="yellow")
    table.add_column("Description")
    
    # Check if services are running (simplified)
    table.add_row("API Server", "Unknown", "8000", "FastAPI REST API")
    table.add_row("Dashboard", "Unknown", "8501", "Streamlit Web UI")
    table.add_row("Worker", "Unknown", "-", "Background Task Processor")
    table.add_row("Database", "Unknown", "5432", "PostgreSQL Database")
    
    console.print(table)
    console.print("\n💡 Use specific commands to start services:", style="dim")
    console.print("  mdt-server     - Start API server", style="dim")
    console.print("  mdt-dashboard  - Start web dashboard", style="dim")
    console.print("  mdt-worker     - Start background worker", style="dim")


@app.command()
def init(
    project_name: str = typer.Option("mdt-project", help="Name of the project"),
    path: Path = typer.Option(".", help="Path to initialize project"),
):
    """Initialize a new MDT Dashboard project."""
    console.print(f"🏗️ Initializing MDT project: {project_name}", style="bold green")
    
    project_path = Path(path) / project_name
    
    try:
        # Create project structure
        project_path.mkdir(exist_ok=True)
        (project_path / "data").mkdir(exist_ok=True)
        (project_path / "models").mkdir(exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)
        (project_path / "config").mkdir(exist_ok=True)
        
        # Create basic config file
        config_content = """# MDT Dashboard Configuration
DATABASE_URL=postgresql://mdt_user:mdt_pass@localhost:5432/mdt_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-change-in-production
ENVIRONMENT=development
DEBUG=true
"""
        
        (project_path / ".env").write_text(config_content)
        
        # Create basic docker-compose file
        docker_compose = """version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mdt_db
      POSTGRES_USER: mdt_user
      POSTGRES_PASSWORD: mdt_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
"""
        
        (project_path / "docker-compose.yml").write_text(docker_compose)
        
        console.print(f"✅ Project initialized at: {project_path}", style="bold green")
        console.print("\n📝 Next steps:", style="bold")
        console.print(f"  cd {project_name}")
        console.print("  docker-compose up -d  # Start databases")
        console.print("  mdt-server            # Start API server")
        console.print("  mdt-dashboard         # Start dashboard")
        
    except Exception as e:
        console.print(f"❌ Failed to initialize project: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def version():
    """Show MDT Dashboard version."""
    from mdt_dashboard import __version__
    console.print(f"MDT Dashboard v{__version__}", style="bold green")


if __name__ == "__main__":
    app()
