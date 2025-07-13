#!/bin/bash

# Production deployment script for MDT Dashboard
# This script handles end-to-end deployment to various environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_ENV="${1:-staging}"
REGISTRY="ghcr.io"
IMAGE_PREFIX="mdt-dashboard"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if kubectl is installed (for k8s deployments)
    if [[ "$DEPLOY_ENV" == "production" ]] && ! command -v kubectl &> /dev/null; then
        log_warning "kubectl is not installed - Kubernetes deployment will not be available"
    fi
    
    log_success "Prerequisites check completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Get version from git tag or use timestamp
    VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "$(date +%Y%m%d%H%M%S)")
    
    # Build API image
    log_info "Building API image..."
    docker build -f docker/Dockerfile.api -t "${REGISTRY}/${IMAGE_PREFIX}-api:${VERSION}" \
        -t "${REGISTRY}/${IMAGE_PREFIX}-api:latest" .
    
    # Build Dashboard image
    log_info "Building Dashboard image..."
    docker build -f docker/Dockerfile.dashboard -t "${REGISTRY}/${IMAGE_PREFIX}-dashboard:${VERSION}" \
        -t "${REGISTRY}/${IMAGE_PREFIX}-dashboard:latest" .
    
    # Build Worker image
    log_info "Building Worker image..."
    docker build -f docker/Dockerfile.worker -t "${REGISTRY}/${IMAGE_PREFIX}-worker:${VERSION}" \
        -t "${REGISTRY}/${IMAGE_PREFIX}-worker:latest" .
    
    log_success "Docker images built successfully"
    echo "Version: $VERSION"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "$(date +%Y%m%d%H%M%S)")
    
    docker push "${REGISTRY}/${IMAGE_PREFIX}-api:${VERSION}"
    docker push "${REGISTRY}/${IMAGE_PREFIX}-api:latest"
    
    docker push "${REGISTRY}/${IMAGE_PREFIX}-dashboard:${VERSION}"
    docker push "${REGISTRY}/${IMAGE_PREFIX}-dashboard:latest"
    
    docker push "${REGISTRY}/${IMAGE_PREFIX}-worker:${VERSION}"
    docker push "${REGISTRY}/${IMAGE_PREFIX}-worker:latest"
    
    log_success "Images pushed to registry"
}

# Deploy using Docker Compose
deploy_docker_compose() {
    log_info "Deploying using Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export COMPOSE_PROJECT_NAME="mdt-dashboard-${DEPLOY_ENV}"
    export ENVIRONMENT="$DEPLOY_ENV"
    
    # Choose the right compose file
    COMPOSE_FILE="docker-compose.yml"
    if [[ "$DEPLOY_ENV" == "production" ]]; then
        COMPOSE_FILE="docker-compose.prod.yml"
    fi
    
    # Stop existing services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required for Kubernetes deployment"
        exit 1
    fi
    
    cd "$PROJECT_ROOT/k8s"
    
    # Apply configurations in order
    kubectl apply -f 00-namespace-config.yaml
    kubectl apply -f 01-postgres.yaml
    kubectl apply -f 02-redis.yaml
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n mdt-dashboard --timeout=300s
    
    # Apply application manifests
    kubectl apply -f 03-api.yaml
    kubectl apply -f 04-dashboard.yaml
    kubectl apply -f 05-worker.yaml
    kubectl apply -f 06-ingress.yaml
    
    # Wait for deployments
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available deployment --all -n mdt-dashboard --timeout=600s
    
    log_success "Kubernetes deployment completed"
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "API service is healthy"
            break
        fi
        if [[ $i -eq 30 ]]; then
            log_error "API service health check failed"
            return 1
        fi
        sleep 5
    done
    
    # Check Dashboard health
    for i in {1..30}; do
        if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
            log_success "Dashboard service is healthy"
            break
        fi
        if [[ $i -eq 30 ]]; then
            log_error "Dashboard service health check failed"
            return 1
        fi
        sleep 5
    done
    
    log_success "All services are healthy"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    if [[ "$DEPLOY_ENV" == "kubernetes" ]]; then
        kubectl exec -n mdt-dashboard deployment/mdt-api -- poetry run alembic upgrade head
    else
        docker-compose exec api poetry run alembic upgrade head
    fi
    
    log_success "Database migrations completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create Grafana dashboards directory
    mkdir -p config/grafana/dashboards
    
    # Copy monitoring configurations
    if [[ ! -f config/grafana/dashboards/mdt-dashboard.json ]]; then
        log_info "Creating default Grafana dashboard..."
        # You would put your Grafana dashboard JSON here
        echo '{"dashboard": {"title": "MDT Dashboard"}}' > config/grafana/dashboards/mdt-dashboard.json
    fi
    
    log_success "Monitoring setup completed"
}

# Generate SSL certificates (for production)
generate_ssl_certs() {
    if [[ "$DEPLOY_ENV" != "production" ]]; then
        return 0
    fi
    
    log_info "Generating SSL certificates..."
    
    mkdir -p config/ssl
    
    # Generate self-signed certificate (replace with proper certificates in production)
    if [[ ! -f config/ssl/cert.pem ]]; then
        openssl req -x509 -newkey rsa:4096 -keyout config/ssl/key.pem -out config/ssl/cert.pem \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Backup data
backup_data() {
    log_info "Creating data backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    if [[ "$DEPLOY_ENV" == "kubernetes" ]]; then
        # Kubernetes backup
        kubectl exec -n mdt-dashboard deployment/postgres -- pg_dump -U mdt_user mdt_dashboard > "$BACKUP_DIR/database.sql"
    else
        # Docker Compose backup
        docker-compose exec -T postgres pg_dump -U mdt_user mdt_dashboard > "$BACKUP_DIR/database.sql"
    fi
    
    log_success "Data backup created: $BACKUP_DIR"
}

# Main deployment function
main() {
    log_info "Starting MDT Dashboard deployment to $DEPLOY_ENV environment"
    
    case "$DEPLOY_ENV" in
        "local"|"development")
            check_prerequisites
            build_images
            deploy_docker_compose
            run_migrations
            ;;
        "staging")
            check_prerequisites
            build_images
            push_images
            deploy_docker_compose
            run_migrations
            setup_monitoring
            ;;
        "production")
            check_prerequisites
            backup_data
            build_images
            push_images
            generate_ssl_certs
            deploy_docker_compose
            run_migrations
            setup_monitoring
            ;;
        "kubernetes"|"k8s")
            check_prerequisites
            build_images
            push_images
            deploy_kubernetes
            run_migrations
            setup_monitoring
            ;;
        *)
            log_error "Unknown environment: $DEPLOY_ENV"
            echo "Usage: $0 [local|staging|production|kubernetes]"
            exit 1
            ;;
    esac
    
    log_success "MDT Dashboard deployment to $DEPLOY_ENV completed successfully!"
    
    # Print access information
    echo ""
    echo "🎉 Deployment completed! Access your services:"
    echo "📊 Dashboard: http://localhost:8501"
    echo "🔗 API: http://localhost:8000"
    echo "📈 API Docs: http://localhost:8000/docs"
    echo "🌸 Flower (Celery): http://localhost:5555"
    echo "📊 Grafana: http://localhost:3000 (admin/admin_password)"
    echo "🔍 Prometheus: http://localhost:9090"
    echo ""
}

# Handle script arguments
if [[ $# -gt 1 ]]; then
    log_error "Too many arguments"
    echo "Usage: $0 [local|staging|production|kubernetes]"
    exit 1
fi

# Run main function
main
