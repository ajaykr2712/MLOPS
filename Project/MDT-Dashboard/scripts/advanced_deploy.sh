#!/bin/bash

# Advanced Deployment Script for MDT Dashboard
# This script provides comprehensive deployment capabilities with:
# - Multi-environment support
# - Blue-green deployment strategy
# - Health checks and rollback capabilities
# - Monitoring integration
# - Security scanning

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration
DEFAULT_ENVIRONMENT="staging"
DEFAULT_DEPLOYMENT_STRATEGY="rolling"
DEFAULT_NAMESPACE="mdt-dashboard"
DEFAULT_TIMEOUT="600"
DEFAULT_HEALTH_CHECK_RETRIES="10"

# Color codes for output
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

# Usage information
usage() {
    cat << EOF
Advanced Deployment Script for MDT Dashboard

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENVIRONMENT   Target environment (dev|staging|prod) [default: staging]
    -s, --strategy STRATEGY         Deployment strategy (rolling|blue-green|canary) [default: rolling]
    -n, --namespace NAMESPACE       Kubernetes namespace [default: mdt-dashboard]
    -t, --timeout TIMEOUT          Deployment timeout in seconds [default: 600]
    -r, --retries RETRIES          Health check retries [default: 10]
    -v, --version VERSION          Application version to deploy
    -c, --config CONFIG_FILE       Custom configuration file
    --dry-run                      Perform a dry run without actual deployment
    --skip-tests                   Skip pre-deployment tests
    --skip-security                Skip security scanning
    --force                        Force deployment even if tests fail
    --rollback                     Rollback to previous version
    --scale REPLICAS               Scale to specific number of replicas
    -h, --help                     Show this help message

EXAMPLES:
    # Basic deployment to staging
    $0 -e staging

    # Blue-green deployment to production
    $0 -e prod -s blue-green

    # Canary deployment with custom config
    $0 -e prod -s canary -c configs/prod-canary.yaml

    # Rollback production deployment
    $0 -e prod --rollback

    # Scale production deployment
    $0 -e prod --scale 10

EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT="$DEFAULT_ENVIRONMENT"
    DEPLOYMENT_STRATEGY="$DEFAULT_DEPLOYMENT_STRATEGY"
    NAMESPACE="$DEFAULT_NAMESPACE"
    TIMEOUT="$DEFAULT_TIMEOUT"
    HEALTH_CHECK_RETRIES="$DEFAULT_HEALTH_CHECK_RETRIES"
    VERSION=""
    CONFIG_FILE=""
    DRY_RUN=false
    SKIP_TESTS=false
    SKIP_SECURITY=false
    FORCE=false
    ROLLBACK=false
    SCALE_REPLICAS=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--strategy)
                DEPLOYMENT_STRATEGY="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -r|--retries)
                HEALTH_CHECK_RETRIES="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-security)
                SKIP_SECURITY=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --scale)
                SCALE_REPLICAS="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|staging|prod)
            log_info "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Must be one of: dev, staging, prod"
            exit 1
            ;;
    esac
}

# Validate deployment strategy
validate_strategy() {
    case $DEPLOYMENT_STRATEGY in
        rolling|blue-green|canary)
            log_info "Using deployment strategy: $DEPLOYMENT_STRATEGY"
            ;;
        *)
            log_error "Invalid deployment strategy: $DEPLOYMENT_STRATEGY. Must be one of: rolling, blue-green, canary"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check required tools
    local required_tools=("kubectl" "docker" "helm" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done

    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist. Creating..."
        kubectl create namespace "$NAMESPACE"
    fi

    # Verify RBAC permissions
    if ! kubectl auth can-i create deployments --namespace="$NAMESPACE" &> /dev/null; then
        log_error "Insufficient RBAC permissions for namespace $NAMESPACE"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Get current version
get_current_version() {
    local current_version=""
    if kubectl get deployment mdt-api -n "$NAMESPACE" &> /dev/null; then
        current_version=$(kubectl get deployment mdt-api -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "unknown")
    fi
    echo "$current_version"
}

# Determine version to deploy
determine_version() {
    if [[ -n "$VERSION" ]]; then
        log_info "Using specified version: $VERSION"
        return
    fi

    # Auto-determine version from Git
    if command -v git &> /dev/null && [[ -d "$PROJECT_ROOT/.git" ]]; then
        VERSION=$(git describe --tags --always --dirty 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "latest")
        log_info "Auto-determined version from Git: $VERSION"
    else
        VERSION="latest"
        log_warning "Could not determine version, using: $VERSION"
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build Docker images with version: $VERSION"
        return
    fi

    # Build API image
    log_info "Building API image..."
    docker build -t "mdt-dashboard/api:$VERSION" \
        -f "$PROJECT_ROOT/docker/Dockerfile.api" \
        "$PROJECT_ROOT"

    # Build dashboard image
    log_info "Building dashboard image..."
    docker build -t "mdt-dashboard/dashboard:$VERSION" \
        -f "$PROJECT_ROOT/docker/Dockerfile.dashboard" \
        "$PROJECT_ROOT"

    # Build worker image
    log_info "Building worker image..."
    docker build -t "mdt-dashboard/worker:$VERSION" \
        -f "$PROJECT_ROOT/docker/Dockerfile.worker" \
        "$PROJECT_ROOT"

    log_success "Docker images built successfully"
}

# Security scanning
security_scan() {
    if [[ "$SKIP_SECURITY" == "true" ]]; then
        log_warning "Skipping security scanning"
        return
    fi

    log_info "Running security scans..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run security scans"
        return
    fi

    # Scan Docker images for vulnerabilities
    local images=("mdt-dashboard/api:$VERSION" "mdt-dashboard/dashboard:$VERSION" "mdt-dashboard/worker:$VERSION")
    
    for image in "${images[@]}"; do
        log_info "Scanning image: $image"
        
        # Using Trivy for vulnerability scanning (if available)
        if command -v trivy &> /dev/null; then
            if ! trivy image --exit-code 1 --severity HIGH,CRITICAL "$image"; then
                if [[ "$FORCE" != "true" ]]; then
                    log_error "Security vulnerabilities found in $image. Use --force to deploy anyway."
                    exit 1
                else
                    log_warning "Security vulnerabilities found but deployment forced"
                fi
            fi
        else
            log_warning "Trivy not available, skipping vulnerability scan for $image"
        fi
    done

    log_success "Security scans completed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests"
        return
    fi

    log_info "Running pre-deployment tests..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run tests"
        return
    fi

    # Unit tests
    log_info "Running unit tests..."
    if ! python -m pytest "$PROJECT_ROOT/tests/unit" -v; then
        if [[ "$FORCE" != "true" ]]; then
            log_error "Unit tests failed. Use --force to deploy anyway."
            exit 1
        else
            log_warning "Unit tests failed but deployment forced"
        fi
    fi

    # Integration tests
    log_info "Running integration tests..."
    if ! python -m pytest "$PROJECT_ROOT/tests/integration" -v; then
        if [[ "$FORCE" != "true" ]]; then
            log_error "Integration tests failed. Use --force to deploy anyway."
            exit 1
        else
            log_warning "Integration tests failed but deployment forced"
        fi
    fi

    log_success "Tests passed"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would push images to registry"
        return
    fi

    # Get registry URL from environment or use default
    local registry="${DOCKER_REGISTRY:-localhost:5000}"
    
    local images=("api" "dashboard" "worker")
    for image in "${images[@]}"; do
        local local_tag="mdt-dashboard/$image:$VERSION"
        local remote_tag="$registry/mdt-dashboard/$image:$VERSION"
        
        log_info "Tagging and pushing $image..."
        docker tag "$local_tag" "$remote_tag"
        docker push "$remote_tag"
    done

    log_success "Images pushed to registry"
}

# Generate Kubernetes manifests
generate_manifests() {
    log_info "Generating Kubernetes manifests..."

    local manifests_dir="$PROJECT_ROOT/k8s/generated"
    mkdir -p "$manifests_dir"

    # Use Helm to generate manifests
    local helm_values_file="$PROJECT_ROOT/k8s/values-$ENVIRONMENT.yaml"
    if [[ -n "$CONFIG_FILE" ]]; then
        helm_values_file="$CONFIG_FILE"
    fi

    helm template mdt-dashboard "$PROJECT_ROOT/k8s/chart" \
        --namespace "$NAMESPACE" \
        --values "$helm_values_file" \
        --set image.tag="$VERSION" \
        --set deployment.strategy="$DEPLOYMENT_STRATEGY" \
        --output-dir "$manifests_dir"

    log_success "Kubernetes manifests generated"
}

# Rolling deployment
deploy_rolling() {
    log_info "Performing rolling deployment..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform rolling deployment"
        return
    fi

    # Apply manifests
    kubectl apply -f "$PROJECT_ROOT/k8s/generated/mdt-dashboard/templates/" --namespace="$NAMESPACE"

    # Wait for rollout to complete
    local deployments=("mdt-api" "mdt-dashboard" "mdt-worker")
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment rollout to complete..."
        if ! kubectl rollout status deployment/"$deployment" --namespace="$NAMESPACE" --timeout="${TIMEOUT}s"; then
            log_error "Rollout failed for $deployment"
            return 1
        fi
    done

    log_success "Rolling deployment completed"
}

# Blue-green deployment
deploy_blue_green() {
    log_info "Performing blue-green deployment..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform blue-green deployment"
        return
    fi

    # Determine current and new colors
    local current_color=$(kubectl get service mdt-api-service -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue")
    local new_color
    if [[ "$current_color" == "blue" ]]; then
        new_color="green"
    else
        new_color="blue"
    fi

    log_info "Current color: $current_color, deploying to: $new_color"

    # Deploy to new color
    kubectl apply -f "$PROJECT_ROOT/k8s/generated/mdt-dashboard/templates/" --namespace="$NAMESPACE"
    kubectl patch deployment mdt-api -n "$NAMESPACE" -p '{"spec":{"selector":{"matchLabels":{"color":"'$new_color'"}},"template":{"metadata":{"labels":{"color":"'$new_color'"}}}}}'

    # Wait for new deployment to be ready
    if ! kubectl rollout status deployment/mdt-api --namespace="$NAMESPACE" --timeout="${TIMEOUT}s"; then
        log_error "Blue-green deployment failed"
        return 1
    fi

    # Health check new deployment
    if ! health_check "$new_color"; then
        log_error "Health check failed for new deployment"
        return 1
    fi

    # Switch traffic to new color
    log_info "Switching traffic to $new_color..."
    kubectl patch service mdt-api-service -n "$NAMESPACE" -p '{"spec":{"selector":{"color":"'$new_color'"}}}'

    # Wait for traffic switch to complete
    sleep 10

    # Final health check
    if ! health_check; then
        log_error "Health check failed after traffic switch"
        # Rollback traffic
        kubectl patch service mdt-api-service -n "$NAMESPACE" -p '{"spec":{"selector":{"color":"'$current_color'"}}}'
        return 1
    fi

    log_success "Blue-green deployment completed"
}

# Canary deployment
deploy_canary() {
    log_info "Performing canary deployment..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform canary deployment"
        return
    fi

    # Deploy canary version with limited traffic
    local canary_replicas=1
    local total_replicas=$(kubectl get deployment mdt-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "3")
    
    # Create canary deployment
    kubectl apply -f "$PROJECT_ROOT/k8s/generated/mdt-dashboard/templates/" --namespace="$NAMESPACE"
    kubectl patch deployment mdt-api -n "$NAMESPACE" -p '{"metadata":{"name":"mdt-api-canary"},"spec":{"replicas":'$canary_replicas'}}'

    # Wait for canary to be ready
    if ! kubectl rollout status deployment/mdt-api-canary --namespace="$NAMESPACE" --timeout="${TIMEOUT}s"; then
        log_error "Canary deployment failed"
        return 1
    fi

    # Monitor canary for specified duration
    local canary_duration=300  # 5 minutes
    log_info "Monitoring canary for $canary_duration seconds..."
    
    for ((i=0; i<canary_duration; i+=30)); do
        if ! health_check "canary"; then
            log_error "Canary health check failed"
            # Cleanup canary
            kubectl delete deployment mdt-api-canary -n "$NAMESPACE" --ignore-not-found
            return 1
        fi
        sleep 30
    done

    # Promote canary to full deployment
    log_info "Promoting canary to full deployment..."
    kubectl patch deployment mdt-api -n "$NAMESPACE" -p '{"spec":{"replicas":'$total_replicas'}}'
    kubectl delete deployment mdt-api-canary -n "$NAMESPACE" --ignore-not-found

    log_success "Canary deployment completed"
}

# Health check
health_check() {
    local target="${1:-}"
    local retries="${2:-$HEALTH_CHECK_RETRIES}"
    
    log_info "Running health checks (retries: $retries)..."

    local service_name="mdt-api-service"
    if [[ "$target" == "canary" ]]; then
        service_name="mdt-api-canary-service"
    fi

    for ((i=1; i<=retries; i++)); do
        log_info "Health check attempt $i/$retries..."
        
        # Check if pods are ready
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=mdt-api --field-selector=status.phase=Running -o json | jq '.items | length')
        if [[ "$ready_pods" -eq 0 ]]; then
            log_warning "No ready pods found"
            sleep 10
            continue
        fi

        # Check API endpoint
        local api_url="http://$(kubectl get service $service_name -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):$(kubectl get service $service_name -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}')"
        
        if curl -f -s "$api_url/health" > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi

        if [[ $i -lt $retries ]]; then
            log_warning "Health check failed, retrying in 10 seconds..."
            sleep 10
        fi
    done

    log_error "Health check failed after $retries attempts"
    return 1
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback deployment"
        return
    fi

    # Get previous revision
    local current_revision=$(kubectl rollout history deployment/mdt-api -n "$NAMESPACE" --revision=0 | tail -n 1 | awk '{print $1}')
    local previous_revision=$((current_revision - 1))

    if [[ $previous_revision -lt 1 ]]; then
        log_error "No previous revision found to rollback to"
        exit 1
    fi

    log_info "Rolling back to revision $previous_revision..."

    # Rollback deployments
    local deployments=("mdt-api" "mdt-dashboard" "mdt-worker")
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            kubectl rollout undo deployment/"$deployment" --namespace="$NAMESPACE" --to-revision="$previous_revision"
            
            if ! kubectl rollout status deployment/"$deployment" --namespace="$NAMESPACE" --timeout="${TIMEOUT}s"; then
                log_error "Rollback failed for $deployment"
                exit 1
            fi
        fi
    done

    # Health check after rollback
    if ! health_check; then
        log_error "Health check failed after rollback"
        exit 1
    fi

    log_success "Rollback completed successfully"
}

# Scale deployment
scale_deployment() {
    local replicas="$1"
    
    log_info "Scaling deployment to $replicas replicas..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would scale deployment to $replicas replicas"
        return
    fi

    # Scale deployments
    local deployments=("mdt-api" "mdt-dashboard" "mdt-worker")
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            kubectl scale deployment "$deployment" --replicas="$replicas" --namespace="$NAMESPACE"
            
            if ! kubectl rollout status deployment/"$deployment" --namespace="$NAMESPACE" --timeout="${TIMEOUT}s"; then
                log_error "Scaling failed for $deployment"
                exit 1
            fi
        fi
    done

    log_success "Scaling completed successfully"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would setup monitoring"
        return
    fi

    # Deploy monitoring manifests if they exist
    if [[ -d "$PROJECT_ROOT/k8s/monitoring" ]]; then
        kubectl apply -f "$PROJECT_ROOT/k8s/monitoring/" --namespace="$NAMESPACE"
        log_success "Monitoring setup completed"
    else
        log_warning "No monitoring manifests found"
    fi
}

# Cleanup old resources
cleanup() {
    log_info "Cleaning up old resources..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would cleanup old resources"
        return
    fi

    # Remove old replica sets
    kubectl delete replicasets -n "$NAMESPACE" --field-selector='status.replicas=0'

    # Clean up old images (keep last 3 versions)
    # This would require additional logic to determine which images to clean

    log_success "Cleanup completed"
}

# Main deployment orchestration
deploy() {
    local current_version
    current_version=$(get_current_version)
    
    log_info "Starting deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Strategy: $DEPLOYMENT_STRATEGY"
    log_info "Current version: ${current_version:-none}"
    log_info "Target version: $VERSION"
    log_info "Namespace: $NAMESPACE"

    # Build and push images
    build_images
    security_scan
    push_images

    # Run tests
    run_tests

    # Generate manifests
    generate_manifests

    # Deploy based on strategy
    case $DEPLOYMENT_STRATEGY in
        rolling)
            deploy_rolling
            ;;
        blue-green)
            deploy_blue_green
            ;;
        canary)
            deploy_canary
            ;;
    esac

    # Setup monitoring
    setup_monitoring

    # Final health check
    if ! health_check; then
        log_error "Final health check failed"
        exit 1
    fi

    # Cleanup
    cleanup

    log_success "Deployment completed successfully!"
    log_info "Application version $VERSION is now running in $ENVIRONMENT"
}

# Main execution
main() {
    parse_args "$@"
    validate_environment
    validate_strategy
    check_prerequisites
    determine_version

    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
    elif [[ -n "$SCALE_REPLICAS" ]]; then
        scale_deployment "$SCALE_REPLICAS"
    else
        deploy
    fi
}

# Execute main function with all arguments
main "$@"
