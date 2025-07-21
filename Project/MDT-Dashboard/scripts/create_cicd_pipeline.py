"""
CI/CD Pipeline for ML - GitHub Actions Workflow
Comprehensive MLOps pipeline with testing, validation, and deployment
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any

def create_ml_cicd_pipeline():
    """Create comprehensive CI/CD pipeline for ML projects."""
    
    # Main CI/CD workflow
    main_workflow = {
        'name': 'ML Model CI/CD Pipeline',
        'on': {
            'push': {
                'branches': ['main', 'develop'],
                'paths': [
                    'src/**',
                    'models/**',
                    'data/**',
                    'configs/**',
                    'tests/**',
                    'requirements.txt',
                    'pyproject.toml'
                ]
            },
            'pull_request': {
                'branches': ['main']
            },
            'schedule': [
                {'cron': '0 2 * * *'}  # Daily at 2 AM
            ],
            'workflow_dispatch': {
                'inputs': {
                    'model_version': {
                        'description': 'Model version to deploy',
                        'required': False,
                        'default': 'latest'
                    },
                    'environment': {
                        'description': 'Target environment',
                        'required': True,
                        'default': 'staging',
                        'type': 'choice',
                        'options': ['staging', 'production']
                    }
                }
            }
        },
        'env': {
            'PYTHON_VERSION': '3.9',
            'POETRY_VERSION': '1.6.1',
            'MLFLOW_TRACKING_URI': '${{ secrets.MLFLOW_TRACKING_URI }}',
            'AWS_DEFAULT_REGION': 'us-east-1'
        },
        'jobs': {
            'lint-and-test': {
                'runs-on': 'ubuntu-latest',
                'strategy': {
                    'matrix': {
                        'python-version': ['3.9', '3.10', '3.11']
                    }
                },
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4',
                        'with': {
                            'fetch-depth': 0  # Full history for better analysis
                        }
                    },
                    {
                        'name': 'Set up Python',
                        'uses': 'actions/setup-python@v4',
                        'with': {
                            'python-version': '${{ matrix.python-version }}'
                        }
                    },
                    {
                        'name': 'Install Poetry',
                        'uses': 'snok/install-poetry@v1',
                        'with': {
                            'version': '${{ env.POETRY_VERSION }}',
                            'virtualenvs-create': True,
                            'virtualenvs-in-project': True
                        }
                    },
                    {
                        'name': 'Load cached venv',
                        'id': 'cached-poetry-dependencies',
                        'uses': 'actions/cache@v3',
                        'with': {
                            'path': '.venv',
                            'key': "venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}"
                        }
                    },
                    {
                        'name': 'Install dependencies',
                        'if': 'steps.cached-poetry-dependencies.outputs.cache-hit != \'true\'',
                        'run': 'poetry install --no-interaction --no-root'
                    },
                    {
                        'name': 'Install project',
                        'run': 'poetry install --no-interaction'
                    },
                    {
                        'name': 'Lint with flake8',
                        'run': '''
                            poetry run flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
                            poetry run flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
                        '''
                    },
                    {
                        'name': 'Type check with mypy',
                        'run': 'poetry run mypy src/'
                    },
                    {
                        'name': 'Format check with black',
                        'run': 'poetry run black --check src/ tests/'
                    },
                    {
                        'name': 'Import sort check',
                        'run': 'poetry run isort --check-only src/ tests/'
                    },
                    {
                        'name': 'Security check with bandit',
                        'run': 'poetry run bandit -r src/'
                    },
                    {
                        'name': 'Run unit tests',
                        'run': '''
                            poetry run pytest tests/unit/ \
                              --cov=src \
                              --cov-report=xml \
                              --cov-report=html \
                              --junitxml=test-results.xml \
                              -v
                        '''
                    },
                    {
                        'name': 'Upload coverage to Codecov',
                        'uses': 'codecov/codecov-action@v3',
                        'with': {
                            'file': './coverage.xml',
                            'flags': 'unittests',
                            'name': 'codecov-umbrella',
                            'fail_ci_if_error': True
                        }
                    }
                ]
            },
            
            'data-validation': {
                'runs-on': 'ubuntu-latest',
                'needs': 'lint-and-test',
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Set up Python',
                        'uses': 'actions/setup-python@v4',
                        'with': {
                            'python-version': '${{ env.PYTHON_VERSION }}'
                        }
                    },
                    {
                        'name': 'Install dependencies',
                        'run': '''
                            pip install poetry
                            poetry install
                        '''
                    },
                    {
                        'name': 'Download latest data',
                        'run': 'poetry run python scripts/download_data.py'
                    },
                    {
                        'name': 'Validate data schema',
                        'run': 'poetry run python scripts/validate_data.py'
                    },
                    {
                        'name': 'Data quality checks',
                        'run': 'poetry run great_expectations checkpoint run data_quality_checkpoint'
                    },
                    {
                        'name': 'Data drift detection',
                        'run': '''
                            poetry run python scripts/detect_data_drift.py \
                              --reference-data data/reference/ \
                              --current-data data/current/ \
                              --output-report drift_report.html
                        '''
                    },
                    {
                        'name': 'Upload drift report',
                        'uses': 'actions/upload-artifact@v3',
                        'with': {
                            'name': 'drift-report',
                            'path': 'drift_report.html'
                        }
                    }
                ]
            },
            
            'model-training': {
                'runs-on': 'ubuntu-latest',
                'needs': ['lint-and-test', 'data-validation'],
                'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Set up Python',
                        'uses': 'actions/setup-python@v4',
                        'with': {
                            'python-version': '${{ env.PYTHON_VERSION }}'
                        }
                    },
                    {
                        'name': 'Configure AWS credentials',
                        'uses': 'aws-actions/configure-aws-credentials@v2',
                        'with': {
                            'aws-access-key-id': '${{ secrets.AWS_ACCESS_KEY_ID }}',
                            'aws-secret-access-key': '${{ secrets.AWS_SECRET_ACCESS_KEY }}',
                            'aws-region': '${{ env.AWS_DEFAULT_REGION }}'
                        }
                    },
                    {
                        'name': 'Install dependencies',
                        'run': '''
                            pip install poetry
                            poetry install
                        '''
                    },
                    {
                        'name': 'Set up MLflow tracking',
                        'run': '''
                            echo "MLFLOW_TRACKING_URI=${{ env.MLFLOW_TRACKING_URI }}" >> $GITHUB_ENV
                            echo "MLFLOW_EXPERIMENT_NAME=github-actions-${{ github.run_id }}" >> $GITHUB_ENV
                        '''
                    },
                    {
                        'name': 'Train model',
                        'run': '''
                            poetry run python src/mdt_dashboard/train.py \
                              --config configs/training_config.yaml \
                              --data-path data/processed/ \
                              --output-path models/ \
                              --experiment-name ${{ env.MLFLOW_EXPERIMENT_NAME }}
                        '''
                    },
                    {
                        'name': 'Evaluate model',
                        'run': '''
                            poetry run python src/mdt_dashboard/evaluate.py \
                              --model-path models/latest/ \
                              --test-data data/test/ \
                              --output-path evaluation_results/
                        '''
                    },
                    {
                        'name': 'Model validation',
                        'run': '''
                            poetry run python scripts/validate_model.py \
                              --model-path models/latest/ \
                              --validation-data data/validation/ \
                              --baseline-metrics baseline_metrics.json
                        '''
                    },
                    {
                        'name': 'Upload model artifacts',
                        'uses': 'actions/upload-artifact@v3',
                        'with': {
                            'name': 'model-artifacts',
                            'path': 'models/'
                        }
                    },
                    {
                        'name': 'Upload evaluation results',
                        'uses': 'actions/upload-artifact@v3',
                        'with': {
                            'name': 'evaluation-results',
                            'path': 'evaluation_results/'
                        }
                    }
                ]
            },
            
            'integration-tests': {
                'runs-on': 'ubuntu-latest',
                'needs': 'model-training',
                'services': {
                    'postgres': {
                        'image': 'postgres:13',
                        'env': {
                            'POSTGRES_PASSWORD': 'postgres',
                            'POSTGRES_DB': 'test_db'
                        },
                        'ports': ['5432:5432'],
                        'options': '--health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5'
                    },
                    'redis': {
                        'image': 'redis:7',
                        'ports': ['6379:6379'],
                        'options': '--health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 5'
                    }
                },
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Set up Python',
                        'uses': 'actions/setup-python@v4',
                        'with': {
                            'python-version': '${{ env.PYTHON_VERSION }}'
                        }
                    },
                    {
                        'name': 'Install dependencies',
                        'run': '''
                            pip install poetry
                            poetry install
                        '''
                    },
                    {
                        'name': 'Download model artifacts',
                        'uses': 'actions/download-artifact@v3',
                        'with': {
                            'name': 'model-artifacts',
                            'path': 'models/'
                        }
                    },
                    {
                        'name': 'Start application services',
                        'run': '''
                            poetry run python -m src.mdt_dashboard.api.main &
                            sleep 30  # Wait for services to start
                        '''
                        'env': {
                            'DATABASE_URL': 'postgresql://postgres:postgres@localhost:5432/test_db',
                            'REDIS_URL': 'redis://localhost:6379'
                        }
                    },
                    {
                        'name': 'Run integration tests',
                        'run': '''
                            poetry run pytest tests/integration/ \
                              --junitxml=integration-test-results.xml \
                              -v
                        '''
                    },
                    {
                        'name': 'Run end-to-end tests',
                        'run': '''
                            poetry run pytest tests/e2e/ \
                              --junitxml=e2e-test-results.xml \
                              -v
                        '''
                    },
                    {
                        'name': 'Performance tests',
                        'run': '''
                            poetry run python tests/performance/api_load_test.py
                            poetry run python tests/performance/model_inference_test.py
                        '''
                    }
                ]
            },
            
            'security-scan': {
                'runs-on': 'ubuntu-latest',
                'needs': 'lint-and-test',
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Run Trivy vulnerability scanner',
                        'uses': 'aquasecurity/trivy-action@master',
                        'with': {
                            'scan-type': 'fs',
                            'scan-ref': '.',
                            'format': 'sarif',
                            'output': 'trivy-results.sarif'
                        }
                    },
                    {
                        'name': 'Upload Trivy scan results',
                        'uses': 'github/codeql-action/upload-sarif@v2',
                        'with': {
                            'sarif_file': 'trivy-results.sarif'
                        }
                    },
                    {
                        'name': 'Dependency vulnerability check',
                        'run': '''
                            pip install safety
                            safety check --json --output safety-report.json || true
                        '''
                    },
                    {
                        'name': 'Upload security reports',
                        'uses': 'actions/upload-artifact@v3',
                        'with': {
                            'name': 'security-reports',
                            'path': '*-report.json'
                        }
                    }
                ]
            },
            
            'docker-build': {
                'runs-on': 'ubuntu-latest',
                'needs': ['integration-tests', 'security-scan'],
                'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Set up Docker Buildx',
                        'uses': 'docker/setup-buildx-action@v2'
                    },
                    {
                        'name': 'Login to DockerHub',
                        'uses': 'docker/login-action@v2',
                        'with': {
                            'username': '${{ secrets.DOCKERHUB_USERNAME }}',
                            'password': '${{ secrets.DOCKERHUB_TOKEN }}'
                        }
                    },
                    {
                        'name': 'Download model artifacts',
                        'uses': 'actions/download-artifact@v3',
                        'with': {
                            'name': 'model-artifacts',
                            'path': 'models/'
                        }
                    },
                    {
                        'name': 'Build and push API image',
                        'uses': 'docker/build-push-action@v4',
                        'with': {
                            'context': '.',
                            'file': './docker/Dockerfile.api',
                            'push': True,
                            'tags': '''
                                ${{ secrets.DOCKERHUB_USERNAME }}/mdt-api:latest
                                ${{ secrets.DOCKERHUB_USERNAME }}/mdt-api:${{ github.sha }}
                            ''',
                            'cache-from': 'type=gha',
                            'cache-to': 'type=gha,mode=max'
                        }
                    },
                    {
                        'name': 'Build and push Dashboard image',
                        'uses': 'docker/build-push-action@v4',
                        'with': {
                            'context': '.',
                            'file': './docker/Dockerfile.dashboard',
                            'push': True,
                            'tags': '''
                                ${{ secrets.DOCKERHUB_USERNAME }}/mdt-dashboard:latest
                                ${{ secrets.DOCKERHUB_USERNAME }}/mdt-dashboard:${{ github.sha }}
                            '''
                        }
                    },
                    {
                        'name': 'Build and push Worker image',
                        'uses': 'docker/build-push-action@v4',
                        'with': {
                            'context': '.',
                            'file': './docker/Dockerfile.worker',
                            'push': True,
                            'tags': '''
                                ${{ secrets.DOCKERHUB_USERNAME }}/mdt-worker:latest
                                ${{ secrets.DOCKERHUB_USERNAME }}/mdt-worker:${{ github.sha }}
                            '''
                        }
                    }
                ]
            },
            
            'deploy-staging': {
                'runs-on': 'ubuntu-latest',
                'needs': 'docker-build',
                'environment': 'staging',
                'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Configure AWS credentials',
                        'uses': 'aws-actions/configure-aws-credentials@v2',
                        'with': {
                            'aws-access-key-id': '${{ secrets.AWS_ACCESS_KEY_ID }}',
                            'aws-secret-access-key': '${{ secrets.AWS_SECRET_ACCESS_KEY }}',
                            'aws-region': '${{ env.AWS_DEFAULT_REGION }}'
                        }
                    },
                    {
                        'name': 'Install kubectl',
                        'uses': 'azure/setup-kubectl@v3',
                        'with': {
                            'version': 'v1.28.0'
                        }
                    },
                    {
                        'name': 'Update kubeconfig',
                        'run': 'aws eks update-kubeconfig --name mdt-staging-cluster'
                    },
                    {
                        'name': 'Deploy to staging',
                        'run': '''
                            # Update image tags in Kubernetes manifests
                            sed -i "s|{{IMAGE_TAG}}|${{ github.sha }}|g" k8s/*.yaml
                            
                            # Apply Kubernetes manifests
                            kubectl apply -f k8s/ -n staging
                            
                            # Wait for deployment to complete
                            kubectl rollout status deployment/mdt-api -n staging --timeout=300s
                            kubectl rollout status deployment/mdt-dashboard -n staging --timeout=300s
                            kubectl rollout status deployment/mdt-worker -n staging --timeout=300s
                        '''
                    },
                    {
                        'name': 'Run smoke tests',
                        'run': '''
                            # Get staging URL
                            STAGING_URL=$(kubectl get service mdt-api -n staging -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
                            
                            # Run basic health checks
                            curl -f http://$STAGING_URL/health
                            
                            # Run smoke tests
                            poetry run python tests/smoke/test_staging.py --url http://$STAGING_URL
                        '''
                    }
                ]
            },
            
            'deploy-production': {
                'runs-on': 'ubuntu-latest',
                'needs': 'deploy-staging',
                'environment': 'production',
                'if': "github.event.inputs.environment == 'production'",
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Configure AWS credentials',
                        'uses': 'aws-actions/configure-aws-credentials@v2',
                        'with': {
                            'aws-access-key-id': '${{ secrets.AWS_ACCESS_KEY_ID_PROD }}',
                            'aws-secret-access-key': '${{ secrets.AWS_SECRET_ACCESS_KEY_PROD }}',
                            'aws-region': '${{ env.AWS_DEFAULT_REGION }}'
                        }
                    },
                    {
                        'name': 'Install kubectl',
                        'uses': 'azure/setup-kubectl@v3'
                    },
                    {
                        'name': 'Update kubeconfig',
                        'run': 'aws eks update-kubeconfig --name mdt-production-cluster'
                    },
                    {
                        'name': 'Blue-Green deployment',
                        'run': '''
                            # Deploy to green environment
                            sed -i "s|{{IMAGE_TAG}}|${{ github.sha }}|g" k8s/*.yaml
                            sed -i "s|{{ENV}}|green|g" k8s/*.yaml
                            
                            kubectl apply -f k8s/ -n production-green
                            
                            # Wait for green deployment
                            kubectl rollout status deployment/mdt-api -n production-green --timeout=600s
                            
                            # Run production smoke tests
                            GREEN_URL=$(kubectl get service mdt-api -n production-green -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
                            poetry run python tests/smoke/test_production.py --url http://$GREEN_URL
                            
                            # Switch traffic to green
                            kubectl patch service mdt-api-main -n production -p '{"spec":{"selector":{"version":"green"}}}'
                            
                            # Clean up blue environment after successful switch
                            kubectl delete all -l version=blue -n production
                        '''
                    }
                ]
            },
            
            'monitoring-setup': {
                'runs-on': 'ubuntu-latest',
                'needs': 'deploy-production',
                'if': "success() && github.event.inputs.environment == 'production'",
                'steps': [
                    {
                        'name': 'Setup monitoring alerts',
                        'run': '''
                            # Create Grafana dashboards
                            curl -X POST -H "Content-Type: application/json" \
                              -d @monitoring/grafana-dashboard.json \
                              "http://${{ secrets.GRAFANA_URL }}/api/dashboards/db" \
                              -u "${{ secrets.GRAFANA_USER }}:${{ secrets.GRAFANA_PASSWORD }}"
                            
                            # Setup Prometheus alerts
                            kubectl apply -f monitoring/prometheus-rules.yaml
                        '''
                    },
                    {
                        'name': 'Setup model monitoring',
                        'run': '''
                            # Configure Evidently monitoring
                            poetry run python scripts/setup_monitoring.py \
                              --model-name production-model \
                              --monitoring-endpoint ${{ secrets.MONITORING_ENDPOINT }}
                        '''
                    }
                ]
            }
        }
    }
    
    return main_workflow

def create_model_performance_workflow():
    """Create workflow for continuous model performance monitoring."""
    
    performance_workflow = {
        'name': 'Model Performance Monitoring',
        'on': {
            'schedule': [
                {'cron': '0 */6 * * *'}  # Every 6 hours
            ],
            'workflow_dispatch': {}
        },
        'jobs': {
            'performance-check': {
                'runs-on': 'ubuntu-latest',
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Set up Python',
                        'uses': 'actions/setup-python@v4',
                        'with': {
                            'python-version': '3.9'
                        }
                    },
                    {
                        'name': 'Install dependencies',
                        'run': '''
                            pip install poetry
                            poetry install
                        '''
                    },
                    {
                        'name': 'Check model performance',
                        'run': '''
                            poetry run python scripts/check_model_performance.py \
                              --model-endpoint ${{ secrets.PRODUCTION_MODEL_ENDPOINT }} \
                              --threshold-file thresholds.yaml \
                              --output-report performance_report.json
                        '''
                    },
                    {
                        'name': 'Check data drift',
                        'run': '''
                            poetry run python scripts/check_data_drift.py \
                              --production-data ${{ secrets.PRODUCTION_DATA_SOURCE }} \
                              --reference-data data/reference/ \
                              --output-report drift_report.json
                        '''
                    },
                    {
                        'name': 'Alert on performance degradation',
                        'if': "failure()",
                        'uses': 'actions/github-script@v6',
                        'with': {
                            'script': '''
                                github.rest.issues.create({
                                  owner: context.repo.owner,
                                  repo: context.repo.repo,
                                  title: 'Model Performance Alert - ' + new Date().toISOString(),
                                  body: 'Model performance has degraded below threshold. Please check the performance report.',
                                  labels: ['model-performance', 'alert', 'high-priority']
                                })
                            '''
                        }
                    },
                    {
                        'name': 'Send Slack notification',
                        'if': "failure()",
                        'uses': '8398a7/action-slack@v3',
                        'with': {
                            'status': 'failure',
                            'text': 'Model performance monitoring failed. Check GitHub Actions for details.',
                            'webhook_url': '${{ secrets.SLACK_WEBHOOK }}'
                        }
                    }
                ]
            }
        }
    }
    
    return performance_workflow

def create_security_workflow():
    """Create security-focused workflow."""
    
    security_workflow = {
        'name': 'Security Scan',
        'on': {
            'schedule': [
                {'cron': '0 0 * * 0'}  # Weekly on Sunday
            ],
            'workflow_dispatch': {}
        },
        'jobs': {
            'security-scan': {
                'runs-on': 'ubuntu-latest',
                'steps': [
                    {
                        'name': 'Checkout code',
                        'uses': 'actions/checkout@v4'
                    },
                    {
                        'name': 'Run SAST with CodeQL',
                        'uses': 'github/codeql-action/init@v2',
                        'with': {
                            'languages': 'python'
                        }
                    },
                    {
                        'name': 'Autobuild',
                        'uses': 'github/codeql-action/autobuild@v2'
                    },
                    {
                        'name': 'Perform CodeQL Analysis',
                        'uses': 'github/codeql-action/analyze@v2'
                    },
                    {
                        'name': 'Container security scan',
                        'run': '''
                            docker build -t temp-scan-image .
                            docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
                              aquasec/trivy image temp-scan-image
                        '''
                    },
                    {
                        'name': 'Dependency security check',
                        'run': '''
                            pip install safety
                            safety check --json
                        '''
                    }
                ]
            }
        }
    }
    
    return security_workflow

def save_workflows():
    """Save all workflow files."""
    
    workflows_dir = Path(".github/workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Main CI/CD workflow
    main_workflow = create_ml_cicd_pipeline()
    with open(workflows_dir / "ml-cicd.yml", 'w') as f:
        yaml.dump(main_workflow, f, default_flow_style=False, sort_keys=False)
    
    # Performance monitoring workflow
    performance_workflow = create_model_performance_workflow()
    with open(workflows_dir / "model-performance.yml", 'w') as f:
        yaml.dump(performance_workflow, f, default_flow_style=False, sort_keys=False)
    
    # Security workflow
    security_workflow = create_security_workflow()
    with open(workflows_dir / "security.yml", 'w') as f:
        yaml.dump(security_workflow, f, default_flow_style=False, sort_keys=False)
    
    print("✅ CI/CD workflows created successfully!")
    print(f"📁 Workflows saved to: {workflows_dir}")

if __name__ == "__main__":
    save_workflows()
