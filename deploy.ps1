<#
.SYNOPSIS
    Build and deploy Context Compression Showcase to Rancher local Kubernetes cluster.

.DESCRIPTION
    This script:
    1. Builds the Docker image
    2. Connects to the local Rancher Desktop Kubernetes cluster
    3. Deploys the application (Deployment + Service + Ingress)
    4. Waits for the pod to be ready
    5. Prints the access URL

.NOTES
    Prerequisites:
    - Rancher Desktop installed and running (with Kubernetes enabled)
    - docker CLI available (Rancher Desktop provides this)
    - kubectl configured (Rancher Desktop configures this automatically)

.USAGE
    .\deploy.ps1
    .\deploy.ps1 -Rebuild       # Force rebuild the image
    .\deploy.ps1 -Teardown      # Remove deployment from cluster
#>

param(
    [switch]$Rebuild,
    [switch]$Teardown
)

$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
$IMAGE_NAME = "context-compression"
$IMAGE_TAG = "latest"
$FULL_IMAGE = "${IMAGE_NAME}:${IMAGE_TAG}"
$NAMESPACE = "default"
$NODE_PORT = 30080
$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$K8S_DIR = Join-Path $PROJECT_DIR "k8s"

# ─────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────
function Write-Step {
    param([string]$Message)
    Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# ─────────────────────────────────────────────────────────
# Teardown Mode
# ─────────────────────────────────────────────────────────
if ($Teardown) {
    Write-Step "Tearing down deployment..."
    kubectl delete -f "$K8S_DIR\ingress.yaml" --ignore-not-found 2>$null
    kubectl delete -f "$K8S_DIR\service.yaml" --ignore-not-found 2>$null
    kubectl delete -f "$K8S_DIR\deployment.yaml" --ignore-not-found 2>$null
    Write-Host "`n[OK] All resources removed." -ForegroundColor Green
    exit 0
}

# ─────────────────────────────────────────────────────────
# Pre-flight Checks
# ─────────────────────────────────────────────────────────
Write-Step "Step 0: Pre-flight checks"

# Check Docker
if (-not (Test-Command "docker")) {
    Write-Host "[ERROR] Docker CLI not found. Ensure Rancher Desktop is installed and running." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Docker CLI available" -ForegroundColor Green

# Check kubectl
if (-not (Test-Command "kubectl")) {
    Write-Host "[ERROR] kubectl not found. Ensure Rancher Desktop Kubernetes is enabled." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] kubectl available" -ForegroundColor Green

# Check cluster connectivity
$clusterInfo = kubectl cluster-info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Cannot connect to Kubernetes cluster." -ForegroundColor Red
    Write-Host "  Ensure Rancher Desktop is running with Kubernetes enabled." -ForegroundColor Yellow
    Write-Host "  Check: Rancher Desktop > Preferences > Kubernetes > Enable Kubernetes" -ForegroundColor Yellow
    exit 1
}
Write-Host "  [OK] Connected to Kubernetes cluster" -ForegroundColor Green

# Detect container runtime (dockerd vs containerd)
$rancherContext = kubectl config current-context 2>$null
Write-Host "  [OK] Kubernetes context: $rancherContext" -ForegroundColor Green

# ─────────────────────────────────────────────────────────
# Step 1: Build Docker Image
# ─────────────────────────────────────────────────────────
Write-Step "Step 1: Building Docker image [$FULL_IMAGE]"

Push-Location $PROJECT_DIR

# Check if image already exists (skip build unless -Rebuild)
$existingImage = docker images -q $FULL_IMAGE 2>$null
if ($existingImage -and -not $Rebuild) {
    Write-Host "  [SKIP] Image already exists. Use -Rebuild to force rebuild." -ForegroundColor Yellow
} else {
    Write-Host "  Building image..." -ForegroundColor White
    docker build -t $FULL_IMAGE .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker build failed." -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Write-Host "  [OK] Image built successfully" -ForegroundColor Green
}

Pop-Location

# If using containerd (nerdctl), load image into k8s namespace
$useNerdctl = Test-Command "nerdctl"
if ($useNerdctl) {
    Write-Host "  Loading image into containerd k8s namespace..." -ForegroundColor White
    docker save $FULL_IMAGE | nerdctl --namespace k8s.io load
    Write-Host "  [OK] Image loaded into k8s.io namespace" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────
# Step 2: Connect to Local Kubernetes Cluster
# ─────────────────────────────────────────────────────────
Write-Step "Step 2: Connecting to Rancher local Kubernetes cluster"

# Set context to rancher-desktop if available
$contexts = kubectl config get-contexts -o name 2>$null
$rancherCtx = $contexts | Where-Object { $_ -match "rancher-desktop" } | Select-Object -First 1

if ($rancherCtx) {
    kubectl config use-context $rancherCtx 2>$null
    Write-Host "  [OK] Switched to context: $rancherCtx" -ForegroundColor Green
} else {
    Write-Host "  [INFO] Using current context: $(kubectl config current-context)" -ForegroundColor Yellow
}

# Verify node is ready
$nodeStatus = kubectl get nodes -o jsonpath='{.items[0].status.conditions[-1].type}' 2>$null
if ($nodeStatus -eq "Ready") {
    Write-Host "  [OK] Node is Ready" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Node status: $nodeStatus — deployment may fail" -ForegroundColor Yellow
}

# ─────────────────────────────────────────────────────────
# Step 3: Deploy to Kubernetes
# ─────────────────────────────────────────────────────────
Write-Step "Step 3: Deploying to Kubernetes cluster"

# Apply manifests
Write-Host "  Applying deployment..." -ForegroundColor White
kubectl apply -f "$K8S_DIR\deployment.yaml"

Write-Host "  Applying service..." -ForegroundColor White
kubectl apply -f "$K8S_DIR\service.yaml"

Write-Host "  Applying ingress..." -ForegroundColor White
kubectl apply -f "$K8S_DIR\ingress.yaml"

Write-Host "  [OK] All manifests applied" -ForegroundColor Green

# ─────────────────────────────────────────────────────────
# Step 4: Wait for Pod to be Ready
# ─────────────────────────────────────────────────────────
Write-Step "Step 4: Waiting for pod to be ready..."

$timeout = 120
$elapsed = 0
$ready = $false

while ($elapsed -lt $timeout) {
    $podStatus = kubectl get pods -l app=context-compression -n $NAMESPACE -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>$null
    if ($podStatus -eq "True") {
        $ready = $true
        break
    }
    Write-Host "  Waiting... ($elapsed s)" -ForegroundColor Gray -NoNewline
    Write-Host "`r" -NoNewline
    Start-Sleep -Seconds 5
    $elapsed += 5
}

if (-not $ready) {
    Write-Host "`n  [WARN] Pod not ready after ${timeout}s. Checking status..." -ForegroundColor Yellow
    kubectl get pods -l app=context-compression -n $NAMESPACE
    kubectl describe pod -l app=context-compression -n $NAMESPACE | Select-String -Pattern "Warning|Error|Failed" | Select-Object -First 10
} else {
    Write-Host "  [OK] Pod is running and ready!                    " -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────
# Step 5: Provide Access URL
# ─────────────────────────────────────────────────────────
Write-Step "Step 5: Access Information"

# Get node IP
$nodeIP = kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>$null
if (-not $nodeIP) { $nodeIP = "localhost" }

# Get assigned NodePort (in case it differs)
$assignedPort = kubectl get svc context-compression-svc -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}' 2>$null
if (-not $assignedPort) { $assignedPort = $NODE_PORT }

$accessURL = "http://localhost:${assignedPort}"
$nodeURL = "http://${nodeIP}:${assignedPort}"

Write-Host ""
Write-Host "  ┌──────────────────────────────────────────────────────┐" -ForegroundColor Green
Write-Host "  │  Context Compression Showcase is DEPLOYED!           │" -ForegroundColor Green
Write-Host "  │                                                      │" -ForegroundColor Green
Write-Host "  │  Access URL:  $accessURL                    │" -ForegroundColor Green
Write-Host "  │  Node URL:    $nodeURL               │" -ForegroundColor Green
Write-Host "  │                                                      │" -ForegroundColor Green
Write-Host "  │  Endpoints:                                          │" -ForegroundColor Green
Write-Host "  │    /        - Showcase HTML comparison page           │" -ForegroundColor Green
Write-Host "  │    /health  - Health check                           │" -ForegroundColor Green
Write-Host "  │    /run     - Re-run pipeline & refresh results      │" -ForegroundColor Green
Write-Host "  │                                                      │" -ForegroundColor Green
Write-Host "  │  Ingress:  http://context-compression.localhost      │" -ForegroundColor Green
Write-Host "  └──────────────────────────────────────────────────────┘" -ForegroundColor Green
Write-Host ""

# Show pod status
Write-Host "  Pod Status:" -ForegroundColor White
kubectl get pods -l app=context-compression -n $NAMESPACE -o wide

Write-Host ""
Write-Host "  Useful Commands:" -ForegroundColor White
Write-Host "    kubectl logs -l app=context-compression -f     # Stream logs" -ForegroundColor Gray
Write-Host "    kubectl get all -l app=context-compression     # View all resources" -ForegroundColor Gray
Write-Host "    .\deploy.ps1 -Teardown                         # Remove deployment" -ForegroundColor Gray
Write-Host ""
