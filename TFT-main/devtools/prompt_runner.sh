#!/bin/bash
# Copilot Prompt Runner - Execute all validation tests in batch
# Usage: ./prompt_runner.sh [options]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_CMD="python3"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if virtual environment exists and activate it
setup_environment() {
    print_status "Setting up Python environment..."
    
    # Check if virtual environment exists
    if [ -d "$VENV_PATH" ]; then
        print_status "Activating virtual environment: $VENV_PATH"
        source "$VENV_PATH/bin/activate"
    else
        print_warning "No virtual environment found at $VENV_PATH"
        print_status "Using system Python: $(which $PYTHON_CMD)"
    fi
    
    # Verify Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    print_status "Python version: $PYTHON_VERSION"
    
    # Check if required packages are installed
    if ! $PYTHON_CMD -c "import pandas, numpy" 2>/dev/null; then
        print_warning "Some required packages may be missing"
        print_status "Consider running: pip install -r requirements.txt"
    fi
}

# Function to run Copilot header insertion
run_header_insertion() {
    local dry_run=$1
    
    print_status "Running Copilot header insertion..."
    
    cd "$PROJECT_ROOT"
    
    if [ "$dry_run" = "true" ]; then
        print_status "Running in DRY RUN mode..."
        $PYTHON_CMD devtools/insert_copilot_headers.py --dry-run
    else
        print_status "Inserting Copilot headers..."
        $PYTHON_CMD devtools/insert_copilot_headers.py
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "Header insertion completed successfully"
    else
        print_error "Header insertion failed with exit code $exit_code"
        return $exit_code
    fi
}

# Function to run validation tests
run_validation_tests() {
    local test_category=$1
    
    print_status "Running Polygon.io validation tests..."
    
    cd "$PROJECT_ROOT"
    
    if [ -n "$test_category" ]; then
        print_status "Running specific test category: $test_category"
        $PYTHON_CMD devtools/test_polygon_prompts.py --test "$test_category"
    else
        print_status "Running all validation tests..."
        $PYTHON_CMD devtools/test_polygon_prompts.py --all
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "All validation tests passed"
    else
        print_error "Some validation tests failed with exit code $exit_code"
        return $exit_code
    fi
}

# Function to validate prompt formatting
validate_prompts() {
    print_status "Validating prompt file formatting..."
    
    cd "$PROJECT_ROOT"
    $PYTHON_CMD devtools/test_polygon_prompts.py --validate-prompts
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "Prompt formatting validation passed"
    else
        print_error "Prompt formatting validation failed"
        return $exit_code
    fi
}

# Function to run code quality checks
run_quality_checks() {
    print_status "Running code quality checks..."
    
    cd "$PROJECT_ROOT"
    
    # Check for Python syntax errors
    print_status "Checking Python syntax..."
    find . -name "*.py" -not -path "./.venv/*" -not -path "./.git/*" | \
    while read -r file; do
        if ! $PYTHON_CMD -m py_compile "$file" 2>/dev/null; then
            print_error "Syntax error in: $file"
            return 1
        fi
    done
    
    # Check for import issues (basic check)
    print_status "Checking basic imports..."
    if ! $PYTHON_CMD -c "
import sys
sys.path.append('.')
try:
    import polygon_data_loader
    print('✅ polygon_data_loader imports OK')
except Exception as e:
    print(f'⚠️ polygon_data_loader import issue: {e}')

try:
    import data_preprocessing
    print('✅ data_preprocessing imports OK')
except Exception as e:
    print(f'⚠️ data_preprocessing import issue: {e}')
" 2>/dev/null; then
        print_warning "Some import issues detected (this may be expected if dependencies are missing)"
    fi
    
    print_success "Code quality checks completed"
}

# Function to generate comprehensive report
generate_report() {
    local output_file="$PROJECT_ROOT/devtools/copilot_validation_report.txt"
    
    print_status "Generating comprehensive validation report..."
    
    {
        echo "=================================="
        echo "COPILOT VALIDATION REPORT"
        echo "Generated: $(date)"
        echo "Project: TFT Stock Prediction System"
        echo "=================================="
        echo ""
        
        echo "ENVIRONMENT INFORMATION:"
        echo "------------------------"
        echo "Python Version: $($PYTHON_CMD --version 2>&1)"
        echo "Working Directory: $PROJECT_ROOT"
        echo "Virtual Environment: ${VENV_PATH:-"Not detected"}"
        echo ""
        
        echo "PROMPT FILES STATUS:"
        echo "-------------------"
        if [ -f "$PROJECT_ROOT/devtools/copilot_prompts_polygon.md" ]; then
            echo "✅ copilot_prompts_polygon.md exists"
            echo "   Size: $(wc -l < "$PROJECT_ROOT/devtools/copilot_prompts_polygon.md") lines"
        else
            echo "❌ copilot_prompts_polygon.md missing"
        fi
        
        if [ -f "$PROJECT_ROOT/devtools/test_polygon_prompts.py" ]; then
            echo "✅ test_polygon_prompts.py exists"
        else
            echo "❌ test_polygon_prompts.py missing"
        fi
        
        if [ -f "$PROJECT_ROOT/devtools/insert_copilot_headers.py" ]; then
            echo "✅ insert_copilot_headers.py exists"
        else
            echo "❌ insert_copilot_headers.py missing"
        fi
        echo ""
        
        echo "PYTHON FILES WITH COPILOT HEADERS:"
        echo "---------------------------------"
        find "$PROJECT_ROOT" -name "*.py" -not -path "./.venv/*" -not -path "./.git/*" | \
        while read -r file; do
            if grep -l "COPILOT PROMPT:" "$file" >/dev/null 2>&1; then
                echo "✅ $(basename "$file")"
            else
                echo "⚪ $(basename "$file") (no header)"
            fi
        done
        echo ""
        
        echo "VALIDATION TEST SUMMARY:"
        echo "-----------------------"
        echo "Last validation run: $(date)"
        echo "For detailed test results, run:"
        echo "  ./devtools/prompt_runner.sh --test-all"
        echo ""
        
    } > "$output_file"
    
    print_success "Report saved to: $output_file"
    
    # Display summary
    echo ""
    print_status "VALIDATION SUMMARY:"
    cat "$output_file" | grep -E "✅|❌|⚪" | head -20
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Copilot Validation Runner for TFT Stock Prediction System"
    echo ""
    echo "OPTIONS:"
    echo "  --setup              Set up environment and dependencies"
    echo "  --insert-headers     Insert Copilot headers into Python files"
    echo "  --dry-run           Run header insertion in dry-run mode"
    echo "  --test-all          Run all validation tests"
    echo "  --test CATEGORY     Run specific test category"
    echo "  --validate-prompts  Validate prompt file formatting"
    echo "  --quality-check     Run code quality checks"
    echo "  --report            Generate comprehensive validation report"
    echo "  --full              Run complete validation pipeline"
    echo "  --help              Show this help message"
    echo ""
    echo "TEST CATEGORIES:"
    echo "  api_structure       Test Polygon API call structure"
    echo "  rate_limiting       Test rate limiting and backoff logic"
    echo "  websocket           Test WebSocket integration"
    echo "  feature_engineering Test feature engineering functions"
    echo "  options             Test options symbol handling"
    echo "  data_quality        Test data quality validation"
    echo "  performance         Test performance requirements"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --full                    # Run complete validation"
    echo "  $0 --insert-headers --dry-run # Preview header changes"
    echo "  $0 --test rate_limiting      # Test specific category"
    echo "  $0 --report                  # Generate status report"
}

# Main execution logic
main() {
    local command="$1"
    local param="$2"
    
    # Change to script directory for relative path resolution
    cd "$SCRIPT_DIR"
    
    case "$command" in
        --setup)
            setup_environment
            ;;
        --insert-headers)
            setup_environment
            run_header_insertion false
            ;;
        --dry-run)
            setup_environment
            run_header_insertion true
            ;;
        --test-all)
            setup_environment
            run_validation_tests ""
            ;;
        --test)
            if [ -z "$param" ]; then
                print_error "Test category required. Use --help for available categories."
                exit 1
            fi
            setup_environment
            run_validation_tests "$param"
            ;;
        --validate-prompts)
            setup_environment
            validate_prompts
            ;;
        --quality-check)
            setup_environment
            run_quality_checks
            ;;
        --report)
            setup_environment
            generate_report
            ;;
        --full)
            print_status "Running complete Copilot validation pipeline..."
            echo ""
            
            setup_environment
            echo ""
            
            validate_prompts
            echo ""
            
            run_header_insertion false
            echo ""
            
            run_validation_tests ""
            echo ""
            
            run_quality_checks
            echo ""
            
            generate_report
            echo ""
            
            print_success "🎉 Complete validation pipeline finished!"
            ;;
        --help|"")
            show_usage
            ;;
        *)
            print_error "Unknown option: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
