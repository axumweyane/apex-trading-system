# DevTools for TFT Polygon.io Integration

This directory contains development tools for optimizing GitHub Copilot integration with Polygon.io APIs in the TFT Stock Prediction System.

## 📁 Files Overview

### 🚀 Core Tools

1. **`copilot_prompts_polygon.md`** - Comprehensive prompt library
   - 10 detailed implementation prompts
   - Critical options trading validation
   - Copilot integration standards
   - Usage guidelines and best practices

2. **`test_polygon_prompts.py`** - Validation test suite
   - Unit tests for Polygon API structure
   - Rate limiting and WebSocket testing
   - Feature engineering validation
   - Options handling verification

3. **`insert_copilot_headers.py`** - Automated header insertion
   - Scans Python files for missing prompts
   - Inserts appropriate Copilot headers
   - Supports dry-run mode
   - File-specific prompt mapping

4. **`prompt_runner.sh`** - Batch execution script
   - Complete validation pipeline
   - Environment setup automation
   - Comprehensive reporting
   - Quality checks integration

## 🛠️ Quick Start

### Run Complete Validation Pipeline
```bash
./devtools/prompt_runner.sh --full
```

### Insert Copilot Headers (Preview)
```bash
./devtools/prompt_runner.sh --dry-run
```

### Insert Copilot Headers (Apply)
```bash
./devtools/prompt_runner.sh --insert-headers
```

### Run Specific Tests
```bash
./devtools/prompt_runner.sh --test rate_limiting
./devtools/prompt_runner.sh --test websocket
./devtools/prompt_runner.sh --test options
```

### Generate Status Report
```bash
./devtools/prompt_runner.sh --report
```

## 📋 Available Test Categories

- **`api_structure`** - Polygon API call structure validation
- **`rate_limiting`** - Rate limiting and backoff logic
- **`websocket`** - WebSocket integration testing
- **`feature_engineering`** - Technical indicator calculations
- **`options`** - Options symbol parsing and Greeks
- **`data_quality`** - Data validation and cleaning
- **`performance`** - Performance requirement validation

## 🎯 Usage Workflow

1. **Review Prompts**: Check `copilot_prompts_polygon.md` for available prompts
2. **Insert Headers**: Use `insert_copilot_headers.py` to add prompts to files
3. **Generate Code**: Let GitHub Copilot implement based on prompts
4. **Validate**: Run `test_polygon_prompts.py` to verify implementation
5. **Report**: Generate status reports for team review

## 🔧 Manual Usage

### Individual Tools

#### Copilot Header Insertion
```bash
# Preview changes
python devtools/insert_copilot_headers.py --dry-run

# Apply to all files
python devtools/insert_copilot_headers.py

# Apply to specific files
python devtools/insert_copilot_headers.py --files polygon_data_loader.py api_postgres.py

# List available mappings
python devtools/insert_copilot_headers.py --list-mappings
```

#### Test Validation
```bash
# Run all tests
python devtools/test_polygon_prompts.py --all

# Run specific category
python devtools/test_polygon_prompts.py --test rate_limiting

# Validate prompt formatting
python devtools/test_polygon_prompts.py --validate-prompts
```

## 📊 File Mappings

The header insertion tool uses these mappings:

- **`polygon_data_loader.py`** → Batch OHLCV fetcher with rate limiting
- **`data_preprocessing.py`** → VWAP-based technical indicators
- **`api_postgres.py`** → Real-time prediction endpoints
- **`enhanced_data_pipeline.py`** → News sentiment integration
- **`realtime_handler.py`** → WebSocket client implementation
- **`scheduler.py`** → Batch processing automation
- **`tft_postgres_model.py`** → Polygon-specific TFT features

## 🏗️ Development Standards

### Prompt Format
```python
"""
# COPILOT PROMPT: [Specific implementation request]
# EXPECTED OUTPUT: [Brief description of expected functionality]
# POLYGON INTEGRATION: [Specific Polygon.io features used]
"""
```

### Testing Requirements
- All Copilot-generated code must pass validation tests
- Symbol format validation for all Polygon.io integrations
- Error handling for API failures and rate limits
- Performance requirements verification

### Code Quality
- Type hints for Polygon-specific data structures
- Comprehensive error handling
- Rate limiting and caching implementation
- Production-ready logging and monitoring

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors in Tests**
   - Install required packages: `pip install pytest pandas numpy`
   - Some tests may skip if optional dependencies missing

2. **Header Insertion Conflicts**
   - Files with existing headers are skipped
   - Use `--force` flag to override (with caution)

3. **Test Failures**
   - Review specific test output
   - Ensure Copilot-generated code matches prompt requirements
   - Update prompts based on real-world implementation needs

### Environment Setup
```bash
# Ensure Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Make scripts executable
chmod +x devtools/prompt_runner.sh
```

## 📈 Integration Benefits

### For Developers
- **Consistent Prompts**: Standardized across all files
- **Quality Validation**: Automated testing of AI-generated code
- **Time Savings**: Automated header management
- **Best Practices**: Built-in Polygon.io integration patterns

### For GitHub Copilot
- **Clear Context**: Specific implementation requirements
- **Domain Knowledge**: Financial trading and Polygon.io specifics
- **Error Prevention**: Predefined error handling patterns
- **Performance Optimization**: Built-in efficiency requirements

## 🔄 Maintenance

### Updating Prompts
1. Edit `copilot_prompts_polygon.md`
2. Update mappings in `insert_copilot_headers.py`
3. Add corresponding tests in `test_polygon_prompts.py`
4. Run validation pipeline to verify changes

### Adding New Files
1. Add prompt mapping to `PROMPT_MAPPINGS` in `insert_copilot_headers.py`
2. Create corresponding test case in `test_polygon_prompts.py`
3. Update this README with new file information

---

**Last Updated**: August 2025  
**Version**: 1.0  
**Compatibility**: GitHub Copilot, VS Code, Python 3.8+
