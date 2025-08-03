# MuMDIA Testing Framework

This document describes the comprehensive testing framework for MuMDIA, a proteomics pipeline for mass spectrometry data analysis.

## Overview

The testing framework provides comprehensive coverage for:
- Fragment intensity correlation calculations
- Data structure validation and serialization  
- Utility functions (logging, I/O, pickling)
- Error handling and edge cases
- Performance validation

## Quick Start

### Install Test Dependencies

```bash
pip install -r test_requirements.txt
```

### Run All Tests

```bash
# Using the test runner
python tests/run_tests.py all --verbose

# Using pytest directly
pytest tests/ -v

# Using make
make test
```

### Run Specific Test Categories

```bash
# Fragment intensity tests
python tests/run_tests.py fragment

# Data structure tests  
python tests/run_tests.py data

# Utility function tests
python tests/run_tests.py utils
```

### Generate Coverage Report

```bash
# HTML coverage report
python tests/run_tests.py all --coverage

# Using make
make coverage
```

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures and test configuration
├── test_features_fragment_intensity.py  # Fragment intensity correlation tests
├── test_data_structures.py             # Data structure validation tests
├── test_utilities.py                   # Utility function tests
├── run_tests.py                        # Test runner script
└── README.md                           # This file
```

## Test Categories

### Fragment Intensity Tests (`test_features_fragment_intensity.py`)

Tests the core fragment intensity correlation functionality:

- **Fragment Matching**: Tests `match_fragments()` with various scenarios
- **Correlation Calculations**: Tests multiple correlation metrics (Pearson, Spearman, etc.)
- **Data Processing**: Tests intensity normalization and filtering
- **Edge Cases**: Empty spectra, single peaks, identical intensities
- **Integration**: End-to-end workflow testing

Key test markers: `@pytest.mark.fragmentation`, `@pytest.mark.unit`

### Data Structure Tests (`test_data_structures.py`)

Tests dataclass definitions and validation:

- **CorrelationResults**: Validation of correlation metrics and statistics
- **PickleConfig**: Serialization configuration validation
- **SpectraData**: MS2 spectra data structure validation
- **Serialization**: Pickle/unpickle operations
- **Field Validation**: Type checking and constraint validation

Key test markers: `@pytest.mark.data_validation`, `@pytest.mark.unit`

### Utility Tests (`test_utilities.py`)

Tests supporting utility functions:

- **Logging**: Logger configuration and output validation
- **I/O Operations**: File reading/writing with error handling
- **Pickling**: Serialization with compression and error recovery
- **Error Handling**: Exception scenarios and recovery
- **Performance**: Basic performance validation

Key test markers: `@pytest.mark.io`, `@pytest.mark.unit`

## Test Fixtures

The `conftest.py` file provides shared test fixtures:

### Sample Data Fixtures

- `sample_psm_data`: Representative PSM (Peptide-Spectrum Match) data
- `sample_fragment_data`: Fragment ion theoretical data
- `sample_ms2pip_predictions`: MS2PIP intensity predictions
- `sample_correlation_results`: Expected correlation calculation results

### Mock Objects

- `mock_logger`: Configurable logger mock for testing
- `mock_file_operations`: File I/O operation mocks
- Temporary file and directory fixtures

## Running Tests

### Command Line Options

```bash
# Verbose output
python tests/run_tests.py all --verbose

# Run with coverage
python tests/run_tests.py all --coverage

# Run specific markers
pytest tests/ -m "fragmentation" -v

# Stop on first failure  
pytest tests/ -x

# Run failed tests first
pytest tests/ --lf
```

### Using Make Commands

```bash
make test              # Run all tests
make test-unit         # Run unit tests only
make test-fast         # Run fast tests only
make coverage          # Run with coverage
make lint              # Run code linting
make format            # Format code
make type-check        # Run type checking
make clean             # Clean test artifacts
```

## Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit`: Unit tests for individual functions
- `@pytest.mark.integration`: Integration tests for component interaction
- `@pytest.mark.fast`: Quick tests suitable for CI/CD
- `@pytest.mark.slow`: Longer-running tests
- `@pytest.mark.fragmentation`: Fragment intensity related tests
- `@pytest.mark.data_validation`: Data structure validation tests
- `@pytest.mark.io`: Input/output operation tests
- `@pytest.mark.performance`: Performance benchmarking tests

## Coverage Reports

Coverage reports are generated in multiple formats:

- **Terminal**: Summary displayed after test run
- **HTML**: Detailed report in `htmlcov/` directory
- **XML**: Machine-readable format for CI/CD integration

Target coverage thresholds:
- Overall: >85%
- Critical modules: >90%
- Fragment intensity calculations: >95%

## Performance Testing

Performance tests validate:

- Correlation calculation speed for large datasets
- Memory usage for processing multiple spectra
- I/O performance for file operations
- Scalability with increasing data size

## Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r test_requirements.txt
    make test-fast
    
- name: Run coverage
  run: |
    make coverage
    
- name: Upload coverage
  uses: codecov/codecov-action@v1
```

## Best Practices

### Writing New Tests

1. **Use descriptive test names**: `test_correlation_calculation_with_empty_spectra`
2. **Include docstrings**: Explain test purpose and expected behavior
3. **Use appropriate markers**: Categorize tests for selective running
4. **Test edge cases**: Empty data, invalid inputs, boundary conditions
5. **Mock external dependencies**: Use fixtures for reproducible tests

### Test Data

1. **Use fixtures**: Share common test data via `conftest.py`
2. **Keep data minimal**: Use smallest datasets that validate functionality
3. **Document data sources**: Explain origin and characteristics of test data
4. **Version test data**: Track changes to test datasets

### Error Testing

1. **Test error conditions**: Invalid inputs, missing files, corrupted data
2. **Validate error messages**: Ensure helpful error information
3. **Test recovery**: Verify graceful handling of errors
4. **Test logging**: Ensure errors are properly logged

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure MuMDIA modules are in Python path
2. **Missing dependencies**: Install all packages from `test_requirements.txt`
3. **File permissions**: Ensure write access for temporary test files
4. **Memory issues**: Reduce test data size for memory-constrained environments

### Debug Mode

Run tests with additional debugging:

```bash
# Verbose pytest output
pytest tests/ -v -s --tb=long

# Debug specific test
pytest tests/test_features_fragment_intensity.py::test_correlation_calculation -v -s

# Run with pdb debugger
pytest tests/ --pdb
```

## Contributing

When adding new features to MuMDIA:

1. **Write tests first**: Use TDD approach where possible
2. **Maintain coverage**: Ensure new code has >85% test coverage
3. **Update fixtures**: Add new test data as needed
4. **Document tests**: Update this README for new test categories
5. **Run full suite**: Ensure all tests pass before committing

For questions or issues with the testing framework, please refer to the main MuMDIA documentation or open an issue in the project repository.
