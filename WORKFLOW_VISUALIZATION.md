# MuMDIA Project Structure

## Workflow Visualization

The `workflow_visualization/` folder contains all files related to generating and viewing the MuMDIA workflow diagrams:

- **Source**: PlantUML workflow definition
- **Generators**: Multiple tools for creating diagrams (Java, Python, Shell)
- **Output**: High-quality PNG, SVG, and PDF diagrams
- **Documentation**: Complete usage instructions

### Quick Start
```bash
# Generate workflow diagrams
make diagrams

# Or manually
cd workflow_visualization && ./generate_diagrams.sh
```

For detailed information, see [`workflow_visualization/README.md`](workflow_visualization/README.md).

## Main Project Components

- `feature_generators/` - Feature extraction modules
- `utilities/` - Utility functions and helpers  
- `parsers/` - Data parsing modules
- `tests/` - Comprehensive test suite
- `configs/` - Configuration files
- `workflow_visualization/` - Workflow diagrams and generation tools

The workflow diagrams provide a visual overview of the complete MuMDIA proteomics analysis pipeline.
