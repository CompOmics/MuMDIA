# MuMDIA Workflow Diagram

## Overview
This directory contains a visual representation of the MuMDIA (Multi-modal Data-Independent Acquisition) proteomics analysis workflow.

## Files Generated
- `mumdia_workflow_summary.png` - High-quality PNG diagram (186 KB)
- `mumdia_workflow_summary.svg` - Vector SVG diagram (34 KB)
- `mumdia_workflow_summary.puml` - PlantUML source file
- `generate_workflow_figure.py` - Python script (web service approach)
- `WorkflowDiagramGenerator.java` - Java application (local PlantUML JAR)
- `generate_diagrams.sh` - Simple bash script for diagram generation
- `plantuml.jar` - PlantUML Java library (22 MB)

## Workflow Components

The MuMDIA workflow consists of the following major stages:

### 1. Setup & Configuration
- Parse command-line arguments
- Create output directories  
- Merge configuration with CLI arguments
- Generate configuration dictionary

### 2. Initial Peptide Search
- Run Sage search engine on raw data
- Load and filter peptide-spectrum matches (PSMs)
- Parse fragment ion matches
- Extract fragment intensity data

### 3. Retention Time Modeling
- Perform tryptic digest of FASTA database
- Retrain DeepLC model for retention time prediction
- Calculate RT prediction bounds and windows
- Generate calibrated RT predictions

### 4. Retention Window Search  
- Split mzML files by retention time windows
- Run targeted Sage searches within windows
- Merge results with initial broad search
- Expand PSM dataset with enhanced fragment coverage

### 5. Spectral Data Parsing
- Parse mzML mass spectrometry files
- Extract MS1 (precursor) and MS2 (fragment) spectra
- Build scan-to-scan relationships and mappings

### 6. Feature Engineering
- **Retention Time Features**: DeepLC predictions and error calculations
- **Fragment Intensity Features**: MS2PiP predictions and correlations  
- **MS1 Features**: Precursor ion intensities and properties
- **Statistical Features**: Peptidoform-level aggregations
- Parallel processing with 24 workers and chunked analysis

### 7. Machine Learning Scoring
- Neural network classifier (Keras-based)
- 3-fold cross-validation training
- False Discovery Rate (FDR) control
- Generate PSM confidence scores and q-values

### 8. Results & Cleanup
- Export final scored results
- Generate comprehensive reports
- Clean temporary files
- Output files: mokapot.psms.txt, mokapot.peptides.txt, mokapot.proteins.txt

## Key Features

### Parallel Processing
- Multi-threaded feature computation (24 workers)
- Chunked processing for memory efficiency (500 peptidoforms per chunk)

### Caching System
- Pickle-based intermediate result caching
- Enables workflow resumption and debugging

### Modular Architecture
- Independent processing stages
- Configurable parameters
- Extensible feature engineering

## Technical Stack
- **Search Engine**: Sage
- **Retention Time Prediction**: DeepLC  
- **Fragment Prediction**: MS2PiP
- **Machine Learning**: Keras/TensorFlow neural networks
- **Statistical Control**: Mokapot for FDR estimation
- **Data Processing**: Polars for high-performance data manipulation

## Usage

### Method 1: Makefile (from project root)
```bash
# From the main MuMDIA directory
make diagrams
```

### Method 2: Shell script (from this folder)
```bash
# From workflow_visualization/ directory
./generate_diagrams.sh
```

### Method 3: Java application (from this folder)
```bash
# From workflow_visualization/ directory
java WorkflowDiagramGenerator
```

### Method 4: Python web service (from this folder)
```bash
# From workflow_visualization/ directory
python generate_workflow_figure.py
```

### Method 5: Direct PlantUML JAR (from this folder)
```bash
# From workflow_visualization/ directory
java -jar plantuml.jar -tpng mumdia_workflow_summary.puml
java -jar plantuml.jar -tsvg mumdia_workflow_summary.puml
```

## Output Formats
- **PNG**: High-quality raster image (ideal for presentations, papers)
- **SVG**: Scalable vector graphics (ideal for web, documentation)
- **PDF**: Document format (requires additional dependencies)

The diagram provides a comprehensive overview of the MuMDIA pipeline for multi-modal proteomics analysis, showing data flow, key processing steps, and output generation.
