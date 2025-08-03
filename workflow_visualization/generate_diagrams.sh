#!/bin/bash
# Simple script to generate workflow diagrams using PlantUML JAR

echo "🔧 MuMDIA Workflow Diagram Generator (Simple)"
echo "============================================="

# Check if PlantUML JAR exists
if [ ! -f "plantuml.jar" ]; then
    echo "📦 Downloading PlantUML JAR..."
    wget -q https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar
    echo "✅ PlantUML JAR downloaded"
fi

# Check if input file exists
if [ ! -f "mumdia_workflow_summary.puml" ]; then
    echo "❌ Error: mumdia_workflow_summary.puml not found!"
    exit 1
fi

echo "📁 Input: mumdia_workflow_summary.puml"

# Generate PNG (high quality for presentations)
echo "🔄 Generating PNG..."
java -jar plantuml.jar -tpng mumdia_workflow_summary.puml
echo "✅ PNG generated"

# Generate SVG (vector format for web/documents)
echo "🔄 Generating SVG..."
java -jar plantuml.jar -tsvg mumdia_workflow_summary.puml
echo "✅ SVG generated"

echo ""
echo "📁 Generated files:"
ls -lh mumdia_workflow_summary.png mumdia_workflow_summary.svg 2>/dev/null || echo "Files not found"

echo ""
echo "✅ Workflow diagrams generated successfully!"
echo "🎯 Use PNG for presentations, SVG for web/documents"
