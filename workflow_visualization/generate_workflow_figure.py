#!/usr/bin/env python3
"""
Script to generate a figure from the PlantUML workflow diagram.
Uses PlantUML web service to render the diagram.
"""

import base64
import zlib
from pathlib import Path

import requests


def plantuml_encode(plantuml_text):
    """Encode PlantUML text for web service."""
    compressed = zlib.compress(plantuml_text.encode("utf-8"))
    encoded = base64.b64encode(compressed).decode("ascii")
    # PlantUML uses a custom base64 variant
    return encoded.translate(str.maketrans("+/", "-_")).rstrip("=")


def generate_diagram():
    """Generate PNG diagram from PlantUML file."""

    # Read the PlantUML file
    puml_file = Path("mumdia_workflow_summary.puml")
    if not puml_file.exists():
        print(f"Error: {puml_file} not found!")
        return False

    with open(puml_file, "r") as f:
        puml_content = f.read()

    print("PlantUML content loaded successfully!")
    print(f"Content length: {len(puml_content)} characters")

    try:
        # Encode the PlantUML content
        encoded_content = plantuml_encode(puml_content)

        # Use PlantUML web service
        url = f"http://www.plantuml.com/plantuml/png/{encoded_content}"

        print("Requesting diagram from PlantUML web service...")
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            # Save the PNG image
            output_file = "mumdia_workflow_diagram.png"
            with open(output_file, "wb") as f:
                f.write(response.content)

            print(f"✅ Diagram generated successfully: {output_file}")
            print(f"Image size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🔧 MuMDIA Workflow Diagram Generator")
    print("=" * 50)

    success = generate_diagram()

    if success:
        print("\n✅ Success! The workflow diagram has been generated.")
        print("📁 Output file: mumdia_workflow_diagram.png")
        print("\nThe diagram shows the complete MuMDIA workflow including:")
        print("• Setup & Configuration")
        print("• Initial Peptide Search")
        print("• Retention Time Modeling")
        print("• Retention Window Search")
        print("• Spectral Data Parsing")
        print("• Feature Engineering")
        print("• Machine Learning Scoring")
        print("• Results & Cleanup")
    else:
        print("\n❌ Failed to generate diagram.")
        print("Please check your internet connection and try again.")
