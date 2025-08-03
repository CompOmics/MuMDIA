import java.io.*;
import java.nio.file.*;

/**
 * MuMDIA Workflow Diagram Generator using PlantUML Java Library
 * 
 * This program reads the PlantUML source file and generates a high-quality
 * workflow diagram using the official PlantUML JAR library.
 */
public class WorkflowDiagramGenerator {
    
    private static final String PLANTUML_JAR = "plantuml.jar";
    private static final String INPUT_FILE = "mumdia_workflow_summary.puml";
    private static final String OUTPUT_PREFIX = "mumdia_workflow_java";
    
    public static void main(String[] args) {
        System.out.println("🔧 MuMDIA Workflow Diagram Generator (Java Edition)");
        System.out.println("=".repeat(60));
        
        try {
            // Check if required files exist
            if (!Files.exists(Paths.get(PLANTUML_JAR))) {
                System.err.println("❌ Error: " + PLANTUML_JAR + " not found!");
                System.err.println("Please download PlantUML JAR file first.");
                System.exit(1);
            }
            
            if (!Files.exists(Paths.get(INPUT_FILE))) {
                System.err.println("❌ Error: " + INPUT_FILE + " not found!");
                System.exit(1);
            }
            
            System.out.println("📁 Input file: " + INPUT_FILE);
            System.out.println("📦 PlantUML JAR: " + PLANTUML_JAR);
            
            // Generate different format outputs
            generateDiagram("png", "PNG image");
            generateDiagram("svg", "SVG vector");
            generateDiagram("pdf", "PDF document");
            
            System.out.println("\n✅ Success! All workflow diagrams have been generated.");
            System.out.println("\n📁 Generated files:");
            listGeneratedFiles();
            
            System.out.println("\n🎯 The diagrams show the complete MuMDIA workflow:");
            printWorkflowSummary();
            
        } catch (Exception e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static void generateDiagram(String format, String description) {
        try {
            System.out.println("\n🔄 Generating " + description + "...");
            
            ProcessBuilder pb = new ProcessBuilder(
                "java", "-jar", PLANTUML_JAR,
                "-t" + format,
                "-o", ".",
                INPUT_FILE
            );
            
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            // Capture output
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
            );
            
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("  " + line);
            }
            
            int exitCode = process.waitFor();
            
            if (exitCode == 0) {
                System.out.println("  ✅ " + description + " generated successfully!");
            } else {
                System.err.println("  ❌ Error generating " + description + " (exit code: " + exitCode + ")");
            }
            
        } catch (IOException | InterruptedException e) {
            System.err.println("  ❌ Error generating " + description + ": " + e.getMessage());
        }
    }
    
    private static void listGeneratedFiles() {
        try {
            Files.list(Paths.get("."))
                .filter(path -> {
                    String fileName = path.getFileName().toString();
                    return fileName.startsWith("mumdia_workflow_summary") && 
                           (fileName.endsWith(".png") || fileName.endsWith(".svg") || fileName.endsWith(".pdf"));
                })
                .forEach(path -> {
                    try {
                        long size = Files.size(path);
                        System.out.println("  📄 " + path.getFileName() + " (" + formatFileSize(size) + ")");
                    } catch (IOException e) {
                        System.out.println("  📄 " + path.getFileName());
                    }
                });
        } catch (IOException e) {
            System.err.println("  Error listing files: " + e.getMessage());
        }
    }
    
    private static String formatFileSize(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
    }
    
    private static void printWorkflowSummary() {
        System.out.println("  • Setup & Configuration");
        System.out.println("  • Initial Peptide Search (Sage engine)");
        System.out.println("  • Retention Time Modeling (DeepLC)");
        System.out.println("  • Retention Window Search");
        System.out.println("  • Spectral Data Parsing (mzML)");
        System.out.println("  • Feature Engineering (multi-modal)");
        System.out.println("  • Machine Learning Scoring (Neural Networks)");
        System.out.println("  • Results & Cleanup (Mokapot output)");
        
        System.out.println("\n🚀 Key Features:");
        System.out.println("  • Parallel processing (24 workers)");
        System.out.println("  • Caching system with pickle files");
        System.out.println("  • Multi-modal feature integration");
        System.out.println("  • FDR-controlled statistical validation");
    }
}
