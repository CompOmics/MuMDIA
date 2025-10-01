#!/usr/bin/env python3
"""
Demo script showing how to use the new simplified MuMDIA configuration system.

This script demonstrates:
1. Loading config from JSON with override mechanism
2. How parameters get different values for initial vs full search
3. How to run the main workflow with the new system

Run this with: python demo_new_config.py
"""

from config import load_config_from_json

def main():
    print("🔧 MuMDIA New Configuration System Demo")
    print("=" * 50)
    
    # 1. Load config from JSON
    print("\n1. Loading configuration from configs/config_simple.json...")
    config = load_config_from_json('configs/config_simple.json')
    print(f"✅ Configuration loaded successfully!")
    
    # 2. Show base parameters
    print(f"\n2. Base configuration parameters:")
    print(f"   📁 mzML file: {config.mzml_file}")
    print(f"   📁 FASTA file: {config.fasta_file}")
    print(f"   📁 Result directory: {config.result_dir}")
    print(f"   🔬 Base cleave_at: {config.cleave_at}")
    print(f"   🔬 Base report_psms: {config.report_psms}")
    print(f"   🔬 Base deisotope: {config.deisotope}")
    print(f"   🔬 Base max_variable_mods: {config.max_variable_mods}")
    
    # 3. Show override mechanism in action
    print(f"\n3. Override mechanism for different search stages:")
    
    # Initial search configuration
    initial_config = config.get_initial_search_config()
    print(f"   🟡 Initial search cleave_at: {initial_config['database']['enzyme']['cleave_at']}")
    print(f"   🟡 Initial search report_psms: {initial_config['report_psms']}")
    print(f"   🟡 Initial search deisotope: {initial_config['deisotope']}")
    print(f"   🟡 Initial search max_variable_mods: {initial_config['database']['max_variable_mods']}")
    
    # Full search configuration  
    full_config = config.get_full_search_config()
    print(f"   🟢 Full search cleave_at: {full_config['database']['enzyme']['cleave_at']}")
    print(f"   🟢 Full search report_psms: {full_config['report_psms']}")
    print(f"   🟢 Full search deisotope: {full_config['deisotope']}")
    print(f"   🟢 Full search max_variable_mods: {full_config['database']['max_variable_mods']}")
    
    # 4. Show MuMDIA-specific settings
    print(f"\n4. MuMDIA-specific settings:")
    mumdia_config = config.get_mumdia_config()
    print(f"   📊 FDR initial search: {config.fdr_init_search}")
    print(f"   📊 Number of windows: {config.n_windows}")
    print(f"   📊 Model type: {config.model_type}")
    print(f"   💾 Read initial search pickle: {mumdia_config['read_initial_search_pickle']}")
    print(f"   💾 Write initial search pickle: {mumdia_config['write_initial_search_pickle']}")
    
    # 5. Show legacy format compatibility
    print(f"\n5. Legacy format compatibility:")
    legacy_format = config.to_legacy_format()
    print(f"   ✅ Legacy format has {len(legacy_format)} top-level sections:")
    for section in legacy_format.keys():
        print(f"      - {section}")
    
    print(f"\n🎉 Demo completed!")
    print(f"\nTo run the actual MuMDIA workflow:")
    print(f"   python run.py configs/config_simple.json")

if __name__ == "__main__":
    main()
