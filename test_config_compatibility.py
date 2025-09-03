#!/usr/bin/env python3
"""
Test script to verify that both old and new config formats work with the new system.

This demonstrates that the new system is fully backwards compatible.
"""

from config_new import load_config_from_json

def test_config_format(config_path, format_name):
    """Test loading and using a specific config format."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {format_name} Config Format")
    print(f"📁 File: {config_path}")
    print(f"{'='*60}")
    
    try:
        # Load config
        config = load_config_from_json(config_path)
        print(f"✅ Config loaded successfully!")
        
        # Show basic parameters
        print(f"\n📋 Basic Parameters:")
        print(f"   mzML file: {config.mzml_file}")
        print(f"   FASTA file: {config.fasta_file}")
        print(f"   Result dir: {config.result_dir}")
        
        # Show search parameter differences
        print(f"\n🔬 Search Parameters:")
        initial_config = config.get_initial_search_config()
        full_config = config.get_full_search_config()
        
        print(f"   Parameter           Initial Search    Full Search")
        print(f"   cleave_at           {initial_config['database']['enzyme']['cleave_at']:15} {full_config['database']['enzyme']['cleave_at']}")
        print(f"   deisotope           {str(initial_config['deisotope']):15} {str(full_config['deisotope'])}")
        print(f"   report_psms         {str(initial_config['report_psms']):15} {str(full_config['report_psms'])}")
        print(f"   max_variable_mods   {str(initial_config['database']['max_variable_mods']):15} {str(full_config['database']['max_variable_mods'])}")
        
        # Show MuMDIA settings
        mumdia_config = config.get_mumdia_config()
        print(f"\n📊 MuMDIA Settings:")
        print(f"   FDR initial search: {config.fdr_init_search}")
        print(f"   Read initial pickle: {mumdia_config['read_initial_search_pickle']}")
        print(f"   Write initial pickle: {mumdia_config['write_initial_search_pickle']}")
        
        print(f"\n✅ {format_name} format works perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {format_name} format: {e}")
        return False

def main():
    print("🔧 MuMDIA Config Backwards Compatibility Test")
    print("Testing both old (nested) and new (flat) config formats...")
    
    # Test old nested format
    old_works = test_config_format("configs/config.json", "Legacy/Old Nested")
    
    # Test new flat format
    new_works = test_config_format("configs/config_simple.json", "New Simplified Flat")
    
    print(f"\n{'='*60}")
    print("🎯 Test Summary")
    print(f"{'='*60}")
    print(f"Legacy config (nested):    {'✅ PASS' if old_works else '❌ FAIL'}")
    print(f"New config (flat):         {'✅ PASS' if new_works else '❌ FAIL'}")
    print(f"Backwards compatibility:   {'✅ MAINTAINED' if old_works and new_works else '❌ BROKEN'}")
    
    if old_works and new_works:
        print(f"\n🎉 SUCCESS: Both config formats work!")
        print(f"   • Users can keep using their existing config.json files")
        print(f"   • Users can also switch to the new simplified format")
        print(f"   • The new system automatically detects and converts formats")
        print(f"\nTo run MuMDIA:")
        print(f"   python run.py configs/config.json        # Old format")
        print(f"   python run.py configs/config_simple.json # New format")
    else:
        print(f"\n❌ FAILURE: Config compatibility is broken!")

if __name__ == "__main__":
    main()
