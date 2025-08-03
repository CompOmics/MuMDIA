#!/usr/bin/env python
"""
Example showing the more Pythonic usage of MuMDIA functions.

This demonstrates the transformation from the old, repetitive parameter style
to the new, clean dataclass-based configuration approach.
"""

from mumdia import main
from data_structures import PickleConfig, SpectraData
import polars as pl

def old_style_example():
    """The old way - repetitive and error-prone."""
    print("❌ OLD STYLE - Not Pythonic:")
    print("""
    main(
        df_fragment=df_fragment,
        df_psms=df_psms, 
        df_fragment_max=df_fragment_max,
        df_fragment_max_peptide=df_fragment_max_peptide,
        write_deeplc_pickle=True,
        write_ms2pip_pickle=False,
        write_correlation_pickles=True,
        read_deeplc_pickle=False,
        read_ms2pip_pickle=True,
        read_correlation_pickles=False,
        ms1_dict=ms1_data,
        ms2_to_ms1_dict=ms2_mapping,
        ms2_dict=ms2_data,
        config=config,
        deeplc_model=model
    )
    """)

def new_style_example():
    """The new way - clean and Pythonic."""
    print("✅ NEW STYLE - Pythonic and Clean:")
    print("""
    # Configure pickle settings clearly
    pickle_config = PickleConfig(
        write_deeplc=True,
        write_correlation=True,
        read_ms2pip=True
    )
    
    # Group related spectral data
    spectra_data = SpectraData(
        ms1_dict=ms1_data,
        ms2_to_ms1_dict=ms2_mapping,
        ms2_dict=ms2_data
    )
    
    # Clean function call
    main(
        df_fragment=df_fragment,
        df_psms=df_psms,
        df_fragment_max=df_fragment_max,
        df_fragment_max_peptide=df_fragment_max_peptide,
        pickle_config=pickle_config,
        spectra_data=spectra_data,
        config=config,
        deeplc_model=model
    )
    """)

def comparison_benefits():
    """Show the benefits of the new approach."""
    print("🎯 BENEFITS OF THE NEW APPROACH:")
    print("""
    1. DRY (Don't Repeat Yourself):
       - No more repetitive 'pickle' parameter names
       - Grouped related settings together
    
    2. Type Safety:
       - Dataclasses provide better type hints
       - IDE autocomplete and validation
    
    3. Logical Grouping:
       - Pickle settings in one place
       - Spectral data in another
       - Clear separation of concerns
    
    4. Extensibility:
       - Easy to add new pickle types
       - Easy to add new spectral data
       - No function signature changes needed
    
    5. Default Values:
       - Sensible defaults when not specified
       - No need to specify every parameter
    
    6. Immutable Configuration:
       - Dataclasses can be frozen
       - Prevents accidental modifications
    """)

def practical_examples():
    """Show practical usage patterns."""
    print("📚 PRACTICAL USAGE PATTERNS:")
    
    print("\n1. Cache Everything:")
    cache_all = PickleConfig(
        write_deeplc=True,
        write_ms2pip=True,
        write_correlation=True
    )
    
    print("\n2. Load from Cache:")
    load_cache = PickleConfig(
        read_deeplc=True,
        read_ms2pip=True,
        read_correlation=True
    )
    
    print("\n3. Development Mode (write and read):")
    dev_mode = PickleConfig(
        write_deeplc=True,
        write_ms2pip=True,
        read_deeplc=True,
        read_ms2pip=True
    )
    
    print("\n4. Production Mode (no caching):")
    prod_mode = PickleConfig()  # All False by default
    
    print("\n5. From Configuration File:")
    def from_config_file(config_dict):
        return PickleConfig(
            write_deeplc=config_dict.get('cache_deeplc', False),
            write_ms2pip=config_dict.get('cache_ms2pip', False),
            read_deeplc=config_dict.get('load_cache', False)
        )

if __name__ == "__main__":
    print("🚀 MuMDIA Pythonic Interface Demonstration\n")
    old_style_example()
    print("\n" + "="*60 + "\n")
    new_style_example()
    print("\n" + "="*60 + "\n")
    comparison_benefits()
    print("\n" + "="*60 + "\n")
    practical_examples()
    
    print("\n🎉 The new interface is much more maintainable and Pythonic!")
