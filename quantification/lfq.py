import os
import logging

import polars as pl
from tqdm import tqdm
from Bio import SeqIO
from pyteomics import proforma
import directlfq.config as lfqconfig
import directlfq.protein_intensity_estimation as lfqprot_estimation
import directlfq.utils as lfqutils
from utilities.logger import log_info


from scipy import integrate


def quantify_fragments(df_fragment, mokapot_results, config, output_dir: str = None):
    # read in fasta once for mapping peptides to proteins
    fasta_path = config["sage"]["database"]["fasta"]
    fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

    # extract basename of mzml file for column name
    mzml_filename = os.path.basename(config["sage"]["mzml_paths"][0]).split('.')[0]

    #filter df_fragment to peptides that survived mokapot 
    #TODO: also include charge 
    mokapot_peptides = pl.read_csv(mokapot_results[1], separator="\t")
    df_fragment_mokapot_filtered = df_fragment.filter(pl.col("peptide").is_in(mokapot_peptides["Peptide"]))

    results = []

    logging.info(f"Quantifying fragments")
 
    for (peptidoform, charge, fragment_name), df_fragment_mokapot_filtered_sub in tqdm(
            df_fragment_mokapot_filtered.group_by(["peptide", "charge", "fragment_name"])
        ):
        # extract proteins where peptide is substring of
        proteins = ';'.join([record.id.split('|')[-1] for record in fasta_dict.values() if peptidoform in record.seq])
        proforma_seq = proforma.parse(peptidoform)
        stripped_sequence = ''.join([x[0] for x in proforma_seq[0]])

        results.append({
            "Proteins": proteins,
            "ion": stripped_sequence + "_CHARGE_" + str(charge) + "_" + fragment_name,
            mzml_filename + "_Intensity_peak": quantify_fragment_peak_intensity(df_fragment_mokapot_filtered_sub, margin=False),
            mzml_filename + "_Intensity_peak_margin": quantify_fragment_peak_intensity(df_fragment_mokapot_filtered_sub, margin=True),
            mzml_filename + "_Intensity_integrated": quantify_fragment_integrated_intensity(df_fragment_mokapot_filtered_sub,  margin=False),
            mzml_filename + "_Intensity_integrated_margin": quantify_fragment_integrated_intensity(df_fragment_mokapot_filtered_sub, margin=True),
        })
        # break  # for debugging, remove break for full run
    df_results_fragment = pl.DataFrame(results)
    df_results_fragment.write_csv(os.path.join(output_dir, "fragment_level_intensities.csv"))   

    # TODO: run directLFQ on fragment level intensities 

    # copy pasterino from alphadia:
    log_info.info("Performing label-free quantification using directLFQ")

    _intensity_df = df_results_fragment.select("Proteins", "ion", mzml_filename + "_Intensity_integrated_margin")

    lfqconfig.set_global_protein_and_ion_id(protein_id='Proteins', quant_id=mzml_filename + "_Intensity_integrated_margin")
    lfqconfig.set_compile_normalized_ion_table(
        compile_normalized_ion_table=False
    )  # save compute time by avoiding the creation of a normalized ion table
    lfqconfig.check_wether_to_copy_numpy_arrays_derived_from_pandas()  # avoid read-only pandas bug on linux if applicable
    lfqconfig.set_log_processed_proteins(
        log_processed_proteins=True
    )  # here you can chose wether to log the processed proteins or not

    _intensity_df.sort_values(by='_Intensity_integrated_margin', inplace=True, ignore_index=True)

    lfq_df = lfqutils.index_and_log_transform_input_df(_intensity_df)
    lfq_df = lfqutils.remove_allnan_rows_input_df(lfq_df)

    protein_df, _ = lfqprot_estimation.estimate_protein_intensities(
        lfq_df,
        min_nonan=1,
        num_samples_quadratic=50,
    )

    protein_df.to_csv(os.path.join(output_dir, "protein_level_intensities_directlfq_out.csv"))

    return # df_results_fragment, protein_df


def quantify_precursors(df_fragment, mokapot_results, config, output_dir):
    # extract basename of mzml file for column name
    mzml_filename = os.path.basename(config["sage"]["mzml_paths"][0]).split('.')[0]

    #filter df_fragment to peptides that survived mokapot 
    #TODO: also include charge 
    mokapot_peptides = pl.read_csv(mokapot_results[1], separator="\t")
    df_fragment_mokapot_filtered = df_fragment.filter(pl.col("peptide").is_in(mokapot_peptides["Peptide"]))

    results = []

    logging.info(f"Quantifying peptides")

    #aggregate fragment to precursor level
    for (peptidoform, stripped_sequence, charge, proteins), df_fragment_mokapot_filtered_sub in tqdm(
            df_fragment_mokapot_filtered.group_by(["peptide", "stripped_peptide", "charge", "proteins"])
        ):

        # extract protein id e.g. from rev_sp|P58107|EPIPL_HUMAN|2501|2518;rev_sp|P58107|EPIPL_HUMAN|2501|2518
        proteins = ';'.join(proteinstring.split('|')[2] for proteinstring in proteins.split(';'))

        results.append({
            "Sequence": stripped_sequence,
            "Proteins": proteins,
            "Charge": charge,
            "Modified sequence": peptidoform,
            mzml_filename + '_baseline': quantify_precursor_baseline(df_fragment_mokapot_filtered_sub),
            mzml_filename + '_baseline_margin': quantify_precursor_baseline(df_fragment_mokapot_filtered_sub, margin=True),
            mzml_filename + '_integrated': quantify_precursor_integrated(df_fragment_mokapot_filtered_sub),
            mzml_filename + '_integrated_margin': quantify_precursor_integrated(df_fragment_mokapot_filtered_sub, margin=True)

        })
        # break  # for debugging, remove break for full run

    logging.info(f"Writing quantification results to file in proteobench format")
    # write to file in proteobench format
    df_results = pl.DataFrame(results)
    df_results.write_csv(os.path.join(output_dir, mzml_filename + "_quantification_results.csv"))

    return # df_results

def quantify_precursor_baseline(df_fragment_sub_peptidoform, margin: bool = False):
    # if margin is True, only consider fragments that are within the RT margins
    if margin:
        df_fragment_sub_peptidoform = _filter_margin_rt(df_fragment_sub_peptidoform)

    # Sum up top 3 most intense fragments over entire RT
    result = df_fragment_sub_peptidoform.group_by("fragment_name").agg(
        pl.col("fragment_intensity").sum().alias("fragment_intensity_sum")
    )

    # sort by fragment_intensity_sum, sum the top 3 (if available)
    top_n_sum = (
        result.sort("fragment_intensity_sum", descending=True)
            .head(3)  # adjust N here
            .select(pl.col("fragment_intensity_sum").sum()).item()
    )

    return top_n_sum

def quantify_precursor_integrated(df_fragment_sub_peptidoform, margin: bool = False):
    # integrate all fragments over RT, then sum top 3 most intense fragments
    if margin:
        df_fragment_sub_peptidoform = _filter_margin_rt(df_fragment_sub_peptidoform)

    fragment_intensities = []
    for fragment_name, df_fragment_sub_peptidoform in tqdm(
            df_fragment_sub_peptidoform.group_by(["fragment_name"])
        ):
        fragment_intensities.append(quantify_fragment_integrated_intensity(df_fragment_sub_peptidoform, margin=margin))

    if len(fragment_intensities) > 0:
        # sum top 3 most intense fragments
        return sum(sorted(fragment_intensities, reverse=True)[:3])  # adjust N here
    else:
        return 0

def quantify_fragment_peak_intensity(df_fragment_ion_psms, margin: bool = False):
    # return highest intensity of fragment over RT
    if margin:
        df_fragment_ion_psms = _filter_margin_rt(df_fragment_ion_psms)
    
    return df_fragment_ion_psms["fragment_intensity"].max()

def quantify_fragment_integrated_intensity(df_fragment_ion_psms, margin: bool = False):
    # integrate fragment intensities over RT
    if margin:
        df_fragment_ion_psms = _filter_margin_rt(df_fragment_ion_psms)

    # for fragments only measured at one time point, return that intensity
    if df_fragment_ion_psms.shape[0] == 1:
        return df_fragment_ion_psms["fragment_intensity"].item()

    # approximate integration using trapezoidal rule
    aoc = integrate.trapezoid(
        y=df_fragment_ion_psms["fragment_intensity"].to_numpy(), 
        x=df_fragment_ion_psms["rt"].to_numpy()
    )
    return aoc

def _filter_margin_rt(df):
    if df['psm_id'].n_unique() > 1:
        df = df.filter(
            (pl.col("rt") >= pl.col("rt_lower_margin")) &
            (pl.col("rt") <= pl.col("rt_upper_margin"))
        )
    return df