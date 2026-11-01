//! `mumdia build-library` (issue #233): a FASTA library built without spectra, stored in
//! the library cache under the key a FASTA search computes, and described in
//! `library.json`. The search side of the round trip, `run --fasta` finding the stored
//! library, runs end to end in `ci/smoke.sh`, which has spectra.

use mumdia::stages::library_build;
use mumdia_core::config::{Config, LibraryIrt};
use mumdia_core::manifest::Manifest;

fn fixture_fasta() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../test_data/fixture.fasta"
    )
    .to_string()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "mumdia_build_library_{}_{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Native predictors, the library cache in `cache`, and no DeepLC discovery, so the test
/// keys the same library on any machine.
fn native_config(cache: &std::path::Path) -> Config {
    let mut cfg = Config::default();
    cfg.predict_frag.library_cache = Some(cache.to_string_lossy().into_owned());
    cfg.rt_im_train.library_irt = LibraryIrt::Library;
    cfg
}

fn read_json(path: &std::path::Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn a_library_built_without_spectra_is_stored_described_and_found_again() {
    let root = scratch("roundtrip");
    let cache = root.join("cache");
    let cfg = native_config(&cache);
    let fasta = fixture_fasta();
    let first = root.join("first");
    let first_s = first.to_string_lossy().into_owned();

    let info = library_build::run(library_build::BuildLibraryParams {
        config: &cfg,
        config_path: None,
        fasta: &fasta,
        out_dir: &first_s,
        ion_mobility: false,
    })
    .unwrap();
    assert_eq!(info, read_json(&first.join(library_build::LIBRARY_JSON)));
    assert_eq!(info["origin"], "built");
    assert_eq!(info["cache"]["stored"], true);
    let key = info["cache"]["key"].as_str().unwrap().to_string();
    assert_eq!(key.len(), 24);
    let rows = info["precursors"]["rows"].as_u64().unwrap();
    assert!(rows > 0);
    assert!(info["precursors"]["targets"].as_u64().unwrap() > 0);
    assert!(info["precursors"]["decoys"].as_u64().unwrap() > 0);
    assert!(info["fragments"]["rows"].as_u64().unwrap() > rows);
    assert_eq!(
        info["models"],
        serde_json::json!({"fragment": "native", "rt": "native"})
    );
    assert_eq!(info["predicts_ion_mobility"], false);
    assert_eq!(info["reuse"]["library_irt"], "library");
    assert!(info["reuse"]["library"]
        .as_str()
        .unwrap()
        .contains("--lib-precursors"));
    for f in [
        "fragment_library_precursors.parquet",
        "fragment_library_precursors.parquet.report.json",
        "fragment_library_fragments.parquet",
        "fragment_library_fragments.parquet.report.json",
        "manifest.json",
    ] {
        assert!(first.join(f).is_file(), "{f} was not written");
    }
    let man = read_json(&first.join("manifest.json"));
    assert_eq!(
        man["artifacts"]["fragment_library_precursors"]["producing_stage"],
        "predict-frag"
    );
    assert!(man["inputs"]["fasta"]["content_hash"].is_string());

    // A second build of the same FASTA and configuration is the stored library, byte for
    // byte, and skips the digest.
    let second = root.join("second");
    let second_s = second.to_string_lossy().into_owned();
    let again = library_build::run(library_build::BuildLibraryParams {
        config: &cfg,
        config_path: None,
        fasta: &fasta,
        out_dir: &second_s,
        ion_mobility: false,
    })
    .unwrap();
    assert_eq!(again["origin"], "library-cache");
    assert_eq!(again["cache"]["key"], key.as_str());
    assert!(!second.join("peptides.parquet").exists());
    for f in [
        "fragment_library_precursors.parquet",
        "fragment_library_fragments.parquet",
    ] {
        assert_eq!(
            std::fs::read(first.join(f)).unwrap(),
            std::fs::read(second.join(f)).unwrap(),
            "{f} differs between the build and the restore"
        );
    }
    let man = read_json(&second.join("manifest.json"));
    assert_eq!(
        man["artifacts"]["fragment_library_precursors"]["producing_stage"],
        "library-cache"
    );

    // The branch `run` takes for a FASTA without a prefilter finds the same entry: it is
    // the same function, keyed with what `run` passes for spectra without 1/K0.
    let run_dir = root.join("run");
    std::fs::create_dir_all(&run_dir).unwrap();
    let mut resolved = cfg.clone();
    resolved.predict_frag.sidecar_script_dir =
        mumdia::python::resolve_script_dir(&resolved.predict_frag.sidecar_script_dir, None);
    mumdia::python::resolve(&mut resolved).unwrap();
    resolved.resolve_im_features(false);
    let ch = mumdia_io::hash::blake3_str(&resolved.canonical_json());
    let mut man = Manifest::new(resolved.canonical_json(), ch.clone());
    let engine = (man.mumdia_version.clone(), man.git_sha.clone());
    let mut pre = mumdia::prestage::PreStageTimer::start("run");
    let lib = library_build::build(
        &library_build::BuildParams {
            cfg: &resolved,
            config_hash: &ch,
            fasta: &fasta,
            out_dir: &run_dir.to_string_lossy(),
            rt_placeholder: false,
            run_has_im: false,
            engine: (&engine.0, &engine.1),
            use_cache: true,
            caller: "run",
        },
        Some(&mut man),
        &mut pre,
        None,
    )
    .unwrap();
    assert_eq!(lib.origin, library_build::Origin::LibraryCache);
    assert_eq!(lib.cache.as_ref().unwrap().key(), key);
    assert_eq!(
        std::fs::read(&lib.precursors).unwrap(),
        std::fs::read(first.join("fragment_library_precursors.parquet")).unwrap()
    );

    // `--ion-mobility on` with nothing predicting 1/K0 is the same library.
    let im = root.join("im");
    let on = library_build::run(library_build::BuildLibraryParams {
        config: &cfg,
        config_path: None,
        fasta: &fasta,
        out_dir: &im.to_string_lossy(),
        ion_mobility: true,
    })
    .unwrap();
    assert_eq!(on["cache"]["key"], key.as_str());
    assert_eq!(on["ion_mobility"], true);
    assert_eq!(on["origin"], "library-cache");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_prefilter_before_prediction_is_refused_before_anything_is_written() {
    let root = scratch("prescreen");
    for set in [
        |c: &mut Config| c.prescreen.tag_prefilter = true,
        |c: &mut Config| c.prescreen.score_before_prediction = true,
    ] {
        let mut cfg = native_config(&root.join("cache"));
        set(&mut cfg);
        let out = root.join("out");
        let e = library_build::run(library_build::BuildLibraryParams {
            config: &cfg,
            config_path: None,
            fasta: &fixture_fasta(),
            out_dir: &out.to_string_lossy(),
            ion_mobility: false,
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("prescreen.") && e.contains("spectra"), "{e}");
        assert!(!out.exists(), "a refused build wrote its output directory");
    }
    let e = library_build::run(library_build::BuildLibraryParams {
        config: &native_config(&root.join("cache")),
        config_path: None,
        fasta: &root.join("absent.fasta").to_string_lossy(),
        out_dir: &root.join("out").to_string_lossy(),
        ion_mobility: false,
    })
    .unwrap_err()
    .to_string();
    assert!(e.contains("--fasta not found"), "{e}");
    let _ = std::fs::remove_dir_all(&root);
}
