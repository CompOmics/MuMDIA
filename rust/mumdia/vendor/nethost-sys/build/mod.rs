use build_target::Os;

#[cfg(feature = "download-nuget")]
mod download;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "download-nuget")]
    download::download_nethost_from_nuget()?;

    // NOTE: for some reason we need the rustc argument here, but the link attribute in lib.rs for other os.
    // For more information see https://github.com/OpenByteDev/netcorehost/issues/2.
    if build_target::target_os() == Os::Windows {
        cargo_emit::rustc_link_lib!("libnethost");
        cargo_emit::rustc_link_lib!("advapi32");
    }

    Ok(())
}
