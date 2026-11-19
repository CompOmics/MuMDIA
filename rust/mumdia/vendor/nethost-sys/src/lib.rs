#![no_std]
#![warn(clippy::pedantic, clippy::cargo, unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::multiple_crate_versions,
    clippy::doc_markdown,
    non_camel_case_types,
    dead_code
)]

//! FFI bindings for [nethost](https://github.com/dotnet/runtime/blob/main/docs/design/features/host-components.md#components-of-the-hosting).
//!
//! Supports automatically downloading the latest version of the [nethost](https://github.com/dotnet/runtime/blob/main/docs/design/features/host-components.md#components-of-the-hosting) library from [NuGet](https://www.nuget.org/packages/Microsoft.NETCore.DotNetHost/) with the `download-nuget` feature.
//!
//! ## Related crates
//! - [hostfxr-sys](https://crates.io/crates/hostfxr-sys) - bindings for the hostfxr library.
//! - [coreclr-hosting-shared](https://crates.io/crates/coreclr-hosting-shared) - shared bindings between this crate and [hostfxr-sys](https://crates.io/crates/hostfxr-sys).
//! - [netcorehost](https://crates.io/crates/netcorehost) - rusty wrapper over the nethost and hostfxr libraries.
//!
//! ## Additional Information
//! - [Hosting layer APIs](https://github.com/dotnet/core-setup/blob/master/Documentation/design-docs/hosting-layer-apis.md)
//! - [Native hosting](https://github.com/dotnet/core-setup/blob/master/Documentation/design-docs/native-hosting.md#runtime-properties)
//! - [Write a custom .NET Core host to control the .NET runtime from your native code](https://docs.microsoft.com/en-us/dotnet/core/tutorials/netcore-hosting)
//!
//! ## License
//! Licensed under the MIT license ([LICENSE](https://github.com/OpenByteDev/nethost-sys/blob/master/LICENSE) or <http://opensource.org/licenses/MIT>)

use core::{mem, ptr};
use coreclr_hosting_shared::{char_t, size_t};

// for some reason we need the link attribute here for unix, but the rustc argument in build.rs for windows.
// #[cfg_attr(windows, link(name = "libnethost"))]
#[cfg_attr(unix, link(name = "nethost", kind = "static"))]
#[cfg_attr(
    all(unix, not(target_os = "macos")),
    link(name = "stdc++", kind = "dylib")
)]
#[cfg_attr(target_os = "macos", link(name = "c++", kind = "dylib"))]
extern "system" {
    /// Get the path to the `hostfxr` library.
    ///
    /// # Arguments
    /// * `buffer`:
    ///   Buffer that will be populated with the hostfxr path, including a null terminator.
    /// * `buffer_size`:
    ///    * \[in\] Size of buffer in [`char_t`] units.
    ///    * \[out\] Size of buffer used in [`char_t`] units. If the input value is too small
    ///      or `buffer` is [`null`](core::ptr::null()), this is populated with the minimum required size
    ///      in [`char_t`] units for a buffer to hold the hostfxr path
    ///
    /// * `get_hostfxr_parameters`:
    ///   Optional. Parameters that modify the behaviour for locating the hostfxr library.
    ///   If [`null`](core::ptr::null()), hostfxr is located using the enviroment variable or global registration
    ///
    /// # Return value
    ///  * `0` on success, otherwise failure
    ///  * `0x80008098` - `buffer` is too small ([`HostApiBufferTooSmall`](https://docs.rs/coreclr-hosting-shared/0.1.2/coreclr_hosting_shared/enum.StatusCode.html#variant.HostApiBufferTooSmall)
    ///
    /// # Remarks
    /// The full search for the hostfxr library is done on every call. To minimize the need
    /// to call this function multiple times, pass a large buffer (e.g. [`MAX_PATH`](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=cmd)).
    pub fn get_hostfxr_path(
        buffer: *mut char_t,
        buffer_size: *mut size_t,
        parameters: *const get_hostfxr_parameters,
    ) -> i32;
}

/// Parameters for [`get_hostfxr_path`].
#[repr(C)]
pub struct get_hostfxr_parameters {
    /// Size of the struct.
    /// This is used for versioning.
    pub size: size_t,
    /// Path to the compenent's assembly.
    /// If specified, hostfxr is located as if the `assembly_path` is the apphost
    pub assembly_path: *const char_t,
    /// Path to directory containing the dotnet executable.
    /// If specified, hostfxr is located as if an application is started using
    /// `dotnet app.dll`, which means it will be searched for under the `dotnet_root`
    /// path and the `assembly_path` is ignored.
    pub dotnet_root: *const char_t,
}

impl get_hostfxr_parameters {
    /// Creates a new instance of [`get_hostfxr_parameters`] with the given `dotnet_root`.
    /// The `size` field is set accordingly to the size of the struct and `assembly_path` to [`ptr::null()`].
    #[must_use]
    pub fn with_dotnet_root(dotnet_root: *const char_t) -> Self {
        Self {
            size: mem::size_of::<Self>(),
            assembly_path: ptr::null(),
            dotnet_root,
        }
    }
    /// Creates a new instance of [`get_hostfxr_parameters`] with the given `assembly_path`.
    /// The `size` field is set accordingly to the size of the struct and `dotnet_root` to [`ptr::null()`].
    #[must_use]
    pub fn with_assembly_path(assembly_path: *const char_t) -> Self {
        Self {
            size: mem::size_of::<Self>(),
            assembly_path,
            dotnet_root: ptr::null(),
        }
    }
}
