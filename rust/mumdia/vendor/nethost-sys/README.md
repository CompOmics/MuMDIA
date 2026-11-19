# nethost-sys

[![CI](https://github.com/OpenByteDev/nethost-sys/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenByteDev/nethost-sys/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/nethost-sys.svg)](https://crates.io/crates/nethost-sys)
[![Documentation](https://docs.rs/nethost-sys/badge.svg)](https://docs.rs/nethost-sys)
[![dependency status](https://deps.rs/repo/github/openbytedev/nethost-sys/status.svg)](https://deps.rs/repo/github/openbytedev/nethost-sys)
[![MIT](https://img.shields.io/crates/l/nethost-sys.svg)](https://github.com/OpenByteDev/nethost-sys/blob/master/LICENSE)

<!-- cargo-sync-readme start -->

FFI bindings for [nethost](https://github.com/dotnet/runtime/blob/main/docs/design/features/host-components.md#components-of-the-hosting).

Supports automatically downloading the latest version of the [nethost](https://github.com/dotnet/runtime/blob/main/docs/design/features/host-components.md#components-of-the-hosting) library from [NuGet](https://www.nuget.org/packages/Microsoft.NETCore.DotNetHost/) with the `download-nuget` feature.

### Related crates
- [hostfxr-sys](https://crates.io/crates/hostfxr-sys) - bindings for the hostfxr library.
- [coreclr-hosting-shared](https://crates.io/crates/coreclr-hosting-shared) - shared bindings between this crate and [hostfxr-sys](https://crates.io/crates/hostfxr-sys).
- [netcorehost](https://crates.io/crates/netcorehost) - rusty wrapper over the nethost and hostfxr libraries.

### Additional Information
- [Hosting layer APIs](https://github.com/dotnet/core-setup/blob/master/Documentation/design-docs/hosting-layer-apis.md)
- [Native hosting](https://github.com/dotnet/core-setup/blob/master/Documentation/design-docs/native-hosting.md#runtime-properties)
- [Write a custom .NET Core host to control the .NET runtime from your native code](https://docs.microsoft.com/en-us/dotnet/core/tutorials/netcore-hosting)

### License
Licensed under the MIT license ([LICENSE](https://github.com/OpenByteDev/nethost-sys/blob/master/LICENSE) or http://opensource.org/licenses/MIT)

<!-- cargo-sync-readme end -->
