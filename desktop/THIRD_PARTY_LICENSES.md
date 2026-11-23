# Third-party licences

GENERATED FILE. Do not edit. `ci/gen_third_party_licenses.py` writes it from
`desktop/Cargo.lock` (`--desktop`), and CI fails when it is stale.

MuMDIA's own licence is Apache-2.0; see `LICENSE`. The desktop application
`mumdia-console` is statically linked, so it contains the crates listed below
and this file accompanies it in the desktop installers. The engine the
installers bundle beside it has its own list, `THIRD_PARTY_LICENSES.md`.

## Obligations

456 third-party crates. Every one declares an SPDX expression; none is
unspecified.

The following copyleft identifiers appear, in each case as one arm of a
disjunction whose permissive arm MuMDIA relies on instead (the arm is named
per crate in the table below):

- `LGPL-2.1-or-later`
- `MPL-2.0`

Except for these crates, whose only licence is the one named:

- `cssparser` 0.37.0 (MPL-2.0): https://github.com/servo/rust-cssparser
- `cssparser-macros` 0.7.1 (MPL-2.0): https://github.com/servo/rust-cssparser
- `dtoa-short` 0.3.5 (MPL-2.0): https://github.com/upsuper/dtoa-short
- `option-ext` 0.2.0 (MPL-2.0): https://github.com/soc/option-ext.git
- `selectors` 0.38.0 (MPL-2.0): https://github.com/servo/stylo

MPL-2.0 is file-level copyleft. MuMDIA uses these crates unmodified, and
their Source Code Form (MPL-2.0 section 3.2) is available at exactly the
versions listed, from crates.io (`https://crates.io/crates/<name>/<version>`)
and from the repositories above. The licence text is at
https://mozilla.org/MPL/2.0/.

Licence identifiers by crate count:

| SPDX identifier | crates |
|---|---|
| `MIT` | 417 |
| `Apache-2.0` | 304 |
| `Zlib` | 23 |
| `Unicode-3.0` | 19 |
| `Unlicense` | 11 |
| `BSD-3-Clause` | 7 |
| `ISC` | 5 |
| `MPL-2.0` | 5 |
| `Apache-2.0 WITH LLVM-exception` | 4 |
| `CC0-1.0` | 2 |
| `LGPL-2.1-or-later` | 2 |
| `MIT-0` | 2 |
| `0BSD` | 1 |
| `CDLA-Permissive-2.0` | 1 |

## Crates

| crate | version | licence | relied-on arm | source |
|---|---|---|---|---|
| `adler2` | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/oyvindln/adler2](https://github.com/oyvindln/adler2) |
| `aho-corasick` | 1.1.5 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/aho-corasick](https://github.com/BurntSushi/aho-corasick) |
| `alloc-no-stdlib` | 3.0.0 | BSD-3-Clause | - | [https://github.com/dropbox/rust-alloc-no-stdlib](https://github.com/dropbox/rust-alloc-no-stdlib) |
| `alloc-stdlib` | 0.3.0 | BSD-3-Clause | - | [https://github.com/dropbox/rust-alloc-no-stdlib](https://github.com/dropbox/rust-alloc-no-stdlib) |
| `android_system_properties` | 0.1.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/nical/android_system_properties](https://github.com/nical/android_system_properties) |
| `anyhow` | 1.0.104 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/anyhow](https://github.com/dtolnay/anyhow) |
| `atk` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `atk-sys` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `atomic-waker` | 1.1.2 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/smol-rs/atomic-waker](https://github.com/smol-rs/atomic-waker) |
| `autocfg` | 1.5.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/cuviper/autocfg](https://github.com/cuviper/autocfg) |
| `base64` | 0.21.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/marshallpierce/rust-base64](https://github.com/marshallpierce/rust-base64) |
| `base64` | 0.22.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/marshallpierce/rust-base64](https://github.com/marshallpierce/rust-base64) |
| `base64` | 0.23.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/marshallpierce/rust-base64](https://github.com/marshallpierce/rust-base64) |
| `bit-set` | 0.8.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/contain-rs/bit-set](https://github.com/contain-rs/bit-set) |
| `bit-vec` | 0.8.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/contain-rs/bit-vec](https://github.com/contain-rs/bit-vec) |
| `bitflags` | 1.3.2 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/bitflags/bitflags](https://github.com/bitflags/bitflags) |
| `bitflags` | 2.13.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/bitflags/bitflags](https://github.com/bitflags/bitflags) |
| `block-buffer` | 0.10.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `block-buffer` | 0.12.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `block2` | 0.6.2 | MIT | - | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `brotli` | 9.0.0 | BSD-3-Clause AND MIT | - | [https://github.com/dropbox/rust-brotli](https://github.com/dropbox/rust-brotli) |
| `brotli-decompressor` | 6.0.1 | BSD-3-Clause/MIT | `MIT` | [https://github.com/dropbox/rust-brotli-decompressor](https://github.com/dropbox/rust-brotli-decompressor) |
| `bs58` | 0.5.1 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/Nullus157/bs58-rs](https://github.com/Nullus157/bs58-rs) |
| `bumpalo` | 3.20.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/fitzgen/bumpalo](https://github.com/fitzgen/bumpalo) |
| `bytemuck` | 1.25.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/Lokathor/bytemuck](https://github.com/Lokathor/bytemuck) |
| `byteorder` | 1.5.0 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/byteorder](https://github.com/BurntSushi/byteorder) |
| `bytes` | 1.12.1 | MIT | - | [https://github.com/tokio-rs/bytes](https://github.com/tokio-rs/bytes) |
| `cairo-rs` | 0.18.5 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `cairo-sys-rs` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `camino` | 1.2.5 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/camino-rs/camino](https://github.com/camino-rs/camino) |
| `cargo-platform` | 0.1.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/cargo](https://github.com/rust-lang/cargo) |
| `cargo_metadata` | 0.19.2 | MIT | - | [https://github.com/oli-obk/cargo_metadata](https://github.com/oli-obk/cargo_metadata) |
| `cargo_toml` | 1.0.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://gitlab.com/lib.rs/cargo_toml](https://gitlab.com/lib.rs/cargo_toml) |
| `cc` | 1.4.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/cc-rs](https://github.com/rust-lang/cc-rs) |
| `cesu8` | 1.1.0 | Apache-2.0/MIT | `Apache-2.0` | [https://github.com/emk/cesu8-rs](https://github.com/emk/cesu8-rs) |
| `cfb` | 0.14.0 | MIT | - | [https://github.com/mdsteele/rust-cfb](https://github.com/mdsteele/rust-cfb) |
| `cfg-expr` | 0.15.8 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/EmbarkStudios/cfg-expr](https://github.com/EmbarkStudios/cfg-expr) |
| `cfg-if` | 1.0.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/cfg-if](https://github.com/rust-lang/cfg-if) |
| `cfg_aliases` | 0.2.2 | MIT | - | [https://github.com/katharostech/cfg_aliases](https://github.com/katharostech/cfg_aliases) |
| `chrono` | 0.4.45 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/chronotope/chrono](https://github.com/chronotope/chrono) |
| `combine` | 4.6.8 | MIT | - | [https://github.com/Marwes/combine](https://github.com/Marwes/combine) |
| `const-oid` | 0.10.2 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/RustCrypto/formats](https://github.com/RustCrypto/formats) |
| `constant_time_eq` | 0.6.1 | CC0-1.0 OR MIT-0 OR Apache-2.0 | `Apache-2.0` | [https://codeberg.org/cesarb/constant_time_eq](https://codeberg.org/cesarb/constant_time_eq) |
| `cookie` | 0.18.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/SergioBenitez/cookie-rs](https://github.com/SergioBenitez/cookie-rs) |
| `core-foundation` | 0.10.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/core-foundation-rs](https://github.com/servo/core-foundation-rs) |
| `core-foundation-sys` | 0.8.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/core-foundation-rs](https://github.com/servo/core-foundation-rs) |
| `core-graphics` | 0.25.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/core-foundation-rs](https://github.com/servo/core-foundation-rs) |
| `core-graphics-types` | 0.2.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/core-foundation-rs](https://github.com/servo/core-foundation-rs) |
| `cpufeatures` | 0.2.17 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `cpufeatures` | 0.3.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `crc32fast` | 1.5.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/srijs/rust-crc32fast](https://github.com/srijs/rust-crc32fast) |
| `crossbeam-channel` | 0.5.16 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) |
| `crossbeam-utils` | 0.8.22 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) |
| `crypto-common` | 0.1.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/traits](https://github.com/RustCrypto/traits) |
| `crypto-common` | 0.2.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/traits](https://github.com/RustCrypto/traits) |
| `cssparser` | 0.37.0 | MPL-2.0 | - | [https://github.com/servo/rust-cssparser](https://github.com/servo/rust-cssparser) |
| `cssparser-macros` | 0.7.1 | MPL-2.0 | - | [https://github.com/servo/rust-cssparser](https://github.com/servo/rust-cssparser) |
| `ctor` | 1.0.13 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/mmastrac/linktime](https://github.com/mmastrac/linktime) |
| `darling` | 0.23.0 | MIT | - | [https://github.com/TedDriggs/darling](https://github.com/TedDriggs/darling) |
| `darling_core` | 0.23.0 | MIT | - | [https://github.com/TedDriggs/darling](https://github.com/TedDriggs/darling) |
| `darling_macro` | 0.23.0 | MIT | - | [https://github.com/TedDriggs/darling](https://github.com/TedDriggs/darling) |
| `dbus` | 0.9.12 | Apache-2.0/MIT | `Apache-2.0` | [https://github.com/diwic/dbus-rs](https://github.com/diwic/dbus-rs) |
| `defmt` | 1.1.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/knurling-rs/defmt](https://github.com/knurling-rs/defmt) |
| `defmt-macros` | 1.1.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/knurling-rs/defmt](https://github.com/knurling-rs/defmt) |
| `defmt-parser` | 1.0.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/knurling-rs/defmt](https://github.com/knurling-rs/defmt) |
| `deranged` | 0.5.8 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/jhpratt/deranged](https://github.com/jhpratt/deranged) |
| `derive_more` | 2.1.1 | MIT | - | [https://github.com/JelteF/derive_more](https://github.com/JelteF/derive_more) |
| `derive_more-impl` | 2.1.1 | MIT | - | [https://github.com/JelteF/derive_more](https://github.com/JelteF/derive_more) |
| `digest` | 0.10.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/traits](https://github.com/RustCrypto/traits) |
| `digest` | 0.11.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/traits](https://github.com/RustCrypto/traits) |
| `dirs` | 7.0.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://codeberg.org/dirs/dirs-rs](https://codeberg.org/dirs/dirs-rs) |
| `dirs-sys` | 0.5.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dirs-dev/dirs-sys-rs](https://github.com/dirs-dev/dirs-sys-rs) |
| `dispatch2` | 0.3.1 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `displaydoc` | 0.2.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/yaahc/displaydoc](https://github.com/yaahc/displaydoc) |
| `dlopen2` | 0.8.2 | MIT | - | [https://github.com/OpenByteDev/dlopen2](https://github.com/OpenByteDev/dlopen2) |
| `dlopen2_derive` | 0.4.3 | MIT | - | [https://github.com/OpenByteDev/dlopen2](https://github.com/OpenByteDev/dlopen2) |
| `dom_query` | 0.28.0 | MIT | - | [https://github.com/niklak/dom_query](https://github.com/niklak/dom_query) |
| `dpi` | 0.1.2 | Apache-2.0 AND MIT | - | [https://github.com/rust-windowing/winit](https://github.com/rust-windowing/winit) |
| `dtoa` | 1.0.11 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/dtoa](https://github.com/dtolnay/dtoa) |
| `dtoa-short` | 0.3.5 | MPL-2.0 | - | [https://github.com/upsuper/dtoa-short](https://github.com/upsuper/dtoa-short) |
| `dunce` | 1.0.5 | CC0-1.0 OR MIT-0 OR Apache-2.0 | `Apache-2.0` | [https://gitlab.com/kornelski/dunce](https://gitlab.com/kornelski/dunce) |
| `dyn-clone` | 1.0.20 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/dyn-clone](https://github.com/dtolnay/dyn-clone) |
| `embed-resource` | 3.0.11 | MIT | - | [https://github.com/nabijaczleweli/rust-embed-resource](https://github.com/nabijaczleweli/rust-embed-resource) |
| `embed_plist` | 1.2.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/nvzqz/embed-plist-rs](https://github.com/nvzqz/embed-plist-rs) |
| `equivalent` | 1.0.2 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/indexmap-rs/equivalent](https://github.com/indexmap-rs/equivalent) |
| `erased-serde` | 0.4.10 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/erased-serde](https://github.com/dtolnay/erased-serde) |
| `fallible-iterator` | 0.3.0 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/sfackler/rust-fallible-iterator](https://github.com/sfackler/rust-fallible-iterator) |
| `fallible-streaming-iterator` | 0.1.9 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/sfackler/fallible-streaming-iterator](https://github.com/sfackler/fallible-streaming-iterator) |
| `fastrand` | 2.5.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/smol-rs/fastrand](https://github.com/smol-rs/fastrand) |
| `fdeflate` | 0.3.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/image-rs/fdeflate](https://github.com/image-rs/fdeflate) |
| `field-offset` | 0.3.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/Diggsey/rust-field-offset](https://github.com/Diggsey/rust-field-offset) |
| `find-msvc-tools` | 0.1.11 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/cc-rs](https://github.com/rust-lang/cc-rs) |
| `flate2` | 1.1.10 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/flate2-rs](https://github.com/rust-lang/flate2-rs) |
| `fnv` | 1.0.7 | Apache-2.0 / MIT | `Apache-2.0` | [https://github.com/servo/rust-fnv](https://github.com/servo/rust-fnv) |
| `foldhash` | 0.2.0 | Zlib | - | [https://github.com/orlp/foldhash](https://github.com/orlp/foldhash) |
| `foreign-types` | 0.5.0 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/sfackler/foreign-types](https://github.com/sfackler/foreign-types) |
| `foreign-types-macros` | 0.2.4 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/sfackler/foreign-types](https://github.com/sfackler/foreign-types) |
| `foreign-types-shared` | 0.3.1 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/sfackler/foreign-types](https://github.com/sfackler/foreign-types) |
| `form_urlencoded` | 1.2.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/rust-url](https://github.com/servo/rust-url) |
| `futures-channel` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-core` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-executor` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-io` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-macro` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-sink` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-task` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-util` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `gdk` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `gdk-pixbuf` | 0.18.5 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `gdk-pixbuf-sys` | 0.18.0 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `gdk-sys` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `gdkwayland-sys` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `gdkx11` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `gdkx11-sys` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `generic-array` | 0.14.7 | MIT | - | [https://github.com/fizyk20/generic-array.git](https://github.com/fizyk20/generic-array.git) |
| `getrandom` | 0.2.17 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-random/getrandom](https://github.com/rust-random/getrandom) |
| `getrandom` | 0.3.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-random/getrandom](https://github.com/rust-random/getrandom) |
| `getrandom` | 0.4.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-random/getrandom](https://github.com/rust-random/getrandom) |
| `gio` | 0.18.4 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `gio-sys` | 0.18.1 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `glib` | 0.18.5 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `glib-macros` | 0.18.5 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `glib-sys` | 0.18.1 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `glob` | 0.3.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/glob](https://github.com/rust-lang/glob) |
| `gobject-sys` | 0.18.0 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `gtk` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `gtk-sys` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `gtk3-macros` | 0.18.2 | MIT | - | [https://github.com/gtk-rs/gtk3-rs](https://github.com/gtk-rs/gtk3-rs) |
| `hashbrown` | 0.12.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) |
| `hashbrown` | 0.17.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) |
| `hashlink` | 0.12.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/djc/hashlink](https://github.com/djc/hashlink) |
| `heck` | 0.4.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/withoutboats/heck](https://github.com/withoutboats/heck) |
| `heck` | 0.5.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/withoutboats/heck](https://github.com/withoutboats/heck) |
| `hex` | 0.4.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/KokaKiwi/rust-hex](https://github.com/KokaKiwi/rust-hex) |
| `html5ever` | 0.39.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/html5ever](https://github.com/servo/html5ever) |
| `http` | 1.5.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/hyperium/http](https://github.com/hyperium/http) |
| `http-body` | 1.1.0 | MIT | - | [https://github.com/hyperium/http-body](https://github.com/hyperium/http-body) |
| `http-body-util` | 0.1.5 | MIT | - | [https://github.com/hyperium/http-body](https://github.com/hyperium/http-body) |
| `httparse` | 1.10.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/seanmonstar/httparse](https://github.com/seanmonstar/httparse) |
| `hybrid-array` | 0.4.15 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/hybrid-array](https://github.com/RustCrypto/hybrid-array) |
| `hyper` | 1.11.1 | MIT | - | [https://github.com/hyperium/hyper](https://github.com/hyperium/hyper) |
| `hyper-util` | 0.1.20 | MIT | - | [https://github.com/hyperium/hyper-util](https://github.com/hyperium/hyper-util) |
| `iana-time-zone` | 0.1.65 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/strawlab/iana-time-zone](https://github.com/strawlab/iana-time-zone) |
| `iana-time-zone-haiku` | 0.1.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/strawlab/iana-time-zone](https://github.com/strawlab/iana-time-zone) |
| `ico` | 0.5.0 | MIT | - | [https://github.com/mdsteele/rust-ico](https://github.com/mdsteele/rust-ico) |
| `icu_collections` | 2.3.0 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `icu_locale_core` | 2.3.0 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `icu_normalizer` | 2.3.0 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `icu_normalizer_data` | 2.3.0 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `icu_properties` | 2.3.0 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `icu_properties_data` | 2.3.0 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `icu_provider` | 2.3.1 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `ident_case` | 1.0.1 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/TedDriggs/ident_case](https://github.com/TedDriggs/ident_case) |
| `idna` | 1.1.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/rust-url/](https://github.com/servo/rust-url/) |
| `idna_adapter` | 1.2.2 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/hsivonen/idna_adapter](https://github.com/hsivonen/idna_adapter) |
| `indexmap` | 1.9.3 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/bluss/indexmap](https://github.com/bluss/indexmap) |
| `indexmap` | 2.14.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/indexmap-rs/indexmap](https://github.com/indexmap-rs/indexmap) |
| `infer` | 0.22.0 | MIT | - | [https://github.com/bojand/infer](https://github.com/bojand/infer) |
| `ipnet` | 2.12.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/krisprice/ipnet](https://github.com/krisprice/ipnet) |
| `itoa` | 1.0.18 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/itoa](https://github.com/dtolnay/itoa) |
| `javascriptcore-rs` | 1.1.2 | MIT | - | [https://github.com/tauri-apps/javascriptcore-rs](https://github.com/tauri-apps/javascriptcore-rs) |
| `javascriptcore-rs-sys` | 1.1.1 | MIT | - | [https://github.com/tauri-apps/javascriptcore-rs](https://github.com/tauri-apps/javascriptcore-rs) |
| `jiff` | 0.2.35 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/jiff](https://github.com/BurntSushi/jiff) |
| `jiff-core` | 0.1.0 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/jiff](https://github.com/BurntSushi/jiff) |
| `jiff-static` | 0.2.35 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/jiff](https://github.com/BurntSushi/jiff) |
| `jiff-tzdb` | 0.1.8 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/jiff](https://github.com/BurntSushi/jiff) |
| `jiff-tzdb-platform` | 0.1.3 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/jiff](https://github.com/BurntSushi/jiff) |
| `jni` | 0.21.1 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/jni-rs/jni-rs](https://github.com/jni-rs/jni-rs) |
| `jni-sys` | 0.3.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/jni-rs/jni-sys](https://github.com/jni-rs/jni-sys) |
| `jni-sys` | 0.4.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/jni-rs/jni-sys](https://github.com/jni-rs/jni-sys) |
| `jni-sys-macros` | 0.4.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/jni-rs/jni-sys](https://github.com/jni-rs/jni-sys) |
| `js-sys` | 0.3.104 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/js-sys](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/js-sys) |
| `json-patch` | 4.2.0 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/idubrov/json-patch](https://github.com/idubrov/json-patch) |
| `jsonptr` | 0.7.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/chanced/jsonptr](https://github.com/chanced/jsonptr) |
| `keyboard-types` | 0.8.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-windowing/keyboard-types](https://github.com/rust-windowing/keyboard-types) |
| `libappindicator` | 0.9.0 | Apache-2.0 OR MIT | `Apache-2.0` | - |
| `libappindicator-sys` | 0.9.0 | Apache-2.0 OR MIT | `Apache-2.0` | - |
| `libc` | 0.2.189 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/libc](https://github.com/rust-lang/libc) |
| `libdbus-sys` | 0.2.7 | Apache-2.0/MIT | `Apache-2.0` | [https://github.com/diwic/dbus-rs](https://github.com/diwic/dbus-rs) |
| `libloading` | 0.7.4 | ISC | - | [https://github.com/nagisa/rust_libloading/](https://github.com/nagisa/rust_libloading/) |
| `libredox` | 0.1.21 | MIT | - | [https://gitlab.redox-os.org/redox-os/libredox.git](https://gitlab.redox-os.org/redox-os/libredox.git) |
| `libsqlite3-sys` | 0.40.0 | MIT | - | [https://github.com/rusqlite/rusqlite](https://github.com/rusqlite/rusqlite) |
| `litemap` | 0.8.3 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `lock_api` | 0.4.14 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/Amanieu/parking_lot](https://github.com/Amanieu/parking_lot) |
| `log` | 0.4.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/log](https://github.com/rust-lang/log) |
| `markup5ever` | 0.39.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/html5ever](https://github.com/servo/html5ever) |
| `memchr` | 2.8.3 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/memchr](https://github.com/BurntSushi/memchr) |
| `memoffset` | 0.9.1 | MIT | - | [https://github.com/Gilnaa/memoffset](https://github.com/Gilnaa/memoffset) |
| `mime` | 0.3.17 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/hyperium/mime](https://github.com/hyperium/mime) |
| `miniz_oxide` | 0.8.9 | MIT OR Zlib OR Apache-2.0 | `Apache-2.0` | [https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide](https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide) |
| `miniz_oxide` | 0.9.1 | MIT OR Zlib OR Apache-2.0 | `Apache-2.0` | [https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide](https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide) |
| `mio` | 1.2.2 | MIT | - | [https://github.com/tokio-rs/mio](https://github.com/tokio-rs/mio) |
| `muda` | 0.20.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/muda](https://github.com/tauri-apps/muda) |
| `ndk` | 0.9.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-mobile/ndk](https://github.com/rust-mobile/ndk) |
| `ndk-context` | 0.1.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-windowing/android-ndk-rs](https://github.com/rust-windowing/android-ndk-rs) |
| `ndk-sys` | 0.6.0+11769913 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-mobile/ndk](https://github.com/rust-mobile/ndk) |
| `new_debug_unreachable` | 1.0.6 | MIT | - | [https://github.com/mbrubeck/rust-debug-unreachable](https://github.com/mbrubeck/rust-debug-unreachable) |
| `nix` | 0.31.3 | MIT | - | [https://github.com/nix-rust/nix](https://github.com/nix-rust/nix) |
| `num-conv` | 0.2.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/jhpratt/num-conv](https://github.com/jhpratt/num-conv) |
| `num-traits` | 0.2.19 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-num/num-traits](https://github.com/rust-num/num-traits) |
| `num_enum` | 0.7.6 | BSD-3-Clause OR MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/illicitonion/num_enum](https://github.com/illicitonion/num_enum) |
| `num_enum_derive` | 0.7.6 | BSD-3-Clause OR MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/illicitonion/num_enum](https://github.com/illicitonion/num_enum) |
| `objc2` | 0.6.4 | MIT | - | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-app-kit` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-cloud-kit` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-core-data` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-core-foundation` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-core-graphics` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-core-image` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-core-location` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-core-text` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-encode` | 4.1.0 | MIT | - | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-exception-helper` | 0.1.1 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-foundation` | 0.3.2 | MIT | - | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-io-surface` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-quartz-core` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-ui-kit` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-user-notifications` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `objc2-web-kit` | 0.3.2 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/madsmtm/objc2](https://github.com/madsmtm/objc2) |
| `once_cell` | 1.21.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/matklad/once_cell](https://github.com/matklad/once_cell) |
| `option-ext` | 0.2.0 | MPL-2.0 | - | [https://github.com/soc/option-ext.git](https://github.com/soc/option-ext.git) |
| `pango` | 0.18.3 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `pango-sys` | 0.18.0 | MIT | - | [https://github.com/gtk-rs/gtk-rs-core](https://github.com/gtk-rs/gtk-rs-core) |
| `parking_lot` | 0.12.5 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/Amanieu/parking_lot](https://github.com/Amanieu/parking_lot) |
| `parking_lot_core` | 0.9.12 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/Amanieu/parking_lot](https://github.com/Amanieu/parking_lot) |
| `percent-encoding` | 2.3.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/rust-url/](https://github.com/servo/rust-url/) |
| `phf` | 0.13.1 | MIT | - | [https://github.com/rust-phf/rust-phf](https://github.com/rust-phf/rust-phf) |
| `phf_codegen` | 0.13.1 | MIT | - | [https://github.com/rust-phf/rust-phf](https://github.com/rust-phf/rust-phf) |
| `phf_generator` | 0.13.1 | MIT | - | [https://github.com/rust-phf/rust-phf](https://github.com/rust-phf/rust-phf) |
| `phf_macros` | 0.13.1 | MIT | - | [https://github.com/rust-phf/rust-phf](https://github.com/rust-phf/rust-phf) |
| `phf_shared` | 0.13.1 | MIT | - | [https://github.com/rust-phf/rust-phf](https://github.com/rust-phf/rust-phf) |
| `pin-project-lite` | 0.2.17 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/taiki-e/pin-project-lite](https://github.com/taiki-e/pin-project-lite) |
| `pkg-config` | 0.3.34 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/pkg-config-rs](https://github.com/rust-lang/pkg-config-rs) |
| `plist` | 1.10.0 | MIT | - | [https://github.com/ebarnard/rust-plist/](https://github.com/ebarnard/rust-plist/) |
| `png` | 0.17.16 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/image-rs/image-png](https://github.com/image-rs/image-png) |
| `png` | 0.18.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/image-rs/image-png](https://github.com/image-rs/image-png) |
| `portable-atomic` | 1.15.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/taiki-e/portable-atomic](https://github.com/taiki-e/portable-atomic) |
| `portable-atomic-util` | 0.2.7 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/taiki-e/portable-atomic-util](https://github.com/taiki-e/portable-atomic-util) |
| `potential_utf` | 0.1.6 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `powerfmt` | 0.2.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/jhpratt/powerfmt](https://github.com/jhpratt/powerfmt) |
| `precomputed-hash` | 0.1.1 | MIT | - | [https://github.com/emilio/precomputed-hash](https://github.com/emilio/precomputed-hash) |
| `proc-macro-crate` | 1.3.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/bkchr/proc-macro-crate](https://github.com/bkchr/proc-macro-crate) |
| `proc-macro-crate` | 2.0.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/bkchr/proc-macro-crate](https://github.com/bkchr/proc-macro-crate) |
| `proc-macro-crate` | 3.5.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/bkchr/proc-macro-crate](https://github.com/bkchr/proc-macro-crate) |
| `proc-macro-error` | 1.0.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://gitlab.com/CreepySkeleton/proc-macro-error](https://gitlab.com/CreepySkeleton/proc-macro-error) |
| `proc-macro-error-attr` | 1.0.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://gitlab.com/CreepySkeleton/proc-macro-error](https://gitlab.com/CreepySkeleton/proc-macro-error) |
| `proc-macro2` | 1.0.107 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/proc-macro2](https://github.com/dtolnay/proc-macro2) |
| `quick-xml` | 0.41.0 | MIT | - | [https://github.com/tafia/quick-xml](https://github.com/tafia/quick-xml) |
| `quote` | 1.0.47 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/quote](https://github.com/dtolnay/quote) |
| `r-efi` | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | `Apache-2.0` | [https://github.com/r-efi/r-efi](https://github.com/r-efi/r-efi) |
| `r-efi` | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | `Apache-2.0` | [https://github.com/r-efi/r-efi](https://github.com/r-efi/r-efi) |
| `raw-window-handle` | 0.6.2 | MIT OR Apache-2.0 OR Zlib | `Apache-2.0` | [https://github.com/rust-windowing/raw-window-handle](https://github.com/rust-windowing/raw-window-handle) |
| `redox_syscall` | 0.5.18 | MIT | - | [https://gitlab.redox-os.org/redox-os/syscall](https://gitlab.redox-os.org/redox-os/syscall) |
| `redox_users` | 0.5.2 | MIT | - | [https://gitlab.redox-os.org/redox-os/users](https://gitlab.redox-os.org/redox-os/users) |
| `ref-cast` | 1.0.27 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/ref-cast](https://github.com/dtolnay/ref-cast) |
| `ref-cast-impl` | 1.0.27 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/ref-cast](https://github.com/dtolnay/ref-cast) |
| `regex` | 1.13.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/regex](https://github.com/rust-lang/regex) |
| `regex-automata` | 0.4.18 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/regex](https://github.com/rust-lang/regex) |
| `regex-syntax` | 0.8.11 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/regex](https://github.com/rust-lang/regex) |
| `reqwest` | 0.13.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/seanmonstar/reqwest](https://github.com/seanmonstar/reqwest) |
| `rfd` | 0.17.2 | MIT | - | [https://github.com/PolyMeilex/rfd](https://github.com/PolyMeilex/rfd) |
| `ring` | 0.17.14 | Apache-2.0 AND ISC | - | [https://github.com/briansmith/ring](https://github.com/briansmith/ring) |
| `rsqlite-vfs` | 0.2.0 | MIT | - | [https://github.com/Spxg/sqlite-wasm-rs](https://github.com/Spxg/sqlite-wasm-rs) |
| `rusqlite` | 0.43.0 | MIT | - | [https://github.com/rusqlite/rusqlite](https://github.com/rusqlite/rusqlite) |
| `rustc-hash` | 2.1.3 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/rust-lang/rustc-hash](https://github.com/rust-lang/rustc-hash) |
| `rustc_version` | 0.4.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/djc/rustc-version-rs](https://github.com/djc/rustc-version-rs) |
| `rustls` | 0.23.46 | Apache-2.0 OR ISC OR MIT | `Apache-2.0` | [https://github.com/rustls/rustls](https://github.com/rustls/rustls) |
| `rustls-pki-types` | 1.15.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rustls/pki-types](https://github.com/rustls/pki-types) |
| `rustls-webpki` | 0.103.16 | ISC | - | [https://github.com/rustls/webpki](https://github.com/rustls/webpki) |
| `rustversion` | 1.0.23 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/rustversion](https://github.com/dtolnay/rustversion) |
| `same-file` | 1.0.6 | Unlicense/MIT | `MIT` | [https://github.com/BurntSushi/same-file](https://github.com/BurntSushi/same-file) |
| `schemars` | 0.8.22 | MIT | - | [https://github.com/GREsau/schemars](https://github.com/GREsau/schemars) |
| `schemars` | 0.9.0 | MIT | - | [https://github.com/GREsau/schemars](https://github.com/GREsau/schemars) |
| `schemars` | 1.2.2 | MIT | - | [https://github.com/GREsau/schemars](https://github.com/GREsau/schemars) |
| `schemars_derive` | 0.8.22 | MIT | - | [https://github.com/GREsau/schemars](https://github.com/GREsau/schemars) |
| `scopeguard` | 1.2.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/bluss/scopeguard](https://github.com/bluss/scopeguard) |
| `selectors` | 0.38.0 | MPL-2.0 | - | [https://github.com/servo/stylo](https://github.com/servo/stylo) |
| `semver` | 1.0.28 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/semver](https://github.com/dtolnay/semver) |
| `serde` | 1.0.229 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/serde](https://github.com/serde-rs/serde) |
| `serde-untagged` | 0.1.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/serde-untagged](https://github.com/dtolnay/serde-untagged) |
| `serde_core` | 1.0.229 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/serde](https://github.com/serde-rs/serde) |
| `serde_derive` | 1.0.229 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/serde](https://github.com/serde-rs/serde) |
| `serde_derive_internals` | 0.29.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/serde](https://github.com/serde-rs/serde) |
| `serde_json` | 1.0.154 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/json](https://github.com/serde-rs/json) |
| `serde_repr` | 0.1.21 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/serde-repr](https://github.com/dtolnay/serde-repr) |
| `serde_spanned` | 0.6.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `serde_spanned` | 1.1.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `serde_with` | 3.22.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/jonasbb/serde_with/](https://github.com/jonasbb/serde_with/) |
| `serde_with_macros` | 3.22.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/jonasbb/serde_with/](https://github.com/jonasbb/serde_with/) |
| `serialize-to-javascript` | 0.1.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/chippers/serialize-to-javascript](https://github.com/chippers/serialize-to-javascript) |
| `serialize-to-javascript-impl` | 0.1.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/chippers/serialize-to-javascript](https://github.com/chippers/serialize-to-javascript) |
| `servo_arc` | 0.4.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/stylo](https://github.com/servo/stylo) |
| `sha2` | 0.10.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/hashes](https://github.com/RustCrypto/hashes) |
| `sha2` | 0.11.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/hashes](https://github.com/RustCrypto/hashes) |
| `shlex` | 2.0.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/comex/rust-shlex](https://github.com/comex/rust-shlex) |
| `simd-adler32` | 0.3.10 | MIT | - | [https://github.com/mcountryman/simd-adler32](https://github.com/mcountryman/simd-adler32) |
| `siphasher` | 1.0.3 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/jedisct1/rust-siphash](https://github.com/jedisct1/rust-siphash) |
| `slab` | 0.4.12 | MIT | - | [https://github.com/tokio-rs/slab](https://github.com/tokio-rs/slab) |
| `smallvec` | 1.15.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/rust-smallvec](https://github.com/servo/rust-smallvec) |
| `socket2` | 0.6.5 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/socket2](https://github.com/rust-lang/socket2) |
| `softbuffer` | 0.4.8 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-windowing/softbuffer](https://github.com/rust-windowing/softbuffer) |
| `soup3` | 0.5.0 | MIT | - | [https://gitlab.gnome.org/World/Rust/soup3-rs](https://gitlab.gnome.org/World/Rust/soup3-rs) |
| `soup3-sys` | 0.5.0 | MIT | - | [https://gitlab.gnome.org/World/Rust/soup3-rs](https://gitlab.gnome.org/World/Rust/soup3-rs) |
| `sqlite-wasm-rs` | 0.6.3 | MIT | - | [https://github.com/Spxg/sqlite-wasm-rs](https://github.com/Spxg/sqlite-wasm-rs) |
| `stable_deref_trait` | 1.2.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/storyyeller/stable_deref_trait](https://github.com/storyyeller/stable_deref_trait) |
| `string_cache` | 0.9.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/string-cache](https://github.com/servo/string-cache) |
| `string_cache_codegen` | 0.6.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/string-cache](https://github.com/servo/string-cache) |
| `strsim` | 0.11.1 | MIT | - | [https://github.com/rapidfuzz/strsim-rs](https://github.com/rapidfuzz/strsim-rs) |
| `subtle` | 2.6.1 | BSD-3-Clause | - | [https://github.com/dalek-cryptography/subtle](https://github.com/dalek-cryptography/subtle) |
| `swift-rs` | 1.0.8 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/Brendonovich/swift-rs](https://github.com/Brendonovich/swift-rs) |
| `syn` | 1.0.109 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/syn](https://github.com/dtolnay/syn) |
| `syn` | 2.0.119 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/syn](https://github.com/dtolnay/syn) |
| `syn` | 3.0.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/syn](https://github.com/dtolnay/syn) |
| `sync_wrapper` | 1.0.2 | Apache-2.0 | - | [https://github.com/Actyx/sync_wrapper](https://github.com/Actyx/sync_wrapper) |
| `synstructure` | 0.13.2 | MIT | - | [https://github.com/mystor/synstructure](https://github.com/mystor/synstructure) |
| `system-deps` | 6.2.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/gdesmott/system-deps](https://github.com/gdesmott/system-deps) |
| `tao` | 0.37.1 | Apache-2.0 | - | [https://github.com/tauri-apps/tao](https://github.com/tauri-apps/tao) |
| `tao-macros` | 0.1.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/tauri-apps/tao](https://github.com/tauri-apps/tao) |
| `target-lexicon` | 0.12.16 | Apache-2.0 WITH LLVM-exception | - | [https://github.com/bytecodealliance/target-lexicon](https://github.com/bytecodealliance/target-lexicon) |
| `tauri` | 2.12.2 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| `tauri-build` | 2.7.2 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| `tauri-codegen` | 2.7.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| `tauri-macros` | 2.7.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| `tauri-plugin` | 2.7.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| `tauri-plugin-dialog` | 2.8.4 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/plugins-workspace](https://github.com/tauri-apps/plugins-workspace) |
| `tauri-plugin-fs` | 2.7.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/plugins-workspace](https://github.com/tauri-apps/plugins-workspace) |
| `tauri-runtime` | 2.12.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| `tauri-runtime-wry` | 2.12.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| `tauri-utils` | 2.10.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| `tauri-winres` | 0.3.6 | MIT | - | [https://github.com/tauri-apps/winres](https://github.com/tauri-apps/winres) |
| `tendril` | 0.5.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/html5ever](https://github.com/servo/html5ever) |
| `thiserror` | 1.0.69 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/thiserror](https://github.com/dtolnay/thiserror) |
| `thiserror` | 2.0.20 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/thiserror](https://github.com/dtolnay/thiserror) |
| `thiserror-impl` | 1.0.69 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/thiserror](https://github.com/dtolnay/thiserror) |
| `thiserror-impl` | 2.0.20 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/thiserror](https://github.com/dtolnay/thiserror) |
| `time` | 0.3.55 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/time-rs/time](https://github.com/time-rs/time) |
| `time-core` | 0.1.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/time-rs/time](https://github.com/time-rs/time) |
| `time-macros` | 0.2.32 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/time-rs/time](https://github.com/time-rs/time) |
| `tinystr` | 0.8.4 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `tinyvec` | 1.12.0 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/Lokathor/tinyvec](https://github.com/Lokathor/tinyvec) |
| `tinyvec_macros` | 0.1.1 | MIT OR Apache-2.0 OR Zlib | `Apache-2.0` | [https://github.com/Soveu/tinyvec_macros](https://github.com/Soveu/tinyvec_macros) |
| `tokio` | 1.54.1 | MIT | - | [https://github.com/tokio-rs/tokio](https://github.com/tokio-rs/tokio) |
| `tokio-util` | 0.7.19 | MIT | - | [https://github.com/tokio-rs/tokio](https://github.com/tokio-rs/tokio) |
| `toml` | 0.8.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `toml` | 1.1.4+spec-1.1.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `toml_datetime` | 0.6.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `toml_datetime` | 1.1.1+spec-1.1.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `toml_edit` | 0.19.15 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `toml_edit` | 0.20.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `toml_edit` | 0.25.13+spec-1.1.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `toml_parser` | 1.1.3+spec-1.1.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `toml_writer` | 1.1.2+spec-1.1.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/toml-rs/toml](https://github.com/toml-rs/toml) |
| `tower` | 0.5.3 | MIT | - | [https://github.com/tower-rs/tower](https://github.com/tower-rs/tower) |
| `tower-http` | 0.6.11 | MIT | - | [https://github.com/tower-rs/tower-http](https://github.com/tower-rs/tower-http) |
| `tower-layer` | 0.3.3 | MIT | - | [https://github.com/tower-rs/tower](https://github.com/tower-rs/tower) |
| `tower-service` | 0.3.3 | MIT | - | [https://github.com/tower-rs/tower](https://github.com/tower-rs/tower) |
| `tracing` | 0.1.44 | MIT | - | [https://github.com/tokio-rs/tracing](https://github.com/tokio-rs/tracing) |
| `tracing-core` | 0.1.36 | MIT | - | [https://github.com/tokio-rs/tracing](https://github.com/tokio-rs/tracing) |
| `tray-icon` | 0.25.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/tauri-apps/tray-icon](https://github.com/tauri-apps/tray-icon) |
| `try-lock` | 0.2.5 | MIT | - | [https://github.com/seanmonstar/try-lock](https://github.com/seanmonstar/try-lock) |
| `typed-path` | 0.13.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/chipsenkbeil/typed-path](https://github.com/chipsenkbeil/typed-path) |
| `typeid` | 1.0.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/typeid](https://github.com/dtolnay/typeid) |
| `typenum` | 1.20.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/paholg/typenum](https://github.com/paholg/typenum) |
| `unicode-ident` | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 | `Apache-2.0` | [https://github.com/dtolnay/unicode-ident](https://github.com/dtolnay/unicode-ident) |
| `unicode-segmentation` | 1.13.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/unicode-rs/unicode-segmentation](https://github.com/unicode-rs/unicode-segmentation) |
| `untrusted` | 0.9.0 | ISC | - | [https://github.com/briansmith/untrusted](https://github.com/briansmith/untrusted) |
| `ureq` | 3.5.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/algesten/ureq](https://github.com/algesten/ureq) |
| `ureq-proto` | 0.7.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/algesten/ureq-proto](https://github.com/algesten/ureq-proto) |
| `url` | 2.5.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/rust-url](https://github.com/servo/rust-url) |
| `urlpattern` | 0.6.0 | MIT | - | [https://github.com/denoland/rust-urlpattern](https://github.com/denoland/rust-urlpattern) |
| `utf8-zero` | 0.8.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/algesten/utf8-zero](https://github.com/algesten/utf8-zero) |
| `utf8_iter` | 1.0.4 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/hsivonen/utf8_iter](https://github.com/hsivonen/utf8_iter) |
| `uuid` | 1.26.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/uuid-rs/uuid](https://github.com/uuid-rs/uuid) |
| `vcpkg` | 0.2.15 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/mcgoo/vcpkg-rs](https://github.com/mcgoo/vcpkg-rs) |
| `version-compare` | 0.2.1 | MIT | - | [https://gitlab.com/timvisee/version-compare](https://gitlab.com/timvisee/version-compare) |
| `version_check` | 0.9.5 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/SergioBenitez/version_check](https://github.com/SergioBenitez/version_check) |
| `vswhom` | 0.1.0 | MIT | - | [https://github.com/nabijaczleweli/vswhom.rs](https://github.com/nabijaczleweli/vswhom.rs) |
| `vswhom-sys` | 0.1.3 | MIT | - | [https://github.com/nabijaczleweli/vswhom-sys.rs](https://github.com/nabijaczleweli/vswhom-sys.rs) |
| `walkdir` | 2.5.0 | Unlicense/MIT | `MIT` | [https://github.com/BurntSushi/walkdir](https://github.com/BurntSushi/walkdir) |
| `want` | 0.3.1 | MIT | - | [https://github.com/seanmonstar/want](https://github.com/seanmonstar/want) |
| `wasi` | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/bytecodealliance/wasi](https://github.com/bytecodealliance/wasi) |
| `wasip2` | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/bytecodealliance/wasi-rs](https://github.com/bytecodealliance/wasi-rs) |
| `wasm-bindgen` | 0.2.127 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen](https://github.com/wasm-bindgen/wasm-bindgen) |
| `wasm-bindgen-futures` | 0.4.77 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/futures](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/futures) |
| `wasm-bindgen-macro` | 0.2.127 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro) |
| `wasm-bindgen-macro-support` | 0.2.127 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro-support](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro-support) |
| `wasm-bindgen-shared` | 0.2.127 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/shared](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/shared) |
| `wasm-streams` | 0.5.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/MattiasBuelens/wasm-streams/](https://github.com/MattiasBuelens/wasm-streams/) |
| `web-sys` | 0.3.104 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/web-sys](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/web-sys) |
| `web-time` | 1.1.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/daxpedda/web-time](https://github.com/daxpedda/web-time) |
| `web_atoms` | 0.2.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/html5ever](https://github.com/servo/html5ever) |
| `webkit2gtk` | 2.0.2 | MIT | - | [https://github.com/tauri-apps/webkit2gtk-rs](https://github.com/tauri-apps/webkit2gtk-rs) |
| `webkit2gtk-sys` | 2.0.2 | MIT | - | [https://github.com/tauri-apps/webkit2gtk-rs](https://github.com/tauri-apps/webkit2gtk-rs) |
| `webpki-roots` | 1.0.10 | CDLA-Permissive-2.0 | - | [https://github.com/rustls/webpki-roots](https://github.com/rustls/webpki-roots) |
| `webview2-com` | 0.39.1 | MIT | - | [https://github.com/wravery/webview2-rs](https://github.com/wravery/webview2-rs) |
| `webview2-com-macros` | 0.8.1 | MIT | - | [https://github.com/wravery/webview2-rs](https://github.com/wravery/webview2-rs) |
| `webview2-com-sys` | 0.39.1 | MIT | - | [https://github.com/wravery/webview2-rs](https://github.com/wravery/webview2-rs) |
| `winapi` | 0.3.9 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/retep998/winapi-rs](https://github.com/retep998/winapi-rs) |
| `winapi-i686-pc-windows-gnu` | 0.4.0 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/retep998/winapi-rs](https://github.com/retep998/winapi-rs) |
| `winapi-util` | 0.1.11 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/winapi-util](https://github.com/BurntSushi/winapi-util) |
| `winapi-x86_64-pc-windows-gnu` | 0.4.0 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/retep998/winapi-rs](https://github.com/retep998/winapi-rs) |
| `window-vibrancy` | 0.8.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/tauri-plugin-vibrancy](https://github.com/tauri-apps/tauri-plugin-vibrancy) |
| `windows` | 0.62.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-collections` | 0.3.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-core` | 0.62.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-future` | 0.3.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-implement` | 0.60.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-interface` | 0.59.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-link` | 0.2.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-numerics` | 0.3.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-result` | 0.4.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-strings` | 0.5.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-sys` | 0.45.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-sys` | 0.52.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-sys` | 0.59.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-sys` | 0.61.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-targets` | 0.42.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-targets` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-threading` | 0.2.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-version` | 0.1.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_aarch64_gnullvm` | 0.42.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_aarch64_gnullvm` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_aarch64_msvc` | 0.42.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_aarch64_msvc` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_i686_gnu` | 0.42.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_i686_gnu` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_i686_gnullvm` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_i686_msvc` | 0.42.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_i686_msvc` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_x86_64_gnu` | 0.42.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_x86_64_gnu` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_x86_64_gnullvm` | 0.42.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_x86_64_gnullvm` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_x86_64_msvc` | 0.42.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows_x86_64_msvc` | 0.52.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `winnow` | 0.5.40 | MIT | - | [https://github.com/winnow-rs/winnow](https://github.com/winnow-rs/winnow) |
| `winnow` | 1.0.4 | MIT | - | [https://github.com/winnow-rs/winnow](https://github.com/winnow-rs/winnow) |
| `winreg` | 0.55.0 | MIT | - | [https://github.com/gentoo90/winreg-rs](https://github.com/gentoo90/winreg-rs) |
| `wit-bindgen` | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/bytecodealliance/wit-bindgen](https://github.com/bytecodealliance/wit-bindgen) |
| `writeable` | 0.6.4 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `wry` | 0.57.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/tauri-apps/wry](https://github.com/tauri-apps/wry) |
| `x11` | 2.21.0 | MIT | - | [https://github.com/AltF02/x11-rs.git](https://github.com/AltF02/x11-rs.git) |
| `x11-dl` | 2.21.0 | MIT | - | [https://github.com/AltF02/x11-rs.git](https://github.com/AltF02/x11-rs.git) |
| `yoke` | 0.8.3 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `yoke-derive` | 0.8.2 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `zerofrom` | 0.1.8 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `zerofrom-derive` | 0.1.7 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `zeroize` | 1.9.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `zeroize_derive` | 1.5.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `zerotrie` | 0.2.5 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `zerovec` | 0.11.8 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `zerovec-derive` | 0.11.6 | Unicode-3.0 | - | [https://github.com/unicode-org/icu4x](https://github.com/unicode-org/icu4x) |
| `zip` | 9.8.1 | MIT | - | [https://github.com/zip-rs/zip2](https://github.com/zip-rs/zip2) |
| `zlib-rs` | 0.6.7 | Zlib | - | [https://github.com/trifectatechfoundation/zlib-rs](https://github.com/trifectatechfoundation/zlib-rs) |
| `zmij` | 1.0.23 | MIT | - | [https://github.com/dtolnay/zmij](https://github.com/dtolnay/zmij) |
| `zopfli` | 0.8.3 | Apache-2.0 | - | [https://github.com/zopfli-rs/zopfli](https://github.com/zopfli-rs/zopfli) |

## Licence texts

Apache-2.0 is reproduced in full in `LICENSE`, which is MuMDIA's own licence and
covers the Apache-2.0 dependencies as well. The remaining families are below.
Individual copyright holders are named in each crate's own repository, linked in
the table above; this file reproduces the licence terms, not per-crate
attribution lines.

### BSD-3-Clause

```text
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list
   of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT, ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### MIT

```text
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### Zlib

```text
This software is provided 'as-is', without any express or implied warranty. In no
event will the authors be held liable for any damages arising from the use of this
software.

Permission is granted to anyone to use this software for any purpose, including
commercial applications, and to alter it and redistribute it freely, subject to the
following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that
   you wrote the original software.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```

### Unicode-3.0

Applies through an `AND`, so it is not one of several arms to choose from. It
covers the Unicode character tables embedded in the crate rather than its
code, and the canonical text ships with the crate:

- `icu_collections` -- https://github.com/unicode-org/icu4x
- `icu_locale_core` -- https://github.com/unicode-org/icu4x
- `icu_normalizer` -- https://github.com/unicode-org/icu4x
- `icu_normalizer_data` -- https://github.com/unicode-org/icu4x
- `icu_properties` -- https://github.com/unicode-org/icu4x
- `icu_properties_data` -- https://github.com/unicode-org/icu4x
- `icu_provider` -- https://github.com/unicode-org/icu4x
- `litemap` -- https://github.com/unicode-org/icu4x
- `potential_utf` -- https://github.com/unicode-org/icu4x
- `tinystr` -- https://github.com/unicode-org/icu4x
- `unicode-ident` -- https://github.com/dtolnay/unicode-ident
- `writeable` -- https://github.com/unicode-org/icu4x
- `yoke` -- https://github.com/unicode-org/icu4x
- `yoke-derive` -- https://github.com/unicode-org/icu4x
- `zerofrom` -- https://github.com/unicode-org/icu4x
- `zerofrom-derive` -- https://github.com/unicode-org/icu4x
- `zerotrie` -- https://github.com/unicode-org/icu4x
- `zerovec` -- https://github.com/unicode-org/icu4x
- `zerovec-derive` -- https://github.com/unicode-org/icu4x

### Public-domain dedications

These impose no reproduction requirement, and are recorded so it is clear they
were considered: `0BSD`, `CC0-1.0`, `MIT-0`, `Unlicense`.

## Copyright notices

Reproduced from each crate's own `LICENSE`/`COPYING`/`NOTICE` files, because the
licence text alone does not discharge the obligation: MIT requires the copyright
notice in all copies, BSD requires it retained, and Apache-2.0 section 4(d)
requires the contents of a NOTICE file to be propagated.

Notices recovered for 328 of 456 crates (the remainder ship no copyright line in a licence file; see their repositories, linked above).

**adler2 2.0.1**

- Copyright (C) Jonas Schievink <jonasschievink@gmail.com>

**aho-corasick 1.1.5**

- Copyright (c) 2015 Andrew Gallant

**alloc-no-stdlib 3.0.0**

- Copyright (c) 2016 Dropbox, Inc.

**android_system_properties 0.1.6**

- Copyright 2016 Nicolas Silva
- Copyright (c) 2013 Nicolas Silva

**atomic-waker 1.1.2**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**autocfg 1.5.1**

- Copyright (c) 2018 Josh Stone

**base64 0.21.7**

- Copyright (c) 2015 Alice Maz

**base64 0.22.1**

- Copyright (c) 2015 Alice Maz

**base64 0.23.1**

- Copyright (c) 2025 Alice Maz, Marshall Pierce

**bit-set 0.8.0**

- Copyright (c) 2023 The Rust Project Developers

**bit-vec 0.8.0**

- Copyright (c) 2023 The Rust Project Developers

**bitflags 1.3.2**

- Copyright (c) 2014 The Rust Project Developers

**bitflags 2.13.1**

- Copyright (c) 2014 The Rust Project Developers

**block-buffer 0.10.4**

- Copyright (c) 2018-2019 The RustCrypto Project Developers

**block-buffer 0.12.1**

- Copyright (c) 2018-2025 The RustCrypto Project Developers

**brotli 9.0.0**

- Copyright (c) 2016 Dropbox, Inc.
- Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors.

**brotli-decompressor 6.0.1**

- Copyright (c) 2016 Dropbox, Inc.

**bs58 0.5.1**

- Copyright (c) 2016 The roaring-rs developers.

**bumpalo 3.20.3**

- Copyright (c) 2019 Nick Fitzgerald

**bytemuck 1.25.2**

- Copyright (c) 2019 Daniel "Lokathor" Gee.

**byteorder 1.5.0**

- Copyright (c) 2015 Andrew Gallant

**bytes 1.12.1**

- Copyright (c) 2018 Carl Lerche

**cc 1.4.4**

- Copyright (c) 2014 Alex Crichton

**cfb 0.14.0**

- Copyright (c) 2017 Matthew D. Steele

**cfg-expr 0.15.8**

- Copyright (c) 2019 Embark Studios

**cfg-if 1.0.4**

- Copyright (c) 2014 Alex Crichton

**cfg_aliases 0.2.2**

- Copyright (c) 2020 Katharos Technology

**chrono 0.4.45**

- Copyright (c) 2014, Kang Seonghoon.

**combine 4.6.8**

- Copyright (c) 2015 Markus Westerlind

**const-oid 0.10.2**

- Copyright (c) 2020-2026 The RustCrypto Project Developers

**cookie 0.18.3**

- Copyright 2017 Sergio Benitez
- Copyright 2014 Alex Chricton
- Copyright (c) 2017 Sergio Benitez
- Copyright (c) 2014 Alex Crichton

**core-foundation 0.10.1**

- Copyright (c) 2012-2013 Mozilla Foundation

**core-foundation-sys 0.8.7**

- Copyright (c) 2012-2013 Mozilla Foundation

**core-graphics 0.25.0**

- Copyright (c) 2012-2013 Mozilla Foundation

**core-graphics-types 0.2.0**

- Copyright (c) 2012-2013 Mozilla Foundation

**cpufeatures 0.2.17**

- Copyright (c) 2020-2025 The RustCrypto Project Developers

**cpufeatures 0.3.1**

- Copyright (c) 2020-2026 The RustCrypto Project Developers

**crc32fast 1.5.1**

- Copyright (c) 2018 Sam Rijs, Alex Crichton and contributors

**crossbeam-channel 0.5.16**

- Copyright (c) 2019 The Crossbeam Project Developers
- Copyright (c) 2009 The Go Authors. All rights reserved.

**crossbeam-utils 0.8.22**

- Copyright (c) 2019 The Crossbeam Project Developers

**crypto-common 0.1.7**

- Copyright (c) 2021 RustCrypto Developers

**crypto-common 0.2.2**

- Copyright (c) 2021-2026 RustCrypto Developers

**darling 0.23.0**

- Copyright (c) 2017 Ted Driggs

**darling_core 0.23.0**

- Copyright (c) 2017 Ted Driggs

**darling_macro 0.23.0**

- Copyright (c) 2017 Ted Driggs

**dbus 0.9.12**

- Copyright 2014-2018 David Henningsson <diwic@ubuntu.com> and other contributors
- Copyright (c) 2014-2018 David Henningsson <diwic@ubuntu.com> and other contributors

**defmt 1.1.1**

- Copyright (c) Ferrous Systems

**defmt-macros 1.1.1**

- Copyright (c) Ferrous Systems

**deranged 0.5.8**

- Copyright 2024 Jacob Pratt et al.
- Copyright (c) 2024 Jacob Pratt et al.

**derive_more 2.1.1**

- Copyright (c) 2016 Jelte Fennema

**derive_more-impl 2.1.1**

- Copyright (c) 2016 Jelte Fennema

**digest 0.10.7**

- Copyright (c) 2017 Artyom Pavlov

**digest 0.11.3**

- Copyright (c) 2017-2025 RustCrypto Developers
- Copyright (c) 2017 Artyom Pavlov

**dirs 7.0.0**

- Copyright (c) 2018-2019 dirs-rs contributors

**dirs-sys 0.5.0**

- Copyright (c) 2018-2019 dirs-rs contributors

**dom_query 0.28.0**

- Copyright (c) 2023 Mykola Humanov

**dpi 0.1.2**

- Copyright (c) 2018 Jorge Aparicio
- Copyright © 2005-2020 Rich Felker, et al.
- Copyright © 1993,2004 Sun Microsystems or
- Copyright © 2003-2011 David Schultz or
- Copyright © 2003-2009 Steven G. Kargl or
- Copyright © 2003-2009 Bruce D. Evans or
- Copyright © 2008 Stephen L. Moshier or
- Copyright © 2017-2018 Arm Limited

**embed-resource 3.0.11**

- Copyright (c) 2017 nabijaczleweli

**embed_plist 1.2.2**

- Copyright (c) 2020 Nikolai Vazquez

**equivalent 1.0.2**

- Copyright (c) 2016--2023

**fallible-iterator 0.3.0**

- Copyright (c) 2015 The rust-openssl-verify Developers

**fallible-streaming-iterator 0.1.9**

- Copyright (c) 2016 The fallible-streaming-iterator Developers

**field-offset 0.3.6**

- Copyright (c) 2016-2021 Diggory Blake, and other contributors.

**find-msvc-tools 0.1.11**

- Copyright (c) 2014 Alex Crichton

**flate2 1.1.10**

- Copyright (c) 2014-2026 Alex Crichton

**fnv 1.0.7**

- Copyright (c) 2017 Contributors

**foldhash 0.2.0**

- Copyright (c) 2024 Orson Peters

**foreign-types 0.5.0**

- Copyright (c) 2017 The foreign-types Developers

**foreign-types-macros 0.2.4**

- Copyright (c) 2017 The foreign-types Developers

**foreign-types-shared 0.3.1**

- Copyright (c) 2017 The foreign-types Developers

**form_urlencoded 1.2.3**

- Copyright (c) 2013-2016 The rust-url developers

**futures-channel 0.3.34**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-core 0.3.34**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-executor 0.3.34**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-io 0.3.34**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-macro 0.3.34**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-sink 0.3.34**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-task 0.3.34**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-util 0.3.34**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**generic-array 0.14.7**

- Copyright (c) 2015 Bartłomiej Kamiński

**getrandom 0.2.17**

- Copyright (c) 2018-2024 The rust-random Project Developers
- Copyright (c) 2014 The Rust Project Developers

**getrandom 0.3.4**

- Copyright (c) 2018-2025 The rust-random Project Developers
- Copyright (c) 2014 The Rust Project Developers

**getrandom 0.4.3**

- Copyright (c) 2018-2026 The rust-random Project Developers
- Copyright (c) 2014 The Rust Project Developers

**glob 0.3.4**

- Copyright (c) 2014 The Rust Project Developers

**hashbrown 0.12.3**

- Copyright (c) 2016 Amanieu d'Antras

**hashbrown 0.17.1**

- Copyright (c) 2016 Amanieu d'Antras

**heck 0.4.1**

- Copyright (c) 2015 The Rust Project Developers

**heck 0.5.0**

- Copyright (c) 2015 The Rust Project Developers

**hex 0.4.3**

- Copyright (c) 2013-2014 The Rust Project Developers.
- Copyright (c) 2015-2020 The rust-hex Developers

**html5ever 0.39.0**

- Copyright (c) 2014 The html5ever Project Developers

**http 1.5.0**

- Copyright 2017 http-rs authors
- Copyright (c) 2017 http-rs authors

**http-body 1.1.0**

- Copyright (c) 2019-2026 Sean McArthur & Hyper Contributors

**http-body-util 0.1.5**

- Copyright (c) 2019-2026 Sean McArthur & Hyper Contributors

**httparse 1.10.1**

- Copyright (c) 2015-2025 Sean McArthur

**hybrid-array 0.4.15**

- Copyright (c) 2022-2026 The RustCrypto Project Developers

**hyper 1.11.1**

- Copyright (c) 2014-2026 Sean McArthur

**hyper-util 0.1.20**

- Copyright (c) 2023-2025 Sean McArthur

**iana-time-zone 0.1.65**

- Copyright 2020 Andrew Straw
- Copyright (c) 2020 Andrew D. Straw

**iana-time-zone-haiku 0.1.2**

- Copyright 2020 Andrew Straw
- Copyright (c) 2020 Andrew D. Straw

**ico 0.5.0**

- Copyright (c) 2018 Matthew D. Steele

**icu_collections 2.3.0**

- Copyright © 2020-2024 Unicode, Inc.

**icu_locale_core 2.3.0**

- Copyright © 2020-2024 Unicode, Inc.

**icu_normalizer 2.3.0**

- Copyright © 2020-2024 Unicode, Inc.

**icu_normalizer_data 2.3.0**

- Copyright © 2020-2024 Unicode, Inc.

**icu_properties 2.3.0**

- Copyright © 2020-2024 Unicode, Inc.

**icu_properties_data 2.3.0**

- Copyright © 2020-2024 Unicode, Inc.

**icu_provider 2.3.1**

- Copyright © 2020-2024 Unicode, Inc.

**idna 1.1.0**

- Copyright (c) 2013-2025 The rust-url developers

**idna_adapter 1.2.2**

- Copyright (c) The rust-url developers

**indexmap 1.9.3**

- Copyright (c) 2016--2017

**indexmap 2.14.1**

- Copyright (c) 2016--2017

**infer 0.22.0**

- Copyright (c) 2019 Bojan

**ipnet 2.12.1**

- Copyright 2017 Juniper Networks, Inc.

**javascriptcore-rs 1.1.2**

- Copyright (c) 2013-2021, The Gtk-rs Project Developers.
- Copyright (c) 2021, Tauri Programme within The Commons Conservancy.

**javascriptcore-rs-sys 1.1.1**

- Copyright (c) 2013-2017, The Gtk-rs Project Developers.

**jiff 0.2.35**

- Copyright (c) 2015 Andrew Gallant

**jiff-core 0.1.0**

- Copyright (c) 2015 Andrew Gallant

**jiff-static 0.2.35**

- Copyright (c) 2015 Andrew Gallant

**jiff-tzdb 0.1.8**

- Copyright (c) 2015 Andrew Gallant

**jiff-tzdb-platform 0.1.3**

- Copyright (c) 2015 Andrew Gallant

**jni 0.21.1**

- Copyright (c) 2016 Prevoty, Inc. and jni-rs contributors

**jni-sys 0.3.1**

- Copyright (c) 2015 The rust-jni-sys Developers

**jni-sys 0.4.1**

- Copyright (c) 2015 The rust-jni-sys Developers

**js-sys 0.3.104**

- Copyright (c) 2014 Alex Crichton

**json-patch 4.2.0**

- Copyright (c) 2017 Ivan Dubrov

**jsonptr 0.7.1**

- Copyright 2024 Chance Dinkins
- Copyright (c) 2022 Chance Dinkins

**keyboard-types 0.8.3**

- Copyright (c) 2017 Pyfisch

**libappindicator 0.9.0**

- Copyright (c) 2017-2021 qDot
- Copyright (c) 2021 Tauri Apps Contributors

**libc 0.2.189**

- Copyright (c) The Rust Project Developers

**libdbus-sys 0.2.7**

- Copyright 2014-2018 David Henningsson <diwic@ubuntu.com> and other contributors
- Copyright (c) 2014-2018 David Henningsson <diwic@ubuntu.com> and other contributors

**libloading 0.7.4**

- Copyright © 2015, Simonas Kazlauskas

**libredox 0.1.21**

- Copyright (c) 2023 4lDO2

**libsqlite3-sys 0.40.0**

- Copyright (c) 2014 The rusqlite developers

**litemap 0.8.3**

- Copyright © 2020-2024 Unicode, Inc.

**lock_api 0.4.14**

- Copyright (c) 2016 The Rust Project Developers

**log 0.4.34**

- Copyright (c) 2014 The Rust Project Developers

**markup5ever 0.39.0**

- Copyright (c) 2014 The html5ever Project Developers

**memchr 2.8.3**

- Copyright (c) 2015 Andrew Gallant

**memoffset 0.9.1**

- Copyright (c) 2017 Gilad Naaman

**mime 0.3.17**

- Copyright (c) 2014 Sean McArthur

**miniz_oxide 0.8.9**

- Copyright 2013-2014 RAD Game Tools and Valve Software
- Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC
- Copyright (c) 2017 Frommi
- Copyright (c) 2017-2024 oyvindln
- Copyright (c) 2020 Frommi

**miniz_oxide 0.9.1**

- Copyright 2013-2014 RAD Game Tools and Valve Software
- Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC
- Copyright (c) 2017 Frommi
- Copyright (c) 2017-2024 oyvindln
- Copyright (c) 2020 Frommi

**mio 1.2.2**

- Copyright (c) 2014 Carl Lerche and other MIO contributors

**muda 0.20.0**

- Copyright (c) 2022-2022 Tauri Programme within The Commons Conservancy

**new_debug_unreachable 1.0.6**

- Copyright (c) 2015 Jonathan Reem

**nix 0.31.3**

- Copyright (c) 2015 Carl Lerche + nix-rust Authors

**num-conv 0.2.2**

- Copyright (c) Jacob Pratt

**num-traits 0.2.19**

- Copyright (c) 2014 The Rust Project Developers

**num_enum 0.7.6**

- Copyright (c) 2018, Daniel Wagner-Hall

**num_enum_derive 0.7.6**

- Copyright (c) 2018, Daniel Wagner-Hall

**parking_lot 0.12.5**

- Copyright (c) 2016 The Rust Project Developers

**parking_lot_core 0.9.12**

- Copyright (c) 2016 The Rust Project Developers

**percent-encoding 2.3.2**

- Copyright (c) 2013-2025 The rust-url developers

**phf 0.13.1**

- Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi

**phf_codegen 0.13.1**

- Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi

**phf_generator 0.13.1**

- Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi

**phf_macros 0.13.1**

- Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi

**phf_shared 0.13.1**

- Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi

**pkg-config 0.3.34**

- Copyright (c) 2014 Alex Crichton

**plist 1.10.0**

- Copyright (c) 2015 Edward Barnard

**png 0.17.16**

- Copyright (c) 2015 nwin

**png 0.18.1**

- Copyright (c) 2015 nwin

**potential_utf 0.1.6**

- Copyright © 2020-2024 Unicode, Inc.

**powerfmt 0.2.0**

- Copyright 2023 Jacob Pratt et al.
- Copyright (c) 2023 Jacob Pratt et al.

**precomputed-hash 0.1.1**

- Copyright (c) 2017 Emilio Cobos Álvarez

**proc-macro-error 1.0.4**

- Copyright 2019-2020 CreepySkeleton <creepy-skeleton@yandex.ru>
- Copyright (c) 2019-2020 CreepySkeleton

**proc-macro-error-attr 1.0.4**

- Copyright 2019-2020 CreepySkeleton <creepy-skeleton@yandex.ru>
- Copyright (c) 2019-2020 CreepySkeleton

**quick-xml 0.41.0**

- Copyright (c) 2016 Johann Tuffe

**r-efi 5.3.0**

- Copyright (C) 2017-2023 Red Hat, Inc.
- Copyright (C) 2019-2023 Microsoft Corporation
- Copyright (C) 2022-2023 David Rheinsberg

**r-efi 6.0.0**

- Copyright (C) 2017-2023 Red Hat, Inc.
- Copyright (C) 2019-2023 Microsoft Corporation
- Copyright (C) 2022-2023 David Rheinsberg

**raw-window-handle 0.6.2**

- Copyright (c) 2019 Osspial
- Copyright (c) 2020 Osspial

**redox_syscall 0.5.18**

- Copyright (c) 2017 Redox OS Developers

**redox_users 0.5.2**

- Copyright (c) 2017 Jose Narvaez

**regex 1.13.1**

- Copyright (c) 2014 The Rust Project Developers

**regex-automata 0.4.18**

- Copyright (c) 2014 The Rust Project Developers

**regex-syntax 0.8.11**

- Copyright (c) 2014 The Rust Project Developers

**reqwest 0.13.4**

- Copyright 2016 Sean McArthur
- Copyright (c) 2016-2026 Sean McArthur

**rfd 0.17.2**

- Copyright (c) 2022 Bartłomiej Maryńczak

**ring 0.17.14**

- Copyright (c) 2009 The Go Authors. All rights reserved.
- Copyright 2015 The Chromium Authors. All rights reserved.
- Copyright 2015-2025 Brian Smith.

**rusqlite 0.43.0**

- Copyright (c) 2014 The rusqlite developers

**rustc_version 0.4.1**

- Copyright (c) 2016 The Rust Project Developers

**rustls 0.23.46**

- Copyright (c) 2016, Joseph Birr-Pixton <jpixton@gmail.com>
- Copyright (c) 2016 Joseph Birr-Pixton <jpixton@gmail.com>

**rustls-pki-types 1.15.1**

- Copyright 2023 Dirkjan Ochtman
- Copyright (c) 2023 Dirkjan Ochtman <dirkjan@ochtman.nl>

**rustls-webpki 0.103.16**

- Copyright 2015 Brian Smith.

**same-file 1.0.6**

- Copyright (c) 2017 Andrew Gallant

**schemars 0.8.22**

- Copyright (c) 2019 Graham Esau

**schemars 0.9.0**

- Copyright (c) 2019 Graham Esau

**schemars 1.2.2**

- Copyright (c) 2019 Graham Esau

**schemars_derive 0.8.22**

- Copyright (c) 2019 Graham Esau

**scopeguard 1.2.0**

- Copyright (c) 2016-2019 Ulrik Sverdrup "bluss" and scopeguard developers

**serde_spanned 0.6.9**

- Copyright (c) Individual contributors

**serde_spanned 1.1.1**

- Copyright (c) Individual contributors

**serde_with 3.22.0**

- Copyright (c) 2015

**serde_with_macros 3.22.0**

- Copyright (c) 2015

**serialize-to-javascript 0.1.2**

- Copyright (c) 2021 Chip Reed

**serialize-to-javascript-impl 0.1.2**

- Copyright (c) 2021 Chip Reed

**sha2 0.10.9**

- Copyright (c) 2006-2009 Graydon Hoare
- Copyright (c) 2009-2013 Mozilla Foundation
- Copyright (c) 2016 Artyom Pavlov

**sha2 0.11.1**

- Copyright (c) 2016-2026 The RustCrypto Project Developers
- Copyright (c) 2016 Artyom Pavlov
- Copyright (c) 2009-2013 Mozilla Foundation
- Copyright (c) 2006-2009 Graydon Hoare

**shlex 2.0.1**

- Copyright 2015 Nicholas Allegra (comex).
- Copyright (c) 2015 Nicholas Allegra (comex).

**simd-adler32 0.3.10**

- Copyright (c) [2021] [Marvin Countryman]

**siphasher 1.0.3**

- Copyright 2012-2016 The Rust Project Developers.
- Copyright 2016-2026 Frank Denis.

**slab 0.4.12**

- Copyright (c) 2019 Carl Lerche

**smallvec 1.15.2**

- Copyright (c) 2018 The Servo Project Developers

**socket2 0.6.5**

- Copyright (c) 2014 Alex Crichton

**softbuffer 0.4.8**

- Copyright 2022 Kirill Chibisov

**soup3 0.5.0**

- Copyright (c) 2013-2017, The Gtk-rs Project Developers.

**soup3-sys 0.5.0**

- Copyright (c) 2013-2017, The Gtk-rs Project Developers.

**sqlite-wasm-rs 0.6.3**

- Copyright (c) 2024 Spxg

**stable_deref_trait 1.2.1**

- Copyright (c) 2017 Robert Grosse

**string_cache 0.9.0**

- Copyright (c) 2012-2013 Mozilla Foundation

**string_cache_codegen 0.6.1**

- Copyright (c) 2012-2013 Mozilla Foundation

**strsim 0.11.1**

- Copyright (c) 2015 Danny Guo
- Copyright (c) 2016 Titus Wormer <tituswormer@gmail.com>
- Copyright (c) 2018 Akash Kurdekar

**subtle 2.6.1**

- Copyright (c) 2016-2017 Isis Agora Lovecruft, Henry de Valence. All rights reserved.
- Copyright (c) 2016-2024 Isis Agora Lovecruft. All rights reserved.

**swift-rs 1.0.8**

- Copyright 2023 The swift-rs developers
- Copyright (c) 2023 The swift-rs Developers

**synstructure 0.13.2**

- Copyright 2016 Nika Layzell

**tao-macros 0.1.4**

- Copyright (c) 2022 - Present Tauri Apps Contributors

**tauri 2.12.2**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-build 2.7.2**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-codegen 2.7.1**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-macros 2.7.1**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-plugin-dialog 2.8.4**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-plugin-fs 2.7.1**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-runtime 2.12.1**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-runtime-wry 2.12.1**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-utils 2.10.1**

- Copyright (c) 2017 - Present Tauri Apps Contributors

**tauri-winres 0.3.6**

- Copyright (c) 2023 - Present Tauri Apps Contributors
- Copyright (c) 2016 Max Resch

**tendril 0.5.1**

- Copyright (c) 2015 Keegan McAllister

**time 0.3.55**

- Copyright (c) Jacob Pratt et al.

**time-core 0.1.9**

- Copyright (c) Jacob Pratt et al.

**time-macros 0.2.32**

- Copyright (c) Jacob Pratt et al.

**tinystr 0.8.4**

- Copyright © 2020-2024 Unicode, Inc.

**tinyvec 1.12.0**

- Copyright (c) 2019 Daniel "Lokathor" Gee.

**tinyvec_macros 0.1.1**

- Copyright 2020 Tomasz "Soveu" Marx
- Copyright (c) 2020 Soveu

**tokio 1.54.1**

- Copyright (c) Tokio Contributors

**tokio-util 0.7.19**

- Copyright (c) Tokio Contributors

**toml 0.8.2**

- Copyright (c) Individual contributors

**toml 1.1.4+spec-1.1.0**

- Copyright (c) Individual contributors

**toml_datetime 0.6.3**

- Copyright (c) 2014 Alex Crichton

**toml_datetime 1.1.1+spec-1.1.0**

- Copyright (c) Individual contributors

**toml_edit 0.19.15**

- Copyright (c) Individual contributors

**toml_edit 0.20.2**

- Copyright (c) Individual contributors

**toml_edit 0.25.13+spec-1.1.0**

- Copyright (c) Individual contributors

**toml_parser 1.1.3+spec-1.1.0**

- Copyright (c) Individual contributors

**toml_writer 1.1.2+spec-1.1.0**

- Copyright (c) Individual contributors

**tower 0.5.3**

- Copyright (c) 2019 Tower Contributors

**tower-http 0.6.11**

- Copyright (c) 2019-2021 Tower Contributors

**tower-layer 0.3.3**

- Copyright (c) 2019 Tower Contributors

**tower-service 0.3.3**

- Copyright (c) 2019 Tower Contributors

**tracing 0.1.44**

- Copyright (c) 2019 Tokio Contributors

**tracing-core 0.1.36**

- Copyright (c) 2019 Tokio Contributors

**tray-icon 0.25.1**

- Copyright (c) 2022-2022 Tauri Programme within The Commons Conservancy

**try-lock 0.2.5**

- Copyright (c) 2018-2023 Sean McArthur
- Copyright (c) 2016 Alex Crichton

**typenum 1.20.1**

- Copyright 2014 Paho Lurie-Gregg
- Copyright (c) 2014 Paho Lurie-Gregg

**unicode-ident 1.0.24**

- Copyright © 1991-2023 Unicode, Inc.

**unicode-segmentation 1.13.3**

- Copyright (c) 2015 The Rust Project Developers

**untrusted 0.9.0**

- Copyright 2015-2016 Brian Smith.

**ureq 3.5.2**

- Copyright (c) 2019 Martin Algesten

**ureq-proto 0.7.1**

- Copyright 2022 Martin Algesten

**url 2.5.9**

- Copyright (c) 2013-2025 The rust-url developers

**urlpattern 0.6.0**

- Copyright (c) 2021 the Deno authors

**uuid 1.26.0**

- Copyright (c) 2014 The Rust Project Developers
- Copyright (c) 2018 Ashley Mannix, Christopher Armstrong, Dylan DPC, Hunar Roop Kahlon

**vcpkg 0.2.15**

- Copyright (c) 2017 Jim McGrath

**version-compare 0.2.1**

- Copyright (c) 2017 Tim Visée

**version_check 0.9.5**

- Copyright (c) 2017-2018 Sergio Benitez

**vswhom 0.1.0**

- Copyright (c) 2019 nabijaczleweli

**vswhom-sys 0.1.3**

- Copyright (c) 2019 nabijaczleweli

**walkdir 2.5.0**

- Copyright (c) 2015 Andrew Gallant

**want 0.3.1**

- Copyright (c) 2018-2019 Sean McArthur

**wasm-bindgen 0.2.127**

- Copyright (c) 2014 Alex Crichton

**wasm-bindgen-futures 0.4.77**

- Copyright (c) 2014 Alex Crichton

**wasm-bindgen-macro 0.2.127**

- Copyright (c) 2014 Alex Crichton

**wasm-bindgen-macro-support 0.2.127**

- Copyright (c) 2014 Alex Crichton

**wasm-bindgen-shared 0.2.127**

- Copyright (c) 2014 Alex Crichton

**web-sys 0.3.104**

- Copyright (c) 2014 Alex Crichton

**web-time 1.1.0**

- Copyright 2023 dAxpeDDa
- Copyright (c) 2023 dAxpeDDa

**web_atoms 0.2.6**

- Copyright (c) 2014 The html5ever Project Developers

**webkit2gtk 2.0.2**

- Copyright (c) 2016 Boucher, Antoni <bouanto@zoho.com>
- Copyright (c) 2017-2021, The Gtk-rs Project Developers.
- Copyright (c) 2021, Tauri Programme within The Commons Conservancy

**webkit2gtk-sys 2.0.2**

- Copyright (c) 2016 Boucher, Antoni <bouanto@zoho.com>

**winapi 0.3.9**

- Copyright (c) 2015-2018 The winapi-rs Developers

**winapi-util 0.1.11**

- Copyright (c) 2017 Andrew Gallant

**window-vibrancy 0.8.1**

- Copyright (c) 2020-2022 Tauri Programme within The Commons Conservancy

**windows 0.62.2**

- Copyright (c) Microsoft Corporation.

**windows-collections 0.3.2**

- Copyright (c) Microsoft Corporation.

**windows-core 0.62.2**

- Copyright (c) Microsoft Corporation.

**windows-future 0.3.2**

- Copyright (c) Microsoft Corporation.

**windows-implement 0.60.2**

- Copyright (c) Microsoft Corporation.

**windows-interface 0.59.3**

- Copyright (c) Microsoft Corporation.

**windows-link 0.2.1**

- Copyright (c) Microsoft Corporation.

**windows-numerics 0.3.1**

- Copyright (c) Microsoft Corporation.

**windows-result 0.4.1**

- Copyright (c) Microsoft Corporation.

**windows-strings 0.5.1**

- Copyright (c) Microsoft Corporation.

**windows-sys 0.45.0**

- Copyright (c) Microsoft Corporation.

**windows-sys 0.52.0**

- Copyright (c) Microsoft Corporation.

**windows-sys 0.59.0**

- Copyright (c) Microsoft Corporation.

**windows-sys 0.61.2**

- Copyright (c) Microsoft Corporation.

**windows-targets 0.42.2**

- Copyright (c) Microsoft Corporation.

**windows-targets 0.52.6**

- Copyright (c) Microsoft Corporation.

**windows-threading 0.2.1**

- Copyright (c) Microsoft Corporation.

**windows-version 0.1.7**

- Copyright (c) Microsoft Corporation.

**windows_aarch64_gnullvm 0.42.2**

- Copyright (c) Microsoft Corporation.

**windows_aarch64_gnullvm 0.52.6**

- Copyright (c) Microsoft Corporation.

**windows_aarch64_msvc 0.42.2**

- Copyright (c) Microsoft Corporation.

**windows_aarch64_msvc 0.52.6**

- Copyright (c) Microsoft Corporation.

**windows_i686_gnu 0.42.2**

- Copyright (c) Microsoft Corporation.

**windows_i686_gnu 0.52.6**

- Copyright (c) Microsoft Corporation.

**windows_i686_gnullvm 0.52.6**

- Copyright (c) Microsoft Corporation.

**windows_i686_msvc 0.42.2**

- Copyright (c) Microsoft Corporation.

**windows_i686_msvc 0.52.6**

- Copyright (c) Microsoft Corporation.

**windows_x86_64_gnu 0.42.2**

- Copyright (c) Microsoft Corporation.

**windows_x86_64_gnu 0.52.6**

- Copyright (c) Microsoft Corporation.

**windows_x86_64_gnullvm 0.42.2**

- Copyright (c) Microsoft Corporation.

**windows_x86_64_gnullvm 0.52.6**

- Copyright (c) Microsoft Corporation.

**windows_x86_64_msvc 0.42.2**

- Copyright (c) Microsoft Corporation.

**windows_x86_64_msvc 0.52.6**

- Copyright (c) Microsoft Corporation.

**winreg 0.55.0**

- Copyright (c) 2015 Igor Shaula

**writeable 0.6.4**

- Copyright © 2020-2024 Unicode, Inc.

**wry 0.57.0**

- Copyright (c) 2020-2023 Ngo Iok Ui & Tauri Programme within The Commons Conservancy

**yoke 0.8.3**

- Copyright © 2020-2024 Unicode, Inc.

**yoke-derive 0.8.2**

- Copyright © 2020-2024 Unicode, Inc.

**zerofrom 0.1.8**

- Copyright © 2020-2024 Unicode, Inc.

**zerofrom-derive 0.1.7**

- Copyright © 2020-2024 Unicode, Inc.

**zeroize 1.9.0**

- Copyright (c) 2018-2026 The RustCrypto Project Developers

**zeroize_derive 1.5.0**

- Copyright (c) 2019-2026 The RustCrypto Project Developers

**zerotrie 0.2.5**

- Copyright © 2020-2024 Unicode, Inc.

**zerovec 0.11.8**

- Copyright © 2020-2024 Unicode, Inc.

**zerovec-derive 0.11.6**

- Copyright © 2020-2024 Unicode, Inc.

**zip 9.8.1**

- Copyright (c) 2014 Mathijs van de Nes

**zopfli 0.8.3**

- Copyright 2011 Google Inc.

### NOTICE files, verbatim

Apache-2.0 section 4(d) requires these to travel with the distribution.

#### cfg_aliases 0.2.2

```text
# 3rd Party Notices

The `cfg_aliases!` macro uses a lot of the code from [`tectonic_cfg_support::target_cfg!`] macro which is under the following license:

[`tectonic_cfg_support::target_cfg!`]: https://github.com/tectonic-typesetting/tectonic/blob/f2439b936470ad27bdf92882064bc4702ee01899/cfg_support/src/lib.rs#L166

    tectonic_cfg_support is licensed under the MIT License.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the “Software”), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
---
```

## Python sidecars

The Python workers in `scripts/` are MuMDIA's own code under `LICENSE`. Their
dependencies (DeepLC, MS2PIP, mokapot, torch, numpy, pandas, pyarrow) are
installed by the user or by the container build from `env/*.yml` and are not
redistributed inside the binary, so their licences travel with those packages.

## uv

The installers also bundle the `uv` executable (Astral Software Inc., dual
licensed MIT OR Apache-2.0), which installs those environments. Its licence
texts, taken from the uv release the installer bundles, ship beside it as
`uv-LICENSE-MIT` and `uv-LICENSE-APACHE`.

