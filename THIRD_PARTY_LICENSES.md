# Third-party licences

GENERATED FILE. Do not edit. `ci/gen_third_party_licenses.py` writes it from
`rust/mumdia/Cargo.lock`, and CI fails when it is stale.

MuMDIA's own licence is Apache-2.0; see `LICENSE`. The release binary is
statically linked, so it contains the crates listed below and this file
accompanies it in every release archive and in the container image.

## Obligations

173 third-party crates. Every one declares an SPDX expression; none is
unspecified.

The following copyleft identifiers appear, in each case as one arm of a
disjunction whose permissive arm MuMDIA relies on instead (the arm is named
per crate in the table below):

- `LGPL-2.1-or-later`

No crate imposes a copyleft obligation on the distributed binary.

Licence identifiers by crate count:

| SPDX identifier | crates |
|---|---|
| `MIT` | 150 |
| `Apache-2.0` | 140 |
| `Apache-2.0 WITH LLVM-exception` | 4 |
| `Unlicense` | 4 |
| `BSD-2-Clause` | 3 |
| `CC0-1.0` | 3 |
| `BSD-3-Clause` | 2 |
| `Zlib` | 2 |
| `0BSD` | 1 |
| `BSL-1.0` | 1 |
| `LGPL-2.1-or-later` | 1 |
| `MIT-0` | 1 |
| `Unicode-3.0` | 1 |

## Crates

| crate | version | licence | relied-on arm | source |
|---|---|---|---|---|
| `adler2` | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/oyvindln/adler2](https://github.com/oyvindln/adler2) |
| `ahash` | 0.8.12 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/tkaitchuck/ahash](https://github.com/tkaitchuck/ahash) |
| `aho-corasick` | 1.1.4 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/aho-corasick](https://github.com/BurntSushi/aho-corasick) |
| `android_system_properties` | 0.1.5 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/nical/android_system_properties](https://github.com/nical/android_system_properties) |
| `anstream` | 1.0.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-cli/anstyle.git](https://github.com/rust-cli/anstyle.git) |
| `anstyle` | 1.0.14 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-cli/anstyle.git](https://github.com/rust-cli/anstyle.git) |
| `anstyle-parse` | 1.0.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-cli/anstyle.git](https://github.com/rust-cli/anstyle.git) |
| `anstyle-query` | 1.1.5 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-cli/anstyle.git](https://github.com/rust-cli/anstyle.git) |
| `anstyle-wincon` | 3.0.11 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-cli/anstyle.git](https://github.com/rust-cli/anstyle.git) |
| `anyhow` | 1.0.103 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/anyhow](https://github.com/dtolnay/anyhow) |
| `arrayref` | 0.3.9 | BSD-2-Clause | - | [https://github.com/droundy/arrayref](https://github.com/droundy/arrayref) |
| `arrayvec` | 0.7.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/bluss/arrayvec](https://github.com/bluss/arrayvec) |
| `arrow` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-arith` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-array` | 59.0.0 | Apache-2.0 AND MIT | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-buffer` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-cast` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-csv` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-data` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-ipc` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-json` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-ord` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-row` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-schema` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-select` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `arrow-string` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `atoi` | 2.0.0 | MIT | - | [https://github.com/pacman82/atoi-rs](https://github.com/pacman82/atoi-rs) |
| `autocfg` | 1.5.1 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/cuviper/autocfg](https://github.com/cuviper/autocfg) |
| `base64` | 0.22.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/marshallpierce/rust-base64](https://github.com/marshallpierce/rust-base64) |
| `base64-simd` | 0.8.0 | MIT | - | [https://github.com/Nugine/simd](https://github.com/Nugine/simd) |
| `bitflags` | 2.13.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/bitflags/bitflags](https://github.com/bitflags/bitflags) |
| `blake3` | 1.8.5 | CC0-1.0 OR Apache-2.0 OR Apache-2.0 WITH LLVM-exception | `Apache-2.0` | [https://github.com/BLAKE3-team/BLAKE3](https://github.com/BLAKE3-team/BLAKE3) |
| `block-buffer` | 0.10.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `bumpalo` | 3.20.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/fitzgen/bumpalo](https://github.com/fitzgen/bumpalo) |
| `bytemuck` | 1.25.0 | Zlib OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/Lokathor/bytemuck](https://github.com/Lokathor/bytemuck) |
| `bytes` | 1.12.0 | MIT | - | [https://github.com/tokio-rs/bytes](https://github.com/tokio-rs/bytes) |
| `cc` | 1.2.65 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/cc-rs](https://github.com/rust-lang/cc-rs) |
| `cfg-if` | 1.0.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/cfg-if](https://github.com/rust-lang/cfg-if) |
| `chrono` | 0.4.45 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/chronotope/chrono](https://github.com/chronotope/chrono) |
| `clap` | 4.6.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/clap-rs/clap](https://github.com/clap-rs/clap) |
| `clap_builder` | 4.6.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/clap-rs/clap](https://github.com/clap-rs/clap) |
| `clap_derive` | 4.6.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/clap-rs/clap](https://github.com/clap-rs/clap) |
| `clap_lex` | 1.1.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/clap-rs/clap](https://github.com/clap-rs/clap) |
| `colorchoice` | 1.0.5 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-cli/anstyle.git](https://github.com/rust-cli/anstyle.git) |
| `comfy-table` | 7.2.2 | MIT | - | [https://github.com/nukesor/comfy-table](https://github.com/nukesor/comfy-table) |
| `const-random` | 0.1.18 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/tkaitchuck/constrandom](https://github.com/tkaitchuck/constrandom) |
| `const-random-macro` | 0.1.16 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/tkaitchuck/constrandom](https://github.com/tkaitchuck/constrandom) |
| `constant_time_eq` | 0.4.2 | CC0-1.0 OR MIT-0 OR Apache-2.0 | `Apache-2.0` | [https://github.com/cesarb/constant_time_eq](https://github.com/cesarb/constant_time_eq) |
| `core-foundation-sys` | 0.8.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/core-foundation-rs](https://github.com/servo/core-foundation-rs) |
| `cpufeatures` | 0.2.17 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `cpufeatures` | 0.3.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils) |
| `crc32fast` | 1.5.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/srijs/rust-crc32fast](https://github.com/srijs/rust-crc32fast) |
| `crossbeam-deque` | 0.8.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) |
| `crossbeam-epoch` | 0.9.20 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) |
| `crossbeam-utils` | 0.8.21 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) |
| `crunchy` | 0.2.4 | MIT | - | [https://github.com/eira-fransham/crunchy](https://github.com/eira-fransham/crunchy) |
| `crypto-common` | 0.1.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/traits](https://github.com/RustCrypto/traits) |
| `csv` | 1.4.0 | Unlicense/MIT | `MIT` | [https://github.com/BurntSushi/rust-csv](https://github.com/BurntSushi/rust-csv) |
| `csv-core` | 0.1.13 | Unlicense/MIT | `MIT` | [https://github.com/BurntSushi/rust-csv](https://github.com/BurntSushi/rust-csv) |
| `digest` | 0.10.7 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/traits](https://github.com/RustCrypto/traits) |
| `either` | 1.16.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rayon-rs/either](https://github.com/rayon-rs/either) |
| `encoding_rs` | 0.8.35 | (Apache-2.0 OR MIT) AND BSD-3-Clause | `Apache-2.0` | [https://github.com/hsivonen/encoding_rs](https://github.com/hsivonen/encoding_rs) |
| `equivalent` | 1.0.2 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/indexmap-rs/equivalent](https://github.com/indexmap-rs/equivalent) |
| `find-msvc-tools` | 0.1.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/cc-rs](https://github.com/rust-lang/cc-rs) |
| `flatbuffers` | 25.12.19 | Apache-2.0 | - | [https://github.com/google/flatbuffers](https://github.com/google/flatbuffers) |
| `flate2` | 1.1.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/flate2-rs](https://github.com/rust-lang/flate2-rs) |
| `futures-core` | 0.3.32 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-task` | 0.3.32 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `futures-util` | 0.3.32 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/futures-rs](https://github.com/rust-lang/futures-rs) |
| `generic-array` | 0.14.7 | MIT | - | [https://github.com/fizyk20/generic-array.git](https://github.com/fizyk20/generic-array.git) |
| `getrandom` | 0.2.17 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-random/getrandom](https://github.com/rust-random/getrandom) |
| `getrandom` | 0.3.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-random/getrandom](https://github.com/rust-random/getrandom) |
| `half` | 2.7.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/VoidStarKat/half-rs](https://github.com/VoidStarKat/half-rs) |
| `hashbrown` | 0.17.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) |
| `heck` | 0.5.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/withoutboats/heck](https://github.com/withoutboats/heck) |
| `iana-time-zone` | 0.1.65 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/strawlab/iana-time-zone](https://github.com/strawlab/iana-time-zone) |
| `iana-time-zone-haiku` | 0.1.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/strawlab/iana-time-zone](https://github.com/strawlab/iana-time-zone) |
| `identity-hash` | 0.1.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/offsetting/identity-hash](https://github.com/offsetting/identity-hash) |
| `indexmap` | 2.14.0 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/indexmap-rs/indexmap](https://github.com/indexmap-rs/indexmap) |
| `is_terminal_polyfill` | 1.70.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/polyfill-rs/is_terminal_polyfill](https://github.com/polyfill-rs/is_terminal_polyfill) |
| `itoa` | 1.0.18 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/itoa](https://github.com/dtolnay/itoa) |
| `js-sys` | 0.3.103 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/js-sys](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/js-sys) |
| `lazy_static` | 1.5.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang-nursery/lazy-static.rs](https://github.com/rust-lang-nursery/lazy-static.rs) |
| `lexical-core` | 1.0.6 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/Alexhuszagh/rust-lexical](https://github.com/Alexhuszagh/rust-lexical) |
| `lexical-parse-float` | 1.0.6 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/Alexhuszagh/rust-lexical](https://github.com/Alexhuszagh/rust-lexical) |
| `lexical-parse-integer` | 1.0.6 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/Alexhuszagh/rust-lexical](https://github.com/Alexhuszagh/rust-lexical) |
| `lexical-util` | 1.0.7 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/Alexhuszagh/rust-lexical](https://github.com/Alexhuszagh/rust-lexical) |
| `lexical-write-float` | 1.0.6 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/Alexhuszagh/rust-lexical](https://github.com/Alexhuszagh/rust-lexical) |
| `lexical-write-integer` | 1.0.6 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/Alexhuszagh/rust-lexical](https://github.com/Alexhuszagh/rust-lexical) |
| `libc` | 0.2.186 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/libc](https://github.com/rust-lang/libc) |
| `libm` | 0.2.16 | MIT | - | [https://github.com/rust-lang/compiler-builtins](https://github.com/rust-lang/compiler-builtins) |
| `libmimalloc-sys` | 0.1.49 | MIT | - | [https://github.com/purpleprotocol/mimalloc_rust/tree/master/libmimalloc-sys](https://github.com/purpleprotocol/mimalloc_rust/tree/master/libmimalloc-sys) |
| `log` | 0.4.33 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/log](https://github.com/rust-lang/log) |
| `matchers` | 0.2.0 | MIT | - | [https://github.com/hawkw/matchers](https://github.com/hawkw/matchers) |
| `memchr` | 2.8.2 | Unlicense OR MIT | `MIT` | [https://github.com/BurntSushi/memchr](https://github.com/BurntSushi/memchr) |
| `mimalloc` | 0.1.52 | MIT | - | [https://github.com/purpleprotocol/mimalloc_rust](https://github.com/purpleprotocol/mimalloc_rust) |
| `miniz_oxide` | 0.8.9 | MIT OR Zlib OR Apache-2.0 | `Apache-2.0` | [https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide](https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide) |
| `mzdata` | 0.65.5 | Apache-2.0 | - | [https://github.com/mobiusklein/mzdata](https://github.com/mobiusklein/mzdata) |
| `mzpeaks` | 1.0.9 | Apache-2.0 | - | [https://github.com/mobiusklein/mzpeaks](https://github.com/mobiusklein/mzpeaks) |
| `nu-ansi-term` | 0.50.3 | MIT | - | [https://github.com/nushell/nu-ansi-term](https://github.com/nushell/nu-ansi-term) |
| `num-bigint` | 0.4.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-num/num-bigint](https://github.com/rust-num/num-bigint) |
| `num-complex` | 0.4.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-num/num-complex](https://github.com/rust-num/num-complex) |
| `num-integer` | 0.1.46 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-num/num-integer](https://github.com/rust-num/num-integer) |
| `num-traits` | 0.2.19 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-num/num-traits](https://github.com/rust-num/num-traits) |
| `once_cell` | 1.21.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/matklad/once_cell](https://github.com/matklad/once_cell) |
| `once_cell_polyfill` | 1.70.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/polyfill-rs/once_cell_polyfill](https://github.com/polyfill-rs/once_cell_polyfill) |
| `outref` | 0.5.2 | MIT | - | [https://github.com/Nugine/outref](https://github.com/Nugine/outref) |
| `parquet` | 59.0.0 | Apache-2.0 | - | [https://github.com/apache/arrow-rs](https://github.com/apache/arrow-rs) |
| `paste` | 1.0.15 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/paste](https://github.com/dtolnay/paste) |
| `pin-project-lite` | 0.2.17 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/taiki-e/pin-project-lite](https://github.com/taiki-e/pin-project-lite) |
| `proc-macro2` | 1.0.106 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/proc-macro2](https://github.com/dtolnay/proc-macro2) |
| `quick-xml` | 0.41.0 | MIT | - | [https://github.com/tafia/quick-xml](https://github.com/tafia/quick-xml) |
| `quote` | 1.0.46 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/quote](https://github.com/dtolnay/quote) |
| `r-efi` | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | `Apache-2.0` | [https://github.com/r-efi/r-efi](https://github.com/r-efi/r-efi) |
| `rayon` | 1.12.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rayon-rs/rayon](https://github.com/rayon-rs/rayon) |
| `rayon-core` | 1.13.0 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rayon-rs/rayon](https://github.com/rayon-rs/rayon) |
| `regex` | 1.12.4 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/regex](https://github.com/rust-lang/regex) |
| `regex-automata` | 0.4.14 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/regex](https://github.com/rust-lang/regex) |
| `regex-syntax` | 0.8.11 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rust-lang/regex](https://github.com/rust-lang/regex) |
| `rustc_version` | 0.4.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/djc/rustc-version-rs](https://github.com/djc/rustc-version-rs) |
| `rustversion` | 1.0.22 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/rustversion](https://github.com/dtolnay/rustversion) |
| `ryu` | 1.0.23 | Apache-2.0 OR BSL-1.0 | `Apache-2.0` | [https://github.com/dtolnay/ryu](https://github.com/dtolnay/ryu) |
| `semver` | 1.0.28 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/semver](https://github.com/dtolnay/semver) |
| `seq-macro` | 0.3.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/seq-macro](https://github.com/dtolnay/seq-macro) |
| `serde` | 1.0.228 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/serde](https://github.com/serde-rs/serde) |
| `serde_core` | 1.0.228 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/serde](https://github.com/serde-rs/serde) |
| `serde_derive` | 1.0.228 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/serde](https://github.com/serde-rs/serde) |
| `serde_json` | 1.0.150 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/serde-rs/json](https://github.com/serde-rs/json) |
| `sha1` | 0.10.6 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/RustCrypto/hashes](https://github.com/RustCrypto/hashes) |
| `sharded-slab` | 0.1.7 | MIT | - | [https://github.com/hawkw/sharded-slab](https://github.com/hawkw/sharded-slab) |
| `shlex` | 2.0.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/comex/rust-shlex](https://github.com/comex/rust-shlex) |
| `simd-adler32` | 0.3.9 | MIT | - | [https://github.com/mcountryman/simd-adler32](https://github.com/mcountryman/simd-adler32) |
| `simdutf8` | 0.1.5 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/rusticstuff/simdutf8](https://github.com/rusticstuff/simdutf8) |
| `slab` | 0.4.12 | MIT | - | [https://github.com/tokio-rs/slab](https://github.com/tokio-rs/slab) |
| `smallvec` | 1.15.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/servo/rust-smallvec](https://github.com/servo/rust-smallvec) |
| `snap` | 1.1.1 | BSD-3-Clause | - | [https://github.com/BurntSushi/rust-snappy](https://github.com/BurntSushi/rust-snappy) |
| `strsim` | 0.11.1 | MIT | - | [https://github.com/rapidfuzz/strsim-rs](https://github.com/rapidfuzz/strsim-rs) |
| `syn` | 2.0.118 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/syn](https://github.com/dtolnay/syn) |
| `thiserror` | 2.0.18 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/thiserror](https://github.com/dtolnay/thiserror) |
| `thiserror-impl` | 2.0.18 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/dtolnay/thiserror](https://github.com/dtolnay/thiserror) |
| `thread_local` | 1.1.9 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/Amanieu/thread_local-rs](https://github.com/Amanieu/thread_local-rs) |
| `tiny-keccak` | 2.0.2 | CC0-1.0 | - | - |
| `tracing` | 0.1.44 | MIT | - | [https://github.com/tokio-rs/tracing](https://github.com/tokio-rs/tracing) |
| `tracing-attributes` | 0.1.31 | MIT | - | [https://github.com/tokio-rs/tracing](https://github.com/tokio-rs/tracing) |
| `tracing-core` | 0.1.36 | MIT | - | [https://github.com/tokio-rs/tracing](https://github.com/tokio-rs/tracing) |
| `tracing-log` | 0.2.0 | MIT | - | [https://github.com/tokio-rs/tracing](https://github.com/tokio-rs/tracing) |
| `tracing-subscriber` | 0.3.23 | MIT | - | [https://github.com/tokio-rs/tracing](https://github.com/tokio-rs/tracing) |
| `twox-hash` | 2.1.2 | MIT | - | [https://github.com/shepmaster/twox-hash](https://github.com/shepmaster/twox-hash) |
| `typenum` | 1.20.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/paholg/typenum](https://github.com/paholg/typenum) |
| `unicode-ident` | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 | `Apache-2.0` | [https://github.com/dtolnay/unicode-ident](https://github.com/dtolnay/unicode-ident) |
| `unicode-segmentation` | 1.13.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/unicode-rs/unicode-segmentation](https://github.com/unicode-rs/unicode-segmentation) |
| `unicode-width` | 0.2.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/unicode-rs/unicode-width](https://github.com/unicode-rs/unicode-width) |
| `utf8parse` | 0.2.2 | Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/alacritty/vte](https://github.com/alacritty/vte) |
| `valuable` | 0.1.1 | MIT | - | [https://github.com/tokio-rs/valuable](https://github.com/tokio-rs/valuable) |
| `version_check` | 0.9.5 | MIT/Apache-2.0 | `Apache-2.0` | [https://github.com/SergioBenitez/version_check](https://github.com/SergioBenitez/version_check) |
| `vsimd` | 0.8.0 | MIT | - | [https://github.com/Nugine/simd](https://github.com/Nugine/simd) |
| `wasi` | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/bytecodealliance/wasi](https://github.com/bytecodealliance/wasi) |
| `wasip2` | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/bytecodealliance/wasi-rs](https://github.com/bytecodealliance/wasi-rs) |
| `wasm-bindgen` | 0.2.126 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen](https://github.com/wasm-bindgen/wasm-bindgen) |
| `wasm-bindgen-macro` | 0.2.126 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro) |
| `wasm-bindgen-macro-support` | 0.2.126 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro-support](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro-support) |
| `wasm-bindgen-shared` | 0.2.126 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/shared](https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/shared) |
| `windows-core` | 0.62.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-implement` | 0.60.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-interface` | 0.59.3 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-link` | 0.2.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-result` | 0.4.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-strings` | 0.5.1 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `windows-sys` | 0.61.2 | MIT OR Apache-2.0 | `Apache-2.0` | [https://github.com/microsoft/windows-rs](https://github.com/microsoft/windows-rs) |
| `wit-bindgen` | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/bytecodealliance/wit-bindgen](https://github.com/bytecodealliance/wit-bindgen) |
| `zerocopy` | 0.8.52 | BSD-2-Clause OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/google/zerocopy](https://github.com/google/zerocopy) |
| `zerocopy-derive` | 0.8.52 | BSD-2-Clause OR Apache-2.0 OR MIT | `Apache-2.0` | [https://github.com/google/zerocopy](https://github.com/google/zerocopy) |
| `zmij` | 1.0.21 | MIT | - | [https://github.com/dtolnay/zmij](https://github.com/dtolnay/zmij) |

## Licence texts

Apache-2.0 is reproduced in full in `LICENSE`, which is MuMDIA's own licence and
covers the Apache-2.0 dependencies as well. The remaining families are below.
Individual copyright holders are named in each crate's own repository, linked in
the table above; this file reproduces the licence terms, not per-crate
attribution lines.

### BSD-2-Clause

```text
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list
   of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT, ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

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

- `unicode-ident` -- https://github.com/dtolnay/unicode-ident

### Public-domain dedications

These impose no reproduction requirement, and are recorded so it is clear they
were considered: `0BSD`, `CC0-1.0`, `MIT-0`, `Unlicense`.

## Copyright notices

Reproduced from each crate's own `LICENSE`/`COPYING`/`NOTICE` files, because the
licence text alone does not discharge the obligation: MIT requires the copyright
notice in all copies, BSD requires it retained, and Apache-2.0 section 4(d)
requires the contents of a NOTICE file to be propagated.

Notices recovered for 126 of 173 crates (the remainder ship no copyright line in a licence file; see their repositories, linked above).

**adler2 2.0.1**

- Copyright (C) Jonas Schievink <jonasschievink@gmail.com>

**ahash 0.8.12**

- Copyright (c) 2018 Tom Kaitchuck

**aho-corasick 1.1.4**

- Copyright (c) 2015 Andrew Gallant

**android_system_properties 0.1.5**

- Copyright 2016 Nicolas Silva
- Copyright (c) 2013 Nicolas Silva

**anstream 1.0.0**

- Copyright (c) Individual contributors

**anstyle 1.0.14**

- Copyright (c) Individual contributors

**anstyle-parse 1.0.0**

- Copyright (c) Individual contributors

**anstyle-query 1.1.5**

- Copyright (c) Individual contributors

**anstyle-wincon 3.0.11**

- Copyright (c) Individual contributors

**arrayref 0.3.9**

- Copyright (c) 2015 David Roundy <roundyd@physics.oregonstate.edu>

**arrayvec 0.7.7**

- Copyright (c) Ulrik Sverdrup "bluss" 2015-2023

**arrow-array 59.0.0**

- Copyright (c) 2020-2022 Oliver Margetts

**atoi 2.0.0**

- Copyright (c) 2017

**autocfg 1.5.1**

- Copyright (c) 2018 Josh Stone

**base64 0.22.1**

- Copyright (c) 2015 Alice Maz

**bitflags 2.13.0**

- Copyright (c) 2014 The Rust Project Developers

**blake3 1.8.5**

- Copyright 2019 Jack O'Connor and Samuel Neves

**block-buffer 0.10.4**

- Copyright (c) 2018-2019 The RustCrypto Project Developers

**bumpalo 3.20.3**

- Copyright (c) 2019 Nick Fitzgerald

**bytemuck 1.25.0**

- Copyright (c) 2019 Daniel "Lokathor" Gee.

**bytes 1.12.0**

- Copyright (c) 2018 Carl Lerche

**cc 1.2.65**

- Copyright (c) 2014 Alex Crichton

**cfg-if 1.0.4**

- Copyright (c) 2014 Alex Crichton

**chrono 0.4.45**

- Copyright (c) 2014, Kang Seonghoon.

**clap 4.6.1**

- Copyright (c) Individual contributors

**clap_builder 4.6.0**

- Copyright (c) Individual contributors

**clap_derive 4.6.1**

- Copyright (c) Individual contributors

**clap_lex 1.1.0**

- Copyright (c) Individual contributors

**colorchoice 1.0.5**

- Copyright (c) Individual contributors

**comfy-table 7.2.2**

- Copyright (c) 2019 Arne Beer

**const-random 0.1.18**

- Copyright (c) 2016 Amanieu d'Antras

**const-random-macro 0.1.16**

- Copyright (c) 2016 Amanieu d'Antras

**core-foundation-sys 0.8.7**

- Copyright (c) 2012-2013 Mozilla Foundation

**cpufeatures 0.2.17**

- Copyright (c) 2020-2025 The RustCrypto Project Developers

**cpufeatures 0.3.0**

- Copyright (c) 2020-2025 The RustCrypto Project Developers

**crc32fast 1.5.0**

- Copyright (c) 2018 Sam Rijs, Alex Crichton and contributors

**crossbeam-deque 0.8.6**

- Copyright (c) 2019 The Crossbeam Project Developers

**crossbeam-epoch 0.9.20**

- Copyright (c) 2019 The Crossbeam Project Developers

**crossbeam-utils 0.8.21**

- Copyright (c) 2019 The Crossbeam Project Developers

**crunchy 0.2.4**

- Copyright 2017-2023 Eira Fransham.

**crypto-common 0.1.7**

- Copyright (c) 2021 RustCrypto Developers

**csv 1.4.0**

- Copyright (c) 2015 Andrew Gallant

**csv-core 0.1.13**

- Copyright (c) 2015 Andrew Gallant

**digest 0.10.7**

- Copyright (c) 2017 Artyom Pavlov

**either 1.16.0**

- Copyright (c) 2015

**encoding_rs 0.8.35**

- Copyright © WHATWG (Apple, Google, Mozilla, Microsoft).

**equivalent 1.0.2**

- Copyright (c) 2016--2023

**find-msvc-tools 0.1.9**

- Copyright (c) 2014 Alex Crichton

**flate2 1.1.9**

- Copyright (c) 2014-2026 Alex Crichton

**futures-core 0.3.32**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-task 0.3.32**

- Copyright (c) 2016 Alex Crichton
- Copyright (c) 2017 The Tokio Authors

**futures-util 0.3.32**

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

**hashbrown 0.17.1**

- Copyright (c) 2016 Amanieu d'Antras

**heck 0.5.0**

- Copyright (c) 2015 The Rust Project Developers

**iana-time-zone 0.1.65**

- Copyright 2020 Andrew Straw
- Copyright (c) 2020 Andrew D. Straw

**iana-time-zone-haiku 0.1.2**

- Copyright 2020 Andrew Straw
- Copyright (c) 2020 Andrew D. Straw

**identity-hash 0.1.0**

- Copyright 2018 Parity Technologies (UK) Ltd.
- Copyright 2023 Team Offsetting

**indexmap 2.14.0**

- Copyright (c) 2016--2017

**is_terminal_polyfill 1.70.2**

- Copyright (c) Individual contributors

**js-sys 0.3.103**

- Copyright (c) 2014 Alex Crichton

**lazy_static 1.5.0**

- Copyright (c) 2010 The Rust Project Developers

**lexical-core 1.0.6**

- Copyright (c) 2009 The Go Authors. All rights reserved.
- Copyright 2014, the V8 project authors. All rights reserved.
- Copyright (c) 2013 Andreas Samoljuk

**lexical-parse-float 1.0.6**

- Copyright (c) 2009 The Go Authors. All rights reserved.
- Copyright 2014, the V8 project authors. All rights reserved.
- Copyright (c) 2013 Andreas Samoljuk

**lexical-parse-integer 1.0.6**

- Copyright (c) 2009 The Go Authors. All rights reserved.
- Copyright 2014, the V8 project authors. All rights reserved.
- Copyright (c) 2013 Andreas Samoljuk

**lexical-write-float 1.0.6**

- Copyright (c) 2009 The Go Authors. All rights reserved.
- Copyright 2014, the V8 project authors. All rights reserved.
- Copyright (c) 2013 Andreas Samoljuk

**lexical-write-integer 1.0.6**

- Copyright (c) 2009 The Go Authors. All rights reserved.
- Copyright 2014, the V8 project authors. All rights reserved.
- Copyright (c) 2013 Andreas Samoljuk

**libc 0.2.186**

- Copyright (c) The Rust Project Developers

**libm 0.2.16**

- Copyright (c) 2018 Jorge Aparicio
- Copyright © 2005-2020 Rich Felker, et al.
- Copyright © 1993,2004 Sun Microsystems or
- Copyright © 2003-2011 David Schultz or
- Copyright © 2003-2009 Steven G. Kargl or
- Copyright © 2003-2009 Bruce D. Evans or
- Copyright © 2008 Stephen L. Moshier or
- Copyright © 2017-2018 Arm Limited

**libmimalloc-sys 0.1.49**

- Copyright 2019 Octavian Oncescu

**log 0.4.33**

- Copyright (c) 2014 The Rust Project Developers

**matchers 0.2.0**

- Copyright (c) 2019 Eliza Weisman

**memchr 2.8.2**

- Copyright (c) 2015 Andrew Gallant

**mimalloc 0.1.52**

- Copyright 2019 Octavian Oncescu

**miniz_oxide 0.8.9**

- Copyright 2013-2014 RAD Game Tools and Valve Software
- Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC
- Copyright (c) 2017 Frommi
- Copyright (c) 2017-2024 oyvindln
- Copyright (c) 2020 Frommi

**nu-ansi-term 0.50.3**

- Copyright (c) 2014 Benjamin Sago
- Copyright (c) 2021-2022 The Nushell Project Developers

**num-bigint 0.4.6**

- Copyright (c) 2014 The Rust Project Developers

**num-complex 0.4.6**

- Copyright (c) 2014 The Rust Project Developers

**num-integer 0.1.46**

- Copyright (c) 2014 The Rust Project Developers

**num-traits 0.2.19**

- Copyright (c) 2014 The Rust Project Developers

**once_cell_polyfill 1.70.2**

- Copyright (c) Individual contributors

**outref 0.5.2**

- Copyright (c) 2022 Nugine

**quick-xml 0.41.0**

- Copyright (c) 2016 Johann Tuffe

**r-efi 5.3.0**

- Copyright (C) 2017-2023 Red Hat, Inc.
- Copyright (C) 2019-2023 Microsoft Corporation
- Copyright (C) 2022-2023 David Rheinsberg

**rayon 1.12.0**

- Copyright (c) 2010 The Rust Project Developers

**rayon-core 1.13.0**

- Copyright (c) 2010 The Rust Project Developers

**regex 1.12.4**

- Copyright (c) 2014 The Rust Project Developers

**regex-automata 0.4.14**

- Copyright (c) 2014 The Rust Project Developers

**regex-syntax 0.8.11**

- Copyright (c) 2014 The Rust Project Developers

**rustc_version 0.4.1**

- Copyright (c) 2016 The Rust Project Developers

**sha1 0.10.6**

- Copyright (c) 2006-2009 Graydon Hoare
- Copyright (c) 2009-2013 Mozilla Foundation
- Copyright (c) 2016 Artyom Pavlov

**sharded-slab 0.1.7**

- Copyright (c) 2019 Eliza Weisman

**shlex 2.0.1**

- Copyright 2015 Nicholas Allegra (comex).
- Copyright (c) 2015 Nicholas Allegra (comex).

**simd-adler32 0.3.9**

- Copyright (c) [2021] [Marvin Countryman]

**slab 0.4.12**

- Copyright (c) 2019 Carl Lerche

**smallvec 1.15.2**

- Copyright (c) 2018 The Servo Project Developers

**snap 1.1.1**

- Copyright 2011, The Snappy-Rust Authors. All rights reserved.

**strsim 0.11.1**

- Copyright (c) 2015 Danny Guo
- Copyright (c) 2016 Titus Wormer <tituswormer@gmail.com>
- Copyright (c) 2018 Akash Kurdekar

**thread_local 1.1.9**

- Copyright (c) 2016 The Rust Project Developers

**tracing 0.1.44**

- Copyright (c) 2019 Tokio Contributors

**tracing-attributes 0.1.31**

- Copyright (c) 2019 Tokio Contributors

**tracing-core 0.1.36**

- Copyright (c) 2019 Tokio Contributors

**tracing-log 0.2.0**

- Copyright (c) 2019 Tokio Contributors

**tracing-subscriber 0.3.23**

- Copyright (c) 2019 Tokio Contributors

**twox-hash 2.1.2**

- Copyright (c) 2015 Jake Goulding

**typenum 1.20.1**

- Copyright 2014 Paho Lurie-Gregg
- Copyright (c) 2014 Paho Lurie-Gregg

**unicode-ident 1.0.24**

- Copyright © 1991-2023 Unicode, Inc.

**unicode-segmentation 1.13.3**

- Copyright (c) 2015 The Rust Project Developers

**unicode-width 0.2.2**

- Copyright (c) 2015 The Rust Project Developers

**utf8parse 0.2.2**

- Copyright (c) 2016 Joe Wilm

**version_check 0.9.5**

- Copyright (c) 2017-2018 Sergio Benitez

**wasm-bindgen 0.2.126**

- Copyright (c) 2014 Alex Crichton

**wasm-bindgen-macro 0.2.126**

- Copyright (c) 2014 Alex Crichton

**wasm-bindgen-macro-support 0.2.126**

- Copyright (c) 2014 Alex Crichton

**wasm-bindgen-shared 0.2.126**

- Copyright (c) 2014 Alex Crichton

**windows-core 0.62.2**

- Copyright (c) Microsoft Corporation.

**windows-implement 0.60.2**

- Copyright (c) Microsoft Corporation.

**windows-interface 0.59.3**

- Copyright (c) Microsoft Corporation.

**windows-link 0.2.1**

- Copyright (c) Microsoft Corporation.

**windows-result 0.4.1**

- Copyright (c) Microsoft Corporation.

**windows-strings 0.5.1**

- Copyright (c) Microsoft Corporation.

**windows-sys 0.61.2**

- Copyright (c) Microsoft Corporation.

**zerocopy 0.8.52**

- Copyright 2023 The Fuchsia Authors
- Copyright 2019 The Fuchsia Authors.

**zerocopy-derive 0.8.52**

- Copyright 2023 The Fuchsia Authors
- Copyright 2019 The Fuchsia Authors.

### NOTICE files, verbatim

Apache-2.0 section 4(d) requires these to travel with the distribution.

#### arrow 59.0.0, and 14 more

Shared by: `arrow 59.0.0`, `arrow-arith 59.0.0`, `arrow-array 59.0.0`, `arrow-buffer 59.0.0`, `arrow-cast 59.0.0`, `arrow-csv 59.0.0`, `arrow-data 59.0.0`, `arrow-ipc 59.0.0`, `arrow-json 59.0.0`, `arrow-ord 59.0.0`, `arrow-row 59.0.0`, `arrow-schema 59.0.0`, `arrow-select 59.0.0`, `arrow-string 59.0.0`, `parquet 59.0.0`.

```text
Apache Arrow
Copyright 2016-2026 The Apache Software Foundation

This product includes software developed at
The Apache Software Foundation (http://www.apache.org/).

This product includes software from the chronoutil crate (MIT)
 * Copyright (c) 2020-2022 Oliver Margetts
 * https://github.com/olliemath/chronoutil

This product includes software from the compact-thrift project (Apache 2.0)
 * Copyright Jörn Horstmann
 * https://github.com/jhorstmann/compact-thrift
```

## Python sidecars

The Python workers in `scripts/` are MuMDIA's own code under `LICENSE`. Their
dependencies (DeepLC, MS2PIP, mokapot, torch, numpy, pandas, pyarrow) are
installed by the user or by the container build from `env/*.yml` and are not
redistributed inside the binary, so their licences travel with those packages.

