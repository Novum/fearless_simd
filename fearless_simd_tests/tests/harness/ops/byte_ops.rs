// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn byte_ops_all_widths<S: Simd>(simd: S) {
    macro_rules! check_width {
        (
            $bytes:ident,
            $words:ident,
            $lanes:literal,
            $concat_swizzle:ident,
            $multishift:ident,
            $compress:ident,
            $compress_merge:ident,
            $expand:ident,
            $expand_merge:ident,
            $load_expand:ident,
            $load_expand_merge:ident
        ) => {{
            let low: [u8; $lanes] = core::array::from_fn(|lane| lane as u8);
            let high: [u8; $lanes] = core::array::from_fn(|lane| (lane + $lanes) as u8);
            let indices: [u8; $lanes] =
                core::array::from_fn(|lane| ((lane * 37 + 131) % (2 * $lanes)) as u8);
            let selected = simd.$concat_swizzle(
                $bytes::simd_from(simd, low),
                $bytes::simd_from(simd, high),
                $bytes::simd_from(simd, indices),
            );
            assert_eq!(*selected, indices);

            let data_lanes: [u64; $lanes / 8] = core::array::from_fn(|lane| {
                0x0123_4567_89ab_cdef_u64.rotate_left((lane * 11) as u32)
            });
            let bit_offsets: [u8; $lanes] =
                core::array::from_fn(|lane| (lane * 13 + (lane % 3) * 64) as u8);
            let shifted = simd.$multishift(
                $words::simd_from(simd, data_lanes),
                $bytes::simd_from(simd, bit_offsets),
            );
            let expected: [u8; $lanes] = core::array::from_fn(|lane| {
                data_lanes[lane / 8].rotate_right(u32::from(bit_offsets[lane] & 63)) as u8
            });
            assert_eq!(*shifted, expected);

            let values: [u8; $lanes] = core::array::from_fn(|lane| (lane * 3 + 1) as u8);
            let merge: [u8; $lanes] = core::array::from_fn(|lane| 0xe0_u8.wrapping_add(lane as u8));
            let mut mask = 0_u64;
            for lane in 0..$lanes {
                if lane % 3 == 0 || lane % 7 == 2 {
                    mask |= 1_u64 << lane;
                }
            }
            let values_vec = $bytes::simd_from(simd, values);
            let merge_vec = $bytes::simd_from(simd, merge);
            let compressed = simd.$compress(values_vec, mask);
            let compressed_merge = simd.$compress_merge(values_vec, mask, merge_vec);
            let mut expected = [0; $lanes];
            let mut expected_merge = merge;
            let mut output_lane = 0;
            for (input_lane, value) in values.into_iter().enumerate() {
                if mask & (1_u64 << input_lane) != 0 {
                    expected[output_lane] = value;
                    expected_merge[output_lane] = value;
                    output_lane += 1;
                }
            }
            assert_eq!(*compressed, expected);
            assert_eq!(*compressed_merge, expected_merge);

            let expanded = simd.$expand(values_vec, mask);
            let expanded_merge = simd.$expand_merge(values_vec, mask, merge_vec);
            let mut expected = [0; $lanes];
            let mut expected_merge = merge;
            let mut input_lane = 0;
            for lane in 0..$lanes {
                if mask & (1_u64 << lane) != 0 {
                    expected[lane] = values[input_lane];
                    expected_merge[lane] = values[input_lane];
                    input_lane += 1;
                }
            }
            assert_eq!(*expanded, expected);
            assert_eq!(*expanded_merge, expected_merge);

            let mut source = [0; $lanes];
            for (index, byte) in source
                .iter_mut()
                .take(mask.count_ones() as usize)
                .enumerate()
            {
                *byte = 0x80_u8.wrapping_add(index as u8);
            }
            let loaded = simd.$load_expand(&source, mask);
            let loaded_merge = simd.$load_expand_merge(&source, mask, merge_vec);
            let mut expected = [0; $lanes];
            let mut expected_merge = merge;
            let mut source_index = 0;
            for lane in 0..$lanes {
                if mask & (1_u64 << lane) != 0 {
                    expected[lane] = source[source_index];
                    expected_merge[lane] = source[source_index];
                    source_index += 1;
                }
            }
            assert_eq!(*loaded, expected);
            assert_eq!(*loaded_merge, expected_merge);
        }};
    }

    check_width!(
        u8x16,
        u64x2,
        16,
        concat_swizzle_dyn_u8x16,
        multishift_u8x16,
        compress_u8x16,
        compress_merge_u8x16,
        expand_u8x16,
        expand_merge_u8x16,
        load_expand_u8x16,
        load_expand_merge_u8x16
    );
    check_width!(
        u8x32,
        u64x4,
        32,
        concat_swizzle_dyn_u8x32,
        multishift_u8x32,
        compress_u8x32,
        compress_merge_u8x32,
        expand_u8x32,
        expand_merge_u8x32,
        load_expand_u8x32,
        load_expand_merge_u8x32
    );
    check_width!(
        u8x64,
        u64x8,
        64,
        concat_swizzle_dyn_u8x64,
        multishift_u8x64,
        compress_u8x64,
        compress_merge_u8x64,
        expand_u8x64,
        expand_merge_u8x64,
        load_expand_u8x64,
        load_expand_merge_u8x64
    );
}

#[test]
fn narrow_byte_masks_ignore_high_bits() {
    let simd = Fallback::new();
    let merge = u8x16::splat(simd, 0xa5);
    let values = u8x16::from_fn(simd, |lane| (lane + 1) as u8);
    let mask = 0xffff_ffff_ffff_0001;

    assert_eq!(*simd.compress_u8x16(values, mask), {
        let mut expected = [0; 16];
        expected[0] = 1;
        expected
    });
    assert_eq!(*simd.compress_merge_u8x16(values, mask, merge), {
        let mut expected = *merge;
        expected[0] = 1;
        expected
    });
    assert_eq!(*simd.expand_u8x16(values, mask), {
        let mut expected = [0; 16];
        expected[0] = 1;
        expected
    });
    assert_eq!(*simd.expand_merge_u8x16(values, mask, merge), {
        let mut expected = *merge;
        expected[0] = 1;
        expected
    });
    let mut source = [0; 16];
    source[0] = 7;
    assert_eq!(*simd.load_expand_u8x16(&source, mask), {
        let mut expected = [0; 16];
        expected[0] = 7;
        expected
    });
    assert_eq!(*simd.load_expand_merge_u8x16(&source, mask, merge), {
        let mut expected = *merge;
        expected[0] = 7;
        expected
    });
}
