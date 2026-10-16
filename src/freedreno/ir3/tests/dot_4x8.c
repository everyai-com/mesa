/*
 * Copyright © 2026 Mesa contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * CPU check of the arithmetic used by emit_alu_sdot_4x8_emulated() in
 * ir3_compiler_nir.c, which builds a signed x signed 4x8 dot product out of
 * the signed x unsigned ("mixed") mode of dp4acc/dp2acc on GPUs without a
 * compliant dp4acc.
 *
 * The instructions are modelled here, and the lowering is replayed in the
 * same order as the compiler emits it, for both the dp4acc and the dp2acc
 * variant, with and without saturation. Every pair of int8 values is tried
 * in every byte lane, with the other lanes filled with pseudo-random bytes,
 * against a set of accumulators that includes the saturation edges. Every
 * pair is also tried with all four lanes equal, which reaches the largest
 * and smallest possible dot products.
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static int32_t
s8(uint32_t x, unsigned i)
{
   return (int8_t)(x >> (8 * i));
}

static int32_t
u8(uint32_t x, unsigned i)
{
   return (uint8_t)(x >> (8 * i));
}

/* dp4acc, mixed signedness: src0 bytes signed, src1 bytes unsigned, 32-bit
 * wrapping accumulate.
 */
static uint32_t
dp4acc_mixed(uint32_t a, uint32_t b, uint32_t acc)
{
   for (unsigned i = 0; i < 4; i++)
      acc += (uint32_t)(s8(a, i) * u8(b, i));
   return acc;
}

/* dp2acc, mixed signedness: like dp4acc but only on bytes 0-1 (packed low)
 * or bytes 2-3 (packed high).
 */
static uint32_t
dp2acc_mixed(uint32_t a, uint32_t b, uint32_t acc, bool high)
{
   for (unsigned i = high ? 2 : 0; i < (high ? 4 : 2); i++)
      acc += (uint32_t)(s8(a, i) * u8(b, i));
   return acc;
}

static int32_t
add_s_sat(uint32_t x, uint32_t y)
{
   int64_t r = (int64_t)(int32_t)x + (int64_t)(int32_t)y;
   if (r > INT32_MAX)
      return INT32_MAX;
   if (r < INT32_MIN)
      return INT32_MIN;
   return r;
}

/* Mirrors emit_sudot_4x8(). */
static uint32_t
sudot_4x8(uint32_t a, uint32_t b, uint32_t acc, bool has_dp4acc)
{
   if (has_dp4acc)
      return dp4acc_mixed(a, b, acc);

   return dp2acc_mixed(a, b, dp2acc_mixed(a, b, acc, false), true);
}

/* Mirrors emit_alu_sdot_4x8_emulated(). */
static uint32_t
sdot_4x8_emulated(uint32_t a, uint32_t b, uint32_t acc, bool sat,
                  bool has_dp4acc)
{
   const uint32_t bias = 0x80808080;
   uint32_t b_biased = b ^ bias;
   uint32_t correction = sudot_4x8(a, bias, 0, has_dp4acc);
   uint32_t dot = sudot_4x8(a, b_biased, sat ? 0 : acc, has_dp4acc);
   dot = dot - correction;

   if (sat)
      dot = add_s_sat(dot, acc);

   return dot;
}

/* nir sdot_4x8_iadd / sdot_4x8_iadd_sat. */
static uint32_t
sdot_4x8_ref(uint32_t a, uint32_t b, uint32_t acc, bool sat)
{
   int64_t dot = 0;
   for (unsigned i = 0; i < 4; i++)
      dot += (int64_t)s8(a, i) * s8(b, i);

   if (sat)
      return add_s_sat((uint32_t)dot, acc);

   return (uint32_t)dot + acc;
}

static const uint32_t accs[] = {
   0,
   1,
   0xffffffff, /* -1 */
   0x7fffffff, /* INT32_MAX */
   0x80000000, /* INT32_MIN */
   0x7fff0000, /* INT32_MAX - 65535 */
   0x80010000, /* INT32_MIN + 65536 */
   0x12345678,
   0xdeadbeef,
};

static uint32_t rand_state = 0x9e3779b9;

static uint32_t
xorshift32(void)
{
   rand_state ^= rand_state << 13;
   rand_state ^= rand_state >> 17;
   rand_state ^= rand_state << 5;
   return rand_state;
}

static unsigned failures;
static unsigned checks;

static void
check(uint32_t a, uint32_t b, uint32_t acc)
{
   for (unsigned sat = 0; sat < 2; sat++) {
      uint32_t expected = sdot_4x8_ref(a, b, acc, sat);

      for (unsigned dp4 = 0; dp4 < 2; dp4++) {
         uint32_t got = sdot_4x8_emulated(a, b, acc, sat, dp4);
         checks++;

         if (got != expected) {
            if (failures++ < 16) {
               printf("FAIL: sdot_4x8_iadd%s(0x%08" PRIx32 ", 0x%08" PRIx32
                      ", 0x%08" PRIx32 ") via %s: got 0x%08" PRIx32
                      ", expected 0x%08" PRIx32 "\n",
                      sat ? "_sat" : "", a, b, acc,
                      dp4 ? "dp4acc" : "dp2acc", got, expected);
            }
         }
      }
   }
}

int
main(void)
{
   for (int x = -128; x <= 127; x++) {
      for (int y = -128; y <= 127; y++) {
         uint32_t bx = (uint8_t)x, by = (uint8_t)y;

         /* All four lanes equal: covers the extremes of the dot product. */
         for (unsigned k = 0; k < sizeof(accs) / sizeof(accs[0]); k++)
            check(bx * 0x01010101, by * 0x01010101, accs[k]);

         /* The pair in each lane on its own, other lanes random. */
         for (unsigned lane = 0; lane < 4; lane++) {
            uint32_t mask = ~(0xffu << (8 * lane));
            uint32_t a = (xorshift32() & mask) | (bx << (8 * lane));
            uint32_t b = (xorshift32() & mask) | (by << (8 * lane));

            for (unsigned k = 0; k < sizeof(accs) / sizeof(accs[0]); k++)
               check(a, b, accs[k]);
            check(a, b, xorshift32());
         }
      }
   }

   printf("%u checks, %u failures\n", checks, failures);

   return failures ? 1 : 0;
}
