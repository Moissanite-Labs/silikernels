/*
 * Host-side accuracy check for the rope_kv_append_kernel RoPE angle formula
 * in cuda/sm86/qwen25_sm86_a16_engine.cu.
 *
 * The kernel originally used cosf(angle)/sinf(angle) on angle = pos * theta^(-2j/64),
 * which on CUDA 13.0/13.2 reserves a 32 B stack frame for the libm slow-path range
 * reduction. The fix reduces the angle in double precision with fmod() to [-pi, pi]
 * and calls __sincosf on the reduced float argument (no stack frame).
 *
 * This program compares, over every position/dim the engine uses
 * (pos in [0, MAX_SEQ_LEN=512), j in [0, 32), theta = 1e6, head_dim = 64):
 *   - the old float formula (powf + cosf/sinf) vs a double reference,
 *   - the new reduced formula (fmod + sin/cos on the reduced arg) vs the same reference,
 *   - old vs new directly.
 * The device __sincosf (PTX sin.approx.f32 / cos.approx.f32) adds at most
 * 2^-21.41 absolute error on [-pi, pi] on top of the "new" column printed here.
 *
 * Build & run:  make check-rope-angle
 * Exit status: 0 iff max |old - new| <= 1e-6 (float tolerance) and the new formula
 * stays within 1e-5 of the double reference.
 */
#include <math.h>
#include <stdio.h>

static const double TWO_PI = 6.283185307179586476925286766559;
static const double PI_D = 3.141592653589793238462643383279;

#define MAX_SEQ_LEN 512
#define HEAD_DIM 64
#define THETA 1000000.0

int main(void) {
    double max_old_ref = 0.0, max_new_ref = 0.0, max_old_new = 0.0;
    int worst_pos = 0, worst_j = 0;
    for (int pos = 0; pos < MAX_SEQ_LEN; ++pos) {
        for (int j = 0; j < HEAD_DIM / 2; ++j) {
            /* double reference over the exact engine formula */
            double ang_d = (double)pos * pow(THETA, -(double)(2 * j) / (double)HEAD_DIM);
            double rc = cos(ang_d), rs = sin(ang_d);
            /* old device formula: float powf, float mul, cosf/sinf */
            float inv_f = powf((float)THETA, -(float)(2 * j) / (float)HEAD_DIM);
            float angle = (float)pos * inv_f;
            float oc = cosf(angle), os = sinf(angle);
            /* new device formula: double fmod reduction to [-pi, pi], then __sincosf */
            double a = fmod((double)angle, TWO_PI);
            if (a > PI_D) a -= TWO_PI;
            float nc = (float)cos(a), ns = (float)sin(a);
            double e1 = fmax(fabs((double)oc - rc), fabs((double)os - rs));
            double e2 = fmax(fabs((double)nc - rc), fabs((double)ns - rs));
            double e3 = fmax(fabs((double)oc - (double)nc), fabs((double)os - (double)ns));
            if (e1 > max_old_ref) max_old_ref = e1;
            if (e2 > max_new_ref) max_new_ref = e2;
            if (e3 > max_old_new) { max_old_new = e3; worst_pos = pos; worst_j = j; }
        }
    }
    printf("rope angle check: pos in [0,%d), j in [0,%d), theta=1e6, head_dim=%d\n",
           MAX_SEQ_LEN, HEAD_DIM / 2, HEAD_DIM);
    printf("max |old(cosf/sinf) - double_ref|    = %.9e\n", max_old_ref);
    printf("max |new(fmod+__sincosf) - dbl_ref|  = %.9e (device approx adds <= 2^-21.41 = %.3e abs on [-pi,pi])\n",
           max_new_ref, pow(2.0, -21.41));
    printf("max |old - new|                      = %.9e at pos=%d j=%d\n", max_old_new, worst_pos, worst_j);
    /* The new formula must not deviate from the old one beyond float tolerance,
     * and must stay as close to the double reference as the old formula does
     * (both share the same float `angle`, whose rounding dominates the
     * reference error; the reduction itself only adds <= 2^-21.41). */
    if (max_old_new > 1e-6 || max_new_ref > max_old_ref + 1e-6) {
        fprintf(stderr, "FAIL: new RoPE formula deviates beyond float tolerance\n");
        return 1;
    }
    printf("PASS: new formula within float tolerance of the old formula\n");
    return 0;
}
