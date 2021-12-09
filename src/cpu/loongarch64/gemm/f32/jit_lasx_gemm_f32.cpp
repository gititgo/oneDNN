/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <atomic>
#include <cmath>
#include <mutex>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/f32/gemm_utils_f32.hpp"
#include "cpu/gemm/f32/ref_gemm_f32.hpp"
#include "cpu/gemm/gemm_msan_unpoison.hpp"

#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/gemm/gemm_driver.hpp"

#include "cpu/loongarch64/gemm/f32/jit_lasx_gemm_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

#define CACHE_LINE_SIZE 64

#define STACKSIZE get_size_of_abi_save_regs()
#ifdef _WIN32
#define STACK_K_CAPACITY 128
#else
#define STACK_K_CAPACITY 8192
#endif
#define SIZE 4
#define OFFSET 32
#define BASE_SHIFT 2
#define SECOND_FETCH 14

namespace lasx_gemm_f32 {
using namespace gemm_utils;
using namespace Xbyak_loongarch;

struct xbyak_gemm_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lasx_gemm_f32_xbyak_gemm)
    xbyak_gemm_t(char isTransA, char isTransB, float beta, bool hasBias = false,
            void *code_ptr = nullptr,
            size_t code_size = 80 * Xbyak_loongarch::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , isTransA(isTransA)
        , isTransB(isTransB)
        , hasBias(hasBias)
        , is_avx2(mayiuse(lasx))
        , UNROLL_M(is_avx2 ? 16 : 8)
        , UNROLL_N(6)
        , isBeta0(beta == 0.0)
        , isBetaN(!isBeta0 && beta != 1.0)
        , PREFETCHSIZEA(128)
        , PREFETCHSIZEB((!isTransB) ? -16 : 0) {}

    // Fused multiply add; may become one or two instructions
    void fma(bool useFma, const XVReg &reg0, const XVReg &reg1, const XVReg &reg2,
            bool overWrite = false) {
        if (useFma) {
            //if (is_avx2) {
            //    vfmadd231ps(reg2, reg1, reg0);
            //} else {
            //    assert(UNROLL_M == 8);
            //    auto tent_vreg = overWrite ? reg1 : xr1;
            //    vmulps(tent_vreg, reg1, reg0);
            //    vaddps(reg2, reg2, tent_vreg);
            //}
            xvfmadd_s(reg2, reg1, reg0, reg2);
        } else {
            if (!overWrite) {
                //vmulps(xr15, reg1, reg0);
                xvfmul_s(xr15, reg1, reg0);
                //vaddps(reg2, reg2, xr15);
                xvfadd_s(reg2, reg2, xr15);
            } else {
                //vmulps(reg1, reg1, reg0);
                xvfmul_s(reg1, reg1, reg0);
                //vaddps(reg2, reg2, reg1);
                xvfadd_s(reg2, reg2, reg1);
            }
        }
    }

    // Inner kernel with k=8
    void innerkernel8(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11,
            const XVReg &reg12, const XVReg &reg13, const XVReg &reg14,
            const XVReg &reg15, const XVReg &reg16, const XVReg &reg17,
            const XVReg &reg18, const XVReg &reg19, const XVReg &reg20,
            const XVReg &reg21, const XVReg &reg22, const XVReg &reg23) {
        XVReg fmareg(0);

        if (!isDirect) {
            //prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 0) * SIZE]);
            uni_preld(0, AO1, (PREFETCHSIZEA + 0) * SIZE);
        } else {
            //prefetcht0(ptr[AO1 + LDA4]);
            preldx(0, AO1, LDA4);
        }

        for (int i = 0; i < 8; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    //vmovups(xr0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        //vmovups(xr1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                    } else {
                        //vmaskmovps(xr1, VMASK,
                        //        ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
                //add(AO1, LDA);
                add_d(AO1, AO1, LDA);
            }

            if (!isTransB) {
                //vbroadcastss(xr2, ptr[BO1 + (i - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (i - OFFSET) * SIZE);
            } else {
                //vbroadcastss(xr2, ptr[BO1 + (0 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
            }
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, xr0, xr2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, xr1, xr2, fmareg);
            }
            if (i == 0) {
                //if (!isTransB) { prefetcht0(ptr[BO1 + PREFETCHSIZEB * SIZE]); }
                if (!isTransB) { uni_preld(0, BO1, PREFETCHSIZEB * SIZE); }
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    if (i == 1) {
                        //prefetcht0(ptr[BO1 + LDB + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO1 + LDB * 1 + (i - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (1 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (isCopy) {
                //vmovups(ptr[LDA4 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                //        xr0);
                uni_xvst(xr0, LDA4, (unroll_m * i + 0 * 8 - OFFSET) * SIZE);
                if (unroll_m >= 16) {
                    //vmovups(ptr[LDA4 + (unroll_m * i + 1 * 8 - OFFSET) * SIZE],
                    //        xr1);
                    uni_xvst(xr1, LDA4, (unroll_m * i + 1 * 8 - OFFSET) * SIZE);
                }
                //if (i == 7) { sub(LDA4, -unroll_m * 8 * SIZE); }
                if (i == 7) { add_imm(LDA4, LDA4, unroll_m * 8 * SIZE, X_TMP_0); }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    if (i == 2) {
                        //prefetcht0(ptr[BO1 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO1, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO1 + LDB * 2 + (i - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO1, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (2 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (i == 7) {
                //if (!isTransB) { sub(BO1, -8 * SIZE); }
                if (!isTransB) { add_imm(BO1, BO1, 8 * SIZE, X_TMP_0); }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    //if (i == 3) { prefetcht0(ptr[BO2 + PREFETCHSIZEB * SIZE]); }
                    if (i == 3) { uni_preld(0, BO2, PREFETCHSIZEB * SIZE); }
                    //vbroadcastss(xr2, ptr[BO2 + (i - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO2, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (3 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    if (i == 4) {
                        //prefetcht0(ptr[BO2 + LDB + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO2, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO2 + LDB * 1 + (i - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO2, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (4 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    if (i == 5) {
                        //prefetcht0(ptr[BO2 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO2, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO2 + LDB * 2 + (i - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO2, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (5 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }
            if (isTransB) {
                //prefetcht0(ptr[BO1 + BO2]);
                preldx(0, BO1, BO2);
                //add(BO1, LDB);
                add_d(BO1, BO1, LDB);
            }

            if (i == 0) {
                if (unroll_m >= 4) {
                    if (!isDirect) {
                        //prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 2 * 8) * SIZE]);
                        uni_preld(0, AO1, (PREFETCHSIZEA + 2 * 8) * SIZE);
                    } else {
                        //prefetcht0(ptr[AO1 + LDA4]);
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 1 || i == 2) {
                if (unroll_m >= 8) {
                    if (!isDirect) {
                        //prefetcht0(ptr[AO1
                        //        + (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE]);
                        uni_preld(0, AO1, (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE);
                    } else {
                        //prefetcht0(ptr[AO1 + LDA4]);
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 3 || i == 4 || i == 5 || i == 6) {
                if (unroll_m >= 16) {
                    if (!isDirect) {
                        //prefetcht0(ptr[AO1
                        //        + (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE]);
                        uni_preld(0, AO1, (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE);
                    } else {
                        //prefetcht0(ptr[AO1 + LDA4]);
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 7) {
                if (!isTransB) {
                    //if (unroll_n >= 4) { sub(BO2, -8 * SIZE); }
                    if (unroll_n >=4) { addi_d(BO2, BO2, 8 * SIZE); }
                }
                if (!isTransA) {
                    //prefetcht2(ptr[AA]);
                    preld(2, AA, 0);
                    //lea(AA, ptr[AA + LDA]);
                    add_d(AA, AA, LDA);
                }
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    //vmovups(xr0,
                    //        ptr[AO1
                    //                + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                    //                        * SIZE]);
                    uni_xvld(xr0, AO1, (unroll_m * (i + 1) + 0 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr0, VMASK,
                    //        ptr[AO1
                    //                + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                    //                        * SIZE]);
                    uni_xvld(xr0, AO1, (unroll_m * (i + 1) + 0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        //vmovups(xr1,
                        //        ptr[AO1
                        //                + (unroll_m * (i + 1) + 1 * 8 - OFFSET)
                        //                        * SIZE]);
                        uni_xvld(xr1, AO1, (unroll_m * (i + 1) + 1 * 8 - OFFSET) * SIZE);
                    } else {
                        //vmaskmovps(xr1, VMASK,
                        //        ptr[AO1
                        //                + (unroll_m * (i + 1) + 1 * 8 - OFFSET)
                        //                        * SIZE]);
                        uni_xvld(xr1, AO1, (unroll_m * (i + 1) + 1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
            }
        }

        //if (!isDirect) { sub(AO1, -unroll_m * 8 * SIZE); }
        if (!isDirect) { add_imm(AO1, AO1, unroll_m * 8 * SIZE, X_TMP_0); }
        //sub(LL, 1);
        addi_d(LL, LL, -1);
    }

    // Inner kernel with k=4
    void innerkernel4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11,
            const XVReg &reg12, const XVReg &reg13, const XVReg &reg14,
            const XVReg &reg15, const XVReg &reg16, const XVReg &reg17,
            const XVReg &reg18, const XVReg &reg19, const XVReg &reg20,
            const XVReg &reg21, const XVReg &reg22, const XVReg &reg23) {
        XVReg fmareg(0);

        if (!isDirect) {
            //prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 0) * SIZE]);
            uni_preld(0, AO1, (PREFETCHSIZEA + 0) * SIZE);
        } else {
            //prefetcht0(ptr[AO1 + LDA4]);
            preldx(0, AO1, LDA4);
        }

        for (int i = 0; i < 4; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    //vmovups(xr0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        //vmovups(xr1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                    } else {
                        //vmaskmovps(xr1, VMASK,
                        //        ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
                //add(AO1, LDA);
                add_d(AO1, AO1, LDA);
            }

            if (!isTransB) {
                //vbroadcastss(xr2, ptr[BO1 + (i - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (i - OFFSET) * SIZE);
            } else {
                //vbroadcastss(xr2, ptr[BO1 + (0 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
            }
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, xr0, xr2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, xr1, xr2, fmareg);
            }
            if (i == 0) {
                //if (!isTransB) { prefetcht0(ptr[BO1 + PREFETCHSIZEB * SIZE]); }
                if (!isTransB) { uni_preld(0, BO1, PREFETCHSIZEB * SIZE); }
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    if (i == 1) {
                        //prefetcht0(ptr[BO1 + LDB + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO1 + LDB * 1 + (i - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (1 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (isCopy) {
                //vmovups(ptr[LDA4 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                //        xr0);
                uni_xvst(xr0, LDA4, (unroll_m * i + 0 * 8 - OFFSET) * SIZE);
                if (unroll_m >= 16) {
                    //vmovups(ptr[LDA4 + (unroll_m * i + 1 * 8 - OFFSET) * SIZE],
                    //        xr1);
                    uni_xvst(xr1, LDA4, (unroll_m * i + 1 * 8 - OFFSET) * SIZE);
                }
                //if (i == 3) { sub(LDA4, -unroll_m * 4 * SIZE); }
                if (i == 3) { add_imm(LDA4, LDA4, unroll_m * 4 * SIZE, X_TMP_0); }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    if (i == 2) {
                        //prefetcht0(ptr[BO1 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO1, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO1 + LDB * 2 + (i - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO1, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (2 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (i == 7) {
                //if (!isTransB) { sub(BO1, -8 * SIZE); }
                if (!isTransB) { add_imm(BO1, BO1, 8 * SIZE, X_TMP_0); }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    //if (i == 3) { prefetcht0(ptr[BO2 + PREFETCHSIZEB * SIZE]); }
                    if (i == 3) { uni_preld(0, BO2, PREFETCHSIZEB * SIZE); }
                    //vbroadcastss(xr2, ptr[BO2 + (i - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO2, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (3 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    if (i == 4) {
                        //prefetcht0(ptr[BO2 + LDB + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO2, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO2 + LDB * 1 + (i - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO2, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (4 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    if (i == 5) {
                        //prefetcht0(ptr[BO2 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO2, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO2 + LDB * 2 + (i - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO2, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (5 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }
            if (isTransB) {
                //prefetcht0(ptr[BO1 + BO2]);
                preldx(0, BO1, BO2);
                //add(BO1, LDB);
                add_d(BO1, BO1, LDB);
            }

            if (i == 0) {
                if (unroll_m >= 4) {
                    if (!isDirect) {
                        //prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 2 * 8) * SIZE]);
                        uni_preld(0, AO1, (PREFETCHSIZEA + 2 * 8) * SIZE);
                    } else {
                        //prefetcht0(ptr[AO1 + LDA4]);
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 1 || i == 2) {
                if (unroll_m >= 8) {
                    if (!isDirect) {
                        //prefetcht0(ptr[AO1
                        //        + (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE]);
                        uni_preld(0, AO1, (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE);
                    } else {
                        //prefetcht0(ptr[AO1 + LDA4]);
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 3) {
                if (!isTransB) {
                    //sub(BO1, -4 * SIZE);
                    add_imm(BO1, BO1, 4 * SIZE, X_TMP_0);
                    //if (unroll_n >= 4) { sub(BO2, -4 * SIZE); }
                    if (unroll_n >= 4) { add_imm(BO2, BO2, 4 * SIZE, X_TMP_0); }
                }
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    //vmovups(xr0,
                    //        ptr[AO1
                    //                + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                    //                        * SIZE]);
                    uni_xvld(xr0, AO1, (unroll_m * (i + 1) + 0 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr0, VMASK,
                    //        ptr[AO1
                    //                + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                    //                        * SIZE]);
                    uni_xvld(xr0, AO1, (unroll_m * (i + 1) + 0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        //vmovups(xr1,
                        //        ptr[AO1
                        //                + (unroll_m * (i + 1) + 1 * 8 - OFFSET)
                        //                        * SIZE]);
                        uni_xvld(xr1, AO1, (unroll_m * (i + 1) + 1 * 8 - OFFSET) * SIZE);
                    } else {
                        //vmaskmovps(xr1, VMASK,
                        //        ptr[AO1
                        //                + (unroll_m * (i + 1) + 1 * 8 - OFFSET)
                        //                        * SIZE]);
                        uni_xvld(xr1, AO1, (unroll_m * (i + 1) + 1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
            }
        }

        //if (!isDirect) { sub(AO1, -unroll_m * 4 * SIZE); }
        if (!isDirect) { add_imm(AO1, AO1, unroll_m * 4 * SIZE, X_TMP_0); }
    }

    // Inner kernel with k=2
    void innerkernel2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11,
            const XVReg &reg12, const XVReg &reg13, const XVReg &reg14,
            const XVReg &reg15, const XVReg &reg16, const XVReg &reg17,
            const XVReg &reg18, const XVReg &reg19, const XVReg &reg20,
            const XVReg &reg21, const XVReg &reg22, const XVReg &reg23) {
        XVReg fmareg(0);

        for (int i = 0; i < 2; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    //vmovups(xr0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        //vmovups(xr1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                    } else {
                        //vmaskmovps(xr1, VMASK,
                        //        ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
                //add(AO1, LDA);
                add_d(AO1, AO1, LDA);
            }

            if (!isTransB) {
                //vbroadcastss(xr2, ptr[BO1 + (0 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
            } else {
                //vbroadcastss(xr2, ptr[BO1 + (0 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
            }
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, xr0, xr2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, xr1, xr2, fmareg);
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    //vbroadcastss(
                    //        xr2, ptr[BO1 + LDB * 1 + (0 - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (1 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    if (i == 2) {
                        //prefetcht0(ptr[BO1 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        add_d(X_TMP_1, BO1, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    //vbroadcastss(
                    //        xr2, ptr[BO1 + LDB * 2 + (0 - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO1, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (2 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    //vbroadcastss(xr2, ptr[BO2 + (0 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO2, (0 - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (3 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    //vbroadcastss(
                    //        xr2, ptr[BO2 + LDB * 1 + (0 - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO2, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (4 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    //vbroadcastss(
                    //        xr2, ptr[BO2 + LDB * 2 + (0 - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO2, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
                } else {
                    //vbroadcastss(xr2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                    uni_xvldrepl_w(xr2, BO1, (5 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (isCopy) {
                //vmovups(ptr[LDA4 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE],
                //        xr0);
                uni_xvst(xr0, LDA4, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);
                if (unroll_m >= 16) {
                    //vmovups(ptr[LDA4 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE],
                    //        xr1);
                    uni_xvst(xr1, LDA4, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
                }
                //sub(LDA4, -unroll_m * SIZE);
                add_imm(LDA4, LDA4, unroll_m * SIZE, X_TMP_0);
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    //vmovups(xr0,
                    //        ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr0, AO1, (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr0, VMASK,
                    //        ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr0, AO1, (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        //vmovups(xr1,
                        //        ptr[AO1
                        //                + (unroll_m * 1 + 1 * 8 - OFFSET)
                        //                        * SIZE]);
                        uni_xvld(xr1, AO1, (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE);
                    } else {
                        //vmaskmovps(xr1, VMASK,
                        //        ptr[AO1
                        //                + (unroll_m * 1 + 1 * 8 - OFFSET)
                        //                        * SIZE]);
                        uni_xvld(xr1, AO1, (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
                //sub(AO1, -unroll_m * SIZE);
                add_imm(AO1, AO1, unroll_m * SIZE, X_TMP_0);
            }

            if (!isTransB) {
                //sub(BO1, -SIZE);
                add_imm(BO1, BO1, SIZE, X_TMP_0);
                //if (unroll_n >= 4) { sub(BO2, -SIZE); }
                if (unroll_n >= 4) { add_imm(BO2, BO2, SIZE, X_TMP_0); }
            } else {
                //add(BO1, LDB);
                add_d(BO1, BO1, LDB);
            }
        }
    }

    // Inner kernel with k=1
    void innerkernel1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11) {
        if (isDirect) {
            if (isLoad1Unmasked) {
                //vmovups(xr0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
            } else {
                //vmaskmovps(xr0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                xvand_v(xr0, xr0, VMASK);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    //vmovups(xr1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr1, VMASK, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                    xvand_v(xr1, xr1, VMASK);
                }
            }
            //add(AO1, LDA);
            add_d(AO1, AO1, LDA);
        }

        if (!isTransB) {
            //vbroadcastss(xr2, ptr[BO1 + (0 - OFFSET) * SIZE]);
            uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
        } else {
            //vbroadcastss(xr2, ptr[BO1 + (0 - OFFSET) * SIZE]);
            uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
        }
        fma(useFma, xr0, xr2, reg00);
        if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg06); }

        if (unroll_n >= 2) {
            if (!isTransB) {
                //vbroadcastss(xr2, ptr[BO1 + LDB * 1 + (0 - OFFSET) * SIZE]);
                add_d(X_TMP_1, BO1, LDB);
                uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
            } else {
                //vbroadcastss(xr2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (1 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg01);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg07); }
        }

        if (unroll_n >= 3) {
            if (!isTransB) {
                //vbroadcastss(xr2, ptr[BO1 + LDB * 2 + (0 - OFFSET) * SIZE]);
                add_d(X_TMP_1, BO1, LDB);
                add_d(X_TMP_1, X_TMP_1, LDB);
                uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
            } else {
                //vbroadcastss(xr2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (2 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg02);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg08); }
        }

        if (unroll_n >= 4) {
            if (!isTransB) {
                //vbroadcastss(xr2, ptr[BO2 + (0 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO2, (0 - OFFSET) * SIZE);
            } else {
                //vbroadcastss(xr2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (3 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg03);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg09); }
        }

        if (unroll_n >= 5) {
            if (!isTransB) {
                //vbroadcastss(xr2, ptr[BO2 + LDB * 1 + (0 - OFFSET) * SIZE]);
                add_d(X_TMP_1, BO2, LDB);
                uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
            } else {
                //vbroadcastss(xr2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (4 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg04);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg10); }
        }

        if (unroll_n >= 6) {
            if (!isTransB) {
                //vbroadcastss(xr2, ptr[BO2 + LDB * 2 + (0 - OFFSET) * SIZE]);
                add_d(X_TMP_1, BO2, LDB);
                add_d(X_TMP_1, X_TMP_1, LDB);
                uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
            } else {
                //vbroadcastss(xr2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                uni_xvldrepl_w(xr2, BO1, (5 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg05);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg11); }
        }

        if (isCopy) {
            //vmovups(ptr[LDA4 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE], xr0);
            uni_xvst(xr0, LDA4, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);
            if (unroll_m >= 16) {
                //vmovups(ptr[LDA4 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE],
                //        xr1);
                uni_xvst(xr1, LDA4, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
            }
            //sub(LDA4, -unroll_m * SIZE);
            add_imm(LDA4, LDA4, unroll_m * SIZE, X_TMP_0);
        }

        if (!isDirect) {
            if (isLoad1Unmasked) {
                //vmovups(xr0,
                //        ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
                uni_xvld(xr0, AO1, (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE);
            } else {
                //vmaskmovps(xr0, VMASK,
                //        ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
                uni_xvld(xr0, AO1, (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE);
                xvand_v(xr0, xr0, VMASK);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    //vmovups(xr1,
                    //        ptr[AO1 + (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr1, AO1, (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr1, VMASK,
                    //        ptr[AO1 + (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr1, AO1, (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE);
                    xvand_v(xr1, xr1, VMASK);
                }
            }
            //sub(AO1, -unroll_m * SIZE);
            add_imm(AO1, AO1, unroll_m * SIZE, X_TMP_0);
        }

        if (!isTransB) {
            //sub(BO1, -SIZE);
            add_imm(BO1, BO1, SIZE, X_TMP_0);
            //if (unroll_n >= 4) { sub(BO2, -SIZE); }
            if (unroll_n >= 4) { add_imm(BO2, BO2, SIZE, X_TMP_0); }
        } else {
            //add(BO1, LDB);
            add_d(BO1, BO1, LDB);
        }
    }

    // Main kernel; does prefetching and calls innerkernel{1,2,4,8} as
    // appropriate
    // After calculating results in registers, writes back to C matrix
    void kernel(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11,
            const XVReg &reg12, const XVReg &reg13, const XVReg &reg14,
            const XVReg &reg15, const XVReg &reg16, const XVReg &reg17,
            const XVReg &reg18, const XVReg &reg19, const XVReg &reg20,
            const XVReg &reg21, const XVReg &reg22, const XVReg &reg23) {
        if (!isDirect) {
            //lea(AO1, ptr[rsp + 256 + OFFSET * SIZE]);
            add_imm(AO1, sp, 256 + OFFSET * SIZE, X_TMP_0);
        } else {
            //mov(AO1, A);
            add_d(AO1, A, zero);
        }

        if (isCopy) {
            //lea(LDA4, ptr[rsp + 256 + OFFSET * SIZE]);
            add_imm(LDA4, sp, 256 + OFFSET * SIZE, X_TMP_0);
        } else {
            //lea(LDA4, ptr[LDA * 8 + (8 - 1 - OFFSET) * SIZE]);
            slli_d(X_TMP_1, LDA, 3);
            add_imm(LDA4, X_TMP_1, (8 - 1 - OFFSET) * SIZE, X_TMP_0);
        }

        if (isTransB) {
            //lea(BO2, ptr[LDB * 4 + (8 - 1 - OFFSET) * SIZE]);
            slli_d(X_TMP_0, LDB, 2);
            add_imm(BO2, X_TMP_0, (8 - 1 - OFFSET) * SIZE, X_TMP_1);
            //lea(BO2, ptr[BO2 + LDB * 2]);
            slli_d(X_TMP_0, LDB, 1);
            add_d(BO2, BO2, X_TMP_0);
        }

        if (!isDirect) {
            if (isLoad1Unmasked) {
                //vmovups(xr0,
                //        ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE]);
                uni_xvld(xr0, AO1, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);
            } else {
                //vmaskmovps(xr0, VMASK,
                //        ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE]);
                uni_xvld(xr0, AO1, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);
                xvand_v(xr0, xr0, VMASK);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    //vmovups(xr1,
                    //        ptr[AO1 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr1, AO1, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr1, VMASK,
                    //        ptr[AO1 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr1, AO1, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
                    xvand_v(xr1, xr1, VMASK);
                }
            }
        }

        for (int i = 4; i < 10; i++) {
            //vxorps(XVReg(i), XVReg(i), XVReg(i));
            xvxor_v(XVReg(i), XVReg(i), XVReg(i));
            //vxorps(XVReg(i + 6), XVReg(i + 6), XVReg(i + 6));
            xvxor_v(XVReg(i + 6), XVReg(i + 6), XVReg(i + 6));
        }

        //mov(LL, K);
        add_d(LL, K, zero);
        //sar(LL, 3);
        srai_d(LL, LL, 3);

        std::vector<Label> labels(8);

        //sub(LL, SECOND_FETCH);
        addi_d(LL, LL, -1 * SECOND_FETCH);
        //jle(labels[1], T_NEAR);
        bge(zero, LL, labels[1]);
        //align(16);

        L(labels[0]);
        innerkernel8(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);
        //jg(labels[0], T_NEAR);
        blt(zero, LL, labels[0]);
        //align(16);

        L(labels[1]);
        //prefetcht0(ptr[CO1 + (unroll_m - 1) * SIZE]);
        uni_preld(0, CO1, (unroll_m - 1) * SIZE);
        //if (unroll_n >= 2) prefetcht0(ptr[CO1 + LDC + (unroll_m - 1) * SIZE]);
        if (unroll_n >= 2) {
            add_d(X_TMP_1, CO1, LDC);
            uni_preld(0, X_TMP_1, (unroll_m - 1) * SIZE);
        }
        if (unroll_n >= 3) {
            //prefetcht0(ptr[CO1 + LDC * 2 + (unroll_m - 1) * SIZE]);
            add_d(X_TMP_1, CO1, LDC);
            add_d(X_TMP_1, X_TMP_1, LDC);
            uni_preld(0, X_TMP_1, (unroll_m - 1) * SIZE);
        }
        //if (unroll_n >= 4) prefetcht0(ptr[CO2 + (unroll_m - 1) * SIZE]);
        if (unroll_n >= 4) uni_preld(0, CO2, (unroll_m - 1) * SIZE);
        //if (unroll_n >= 5) prefetcht0(ptr[CO2 + LDC + (unroll_m - 1) * SIZE]);
        if (unroll_n >= 5) {
            add_d(X_TMP_1, CO2, LDC);
            uni_preld(0, X_TMP_1, (unroll_m - 1) * SIZE);
        }
        if (unroll_n >= 6) {
            //prefetcht0(ptr[CO2 + LDC * 2 + (unroll_m - 1) * SIZE]);
            add_d(X_TMP_1, CO2, LDC);
            add_d(X_TMP_1, X_TMP_1, LDC);
            uni_preld(0, X_TMP_1, (unroll_m - 1) * SIZE);
        }

        //add(LL, SECOND_FETCH);
        addi_d(LL, LL, SECOND_FETCH);
        //jle(labels[3], T_NEAR);
        bge(zero, LL, labels[3]);
        //align(16);

        L(labels[2]);
        innerkernel8(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);
        //jg(labels[2], T_NEAR);
        blt(zero, LL, labels[2]);
        //align(16);

        L(labels[3]);
        //test(K, 4);
        andi(X_TMP_0, K, 4);
        //jle(labels[4], T_NEAR);
        bge(zero, X_TMP_0, labels[4]);
        innerkernel4(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);

        L(labels[4]);
        //test(K, 2);
        andi(X_TMP_0, K, 2);
        //jle(labels[5], T_NEAR);
        bge(zero, X_TMP_0, labels[5]);
        innerkernel2(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);
        //align(16);

        L(labels[5]);
        if (unroll_m == 16) {
            if (unroll_n <= 3) {
                //vaddps(reg00, reg00, reg12);
                xvfadd_s(reg00, reg00, reg12);
                //vaddps(reg01, reg01, reg13);
                xvfadd_s(reg01, reg01, reg13);
                //vaddps(reg02, reg02, reg14);
                xvfadd_s(reg02, reg02, reg14);
                //vaddps(reg06, reg06, reg18);
                xvfadd_s(reg06, reg06, reg18);
                //vaddps(reg07, reg07, reg19);
                xvfadd_s(reg07, reg07, reg19);
                //vaddps(reg08, reg08, reg20);
                xvfadd_s(reg08, reg08, reg20);
            }
        }

        if (unroll_m <= 8) {
            //vaddps(reg00, reg00, reg12);
            xvfadd_s(reg00, reg00, reg12);
            //vaddps(reg01, reg01, reg13);
            xvfadd_s(reg01, reg01, reg13);
            //vaddps(reg02, reg02, reg14);
            xvfadd_s(reg02, reg02, reg14);
            //vaddps(reg03, reg03, reg15);
            xvfadd_s(reg03, reg03, reg15);
            //vaddps(reg04, reg04, reg16);
            xvfadd_s(reg04, reg04, reg16);
            //vaddps(reg05, reg05, reg17);
            xvfadd_s(reg05, reg05, reg17);
        }

        //test(K, 1);
        andi(X_TMP_0, K, 1);
        //jle(labels[6], T_NEAR);
        bge(zero, X_TMP_0, labels[6]);
        innerkernel1(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11);
        //align(16);

        L(labels[6]);
        //vbroadcastss(VALPHA, ALPHA);
        xvldrepl_w(VALPHA, sp, 48);

        //if (isBetaN) { vbroadcastss(VBETA, BETA); }
        if (isBetaN) { xvldrepl_w(VBETA, sp, 64); }

        // Write back the results; all beta and bias cases need to be
        // handled
        switch (unroll_n) {
            //case 1: mov(rax, LDC); break;
            case 1: add_d(t3, LDC, zero); break;
            //case 2: lea(rax, ptr[LDC * 2]); break;
            case 2: slli_d(t3, LDC, 1); break;
            //case 3: lea(rax, ptr[LDC + LDC * 2]); break;
            case 3: { mov_imm(X_TMP_0, 3); mul_d(t3, LDC, X_TMP_0); } break;
            //case 4: lea(rax, ptr[LDC + LDC * 4]); break;
            case 4: { mov_imm(X_TMP_0, 5); mul_d(t3, LDC, X_TMP_0); } break;
            case 5:
                //lea(rax, ptr[LDC * 4]);
                //add(rax, LDC);
                mov_imm(X_TMP_0, 5);
                mul_d(t3, LDC, X_TMP_0);
                break;
            case 6:
                //lea(rax, ptr[LDC + LDC * 2]);
                //add(rax, rax);
                mov_imm(X_TMP_0, 6);
                mul_d(t3, LDC, X_TMP_0);
                break;
        }

        if (hasBias) {
            //mov(BIAS1, BIAS);
            ld_d(BIAS1, BIAS.getXReg(), BIAS.getOffset());
            if (isLoad1Unmasked) {
                //vmovups(VBIAS1, ptr[BIAS1 + 0 * SIZE]);
                xvld(VBIAS1, BIAS1, 0);
            } else {
                //vmaskmovps(VBIAS1, VMASK, ptr[BIAS1 + 0 * SIZE]);
                xvld(VBIAS1, BIAS1, 0);
                xvand_v(VBIAS1, VBIAS1, VMASK);
            }
        }

        for (int i = 0; i < unroll_n; i++) {
            //vmulps(XVReg(i + 4), XVReg(i + 4), VALPHA);
            xvfmul_s(XVReg(i + 4), XVReg(i + 4), VALPHA);
            if (!isBeta0) {
                if (isLoad1Unmasked) {
                    switch (i) {
                        //case 0: vmovups(xr0, ptr[CO1 + 0 * SIZE]); break;
                        case 0: xvld(xr0, CO1, 0); break;
                        //case 1: vmovups(xr0, ptr[CO1 + LDC + 0 * SIZE]); break;
                        case 1: xvldx(xr0, CO1, LDC); break;
                        case 2:
                            //vmovups(xr0, ptr[CO1 + LDC * 2 + 0 * SIZE]);
                            slli_d(X_TMP_0, LDC, 1);
                            xvldx(xr0, CO1, X_TMP_0);
                            break;
                        //case 3: vmovups(xr0, ptr[CO2 + 0 * SIZE]); break;
                        case 3: xvld(xr0, CO2, 0); break;
                        //case 4: vmovups(xr0, ptr[CO2 + LDC + 0 * SIZE]); break;
                        case 4: xvldx(xr0, CO2, LDC); break;
                        case 5:
                            //vmovups(xr0, ptr[CO2 + LDC * 2 + 0 * SIZE]);
                            slli_d(X_TMP_0, LDC, 1);
                            xvldx(xr0, CO2, X_TMP_0);
                            break;
                    }
                } else {
                    switch (i) {
                        case 0:
                            //vmaskmovps(xr0, VMASK, ptr[CO1 + 0 * SIZE]);
                            xvld(xr0, CO1, 0);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 1:
                            //vmaskmovps(xr0, VMASK, ptr[CO1 + LDC + 0 * SIZE]);
                            xvldx(xr0, CO1, LDC);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 2:
                            //vmaskmovps(
                            //        xr0, VMASK, ptr[CO1 + LDC * 2 + 0 * SIZE]);
                            slli_d(X_TMP_0, LDC, 1);
                            xvldx(xr0, CO1, X_TMP_0);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 3:
                            //vmaskmovps(xr0, VMASK, ptr[CO2 + 0 * SIZE]);
                            xvld(xr0, CO2, 0);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 4:
                            //vmaskmovps(xr0, VMASK, ptr[CO2 + LDC + 0 * SIZE]);
                            xvldx(xr0, CO2, LDC);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 5:
                            //vmaskmovps(
                            //        xr0, VMASK, ptr[CO2 + LDC * 2 + 0 * SIZE]);
                            slli_d(X_TMP_0, LDC, 1);
                            xvldx(xr0, CO2, X_TMP_0);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                    }
                }

                if (!isBetaN) {
                    //vaddps(XVReg(i + 4), xr0, XVReg(i + 4));
                    xvfadd_s(XVReg(i + 4), xr0, XVReg(i + 4));
                } else {
                    fma(useFma, VBETA, xr0, XVReg(i + 4), true);
                }
            }
            //if (hasBias) { vaddps(XVReg(i + 4), VBIAS1, XVReg(i + 4)); }
            if (hasBias) { xvfadd_s(XVReg(i + 4), VBIAS1, XVReg(i + 4)); }
            if (isLoad1Unmasked) {
                switch (i) {
                    //case 0: vmovups(ptr[CO1 + 0 * SIZE], XVReg(i + 4)); break;
                    case 0: xvst(XVReg(i + 4), CO1, 0); break;
                    case 1:
                        //vmovups(ptr[CO1 + LDC + 0 * SIZE], XVReg(i + 4));
                        xvstx(XVReg(i + 4), CO1, LDC);
                        break;
                    case 2:
                        //vmovups(ptr[CO1 + LDC * 2 + 0 * SIZE], XVReg(i + 4));
                        slli_d(X_TMP_0, LDC, 1);
                        xvstx(XVReg(i + 4), CO1, X_TMP_0);
                        break;
                    //case 3: vmovups(ptr[CO2 + 0 * SIZE], XVReg(i + 4)); break;
                    case 3: xvst(XVReg(i + 4), CO2, 0); break;
                    case 4:
                        //vmovups(ptr[CO2 + LDC + 0 * SIZE], XVReg(i + 4));
                        xvstx(XVReg(i + 4), CO2, LDC);
                        break;
                    case 5:
                        //vmovups(ptr[CO2 + LDC * 2 + 0 * SIZE], XVReg(i + 4));
                        slli_d(X_TMP_0, LDC, 1);
                        xvstx(XVReg(i + 4), CO2, X_TMP_0);
                        break;
                }
            } else {
                switch (i) {
                    case 0:
                        //vmaskmovps(ptr[CO1 + 0 * SIZE], VMASK, XVReg(i + 4));
                        store_mask_words(XVReg(i + 4), CO1, 0, VMASK);
                        break;
                    case 1:
                        //vmaskmovps(
                        //        ptr[CO1 + LDC + 0 * SIZE], VMASK, XVReg(i + 4));
                        add_d(X_TMP_1, CO1, LDC);
                        store_mask_words(XVReg(i + 4), X_TMP_1, 0, VMASK);
                        break;
                    case 2:
                        //vmaskmovps(ptr[CO1 + LDC * 2 + 0 * SIZE], VMASK,
                        //        XVReg(i + 4));
                        add_d(X_TMP_1, CO1, LDC);
                        add_d(X_TMP_1, X_TMP_1, LDC);
                        store_mask_words(XVReg(i + 4), X_TMP_1, 0, VMASK);
                        break;
                    case 3:
                        //vmaskmovps(ptr[CO2 + 0 * SIZE], VMASK, XVReg(i + 4));
                        store_mask_words(XVReg(i + 4), CO2, 0, VMASK);
                        break;
                    case 4:
                        //vmaskmovps(
                        //        ptr[CO2 + LDC + 0 * SIZE], VMASK, XVReg(i + 4));
                        add_d(X_TMP_1, CO2, LDC);
                        store_mask_words(XVReg(i + 4), X_TMP_1, 0, VMASK);
                        break;
                    case 5:
                        //vmaskmovps(ptr[CO2 + LDC * 2 + 0 * SIZE], VMASK,
                        //        XVReg(i + 4));
                        add_d(X_TMP_1, CO2, LDC);
                        add_d(X_TMP_1, X_TMP_1, LDC);
                        store_mask_words(XVReg(i + 4), X_TMP_1, 0, VMASK);
                        break;
                }
            }

            if (unroll_m >= 16) {
                // Re-use xr4 (VBIAS2)
                if (i == 0) {
                    if (hasBias) {
                        if (isLoad1Unmasked) {
                            //vmovups(VBIAS2, ptr[BIAS1 + 8 * SIZE]);
                            uni_xvld(VBIAS2, BIAS1, 8 * SIZE);
                        } else {
                            //vmaskmovps(VBIAS2, VMASK, ptr[BIAS1 + 8 * SIZE]);
                            uni_xvld(VBIAS2, BIAS1, 8 * SIZE);
                            xvand_v(VBIAS2, VBIAS2, VMASK);
                        }
                    }
                }
                //vmulps(XVReg(i + 10), XVReg(i + 10), VALPHA);
                xvfmul_s(XVReg(i + 10), XVReg(i + 10), VALPHA);
                if (!isBeta0) {
                    if (isLoad2Unmasked) {
                        switch (i) {
                            //case 0: vmovups(xr0, ptr[CO1 + 8 * SIZE]); break;
                            case 0: uni_xvld(xr0, CO1, 8 * SIZE); break;
                            case 1:
                                //vmovups(xr0, ptr[CO1 + LDC + 8 * SIZE]);
                                add_d(X_TMP_1, CO1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                break;
                            case 2:
                                //vmovups(xr0, ptr[CO1 + LDC * 2 + 8 * SIZE]);
                                add_d(X_TMP_1, CO1, LDC);
                                add_d(X_TMP_1, X_TMP_1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                break;
                            //case 3: vmovups(xr0, ptr[CO2 + 8 * SIZE]); break;
                            case 3: uni_xvld(xr0, CO2, 8 * SIZE); break;
                            case 4:
                                //vmovups(xr0, ptr[CO2 + LDC + 8 * SIZE]);
                                add_d(X_TMP_1, CO2, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                break;
                            case 5:
                                //vmovups(xr0, ptr[CO2 + LDC * 2 + 8 * SIZE]);
                                add_d(X_TMP_1, CO2, LDC);
                                add_d(X_TMP_1, X_TMP_1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                break;
                        }
                    } else {
                        switch (i) {
                            case 0:
                                //vmaskmovps(xr0, VMASK, ptr[CO1 + 8 * SIZE]);
                                uni_xvld(xr0, CO1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 1:
                                //vmaskmovps(
                                //        xr0, VMASK, ptr[CO1 + LDC + 8 * SIZE]);
                                add_d(X_TMP_1, CO1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 2:
                                //vmaskmovps(xr0, VMASK,
                                //        ptr[CO1 + LDC * 2 + 8 * SIZE]);
                                add_d(X_TMP_1, CO1, LDC);
                                add_d(X_TMP_1, X_TMP_1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 3:
                                //vmaskmovps(xr0, VMASK, ptr[CO2 + 8 * SIZE]);
                                uni_xvld(xr0, CO2, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 4:
                                //vmaskmovps(
                                //        xr0, VMASK, ptr[CO2 + LDC + 8 * SIZE]);
                                add_d(X_TMP_1, CO2, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 5:
                                //vmaskmovps(xr0, VMASK,
                                //        ptr[CO2 + LDC * 2 + 8 * SIZE]);
                                add_d(X_TMP_1, CO2, LDC);
                                add_d(X_TMP_1, X_TMP_1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                        }
                    }
                    if (!isBetaN) {
                        //vaddps(XVReg(i + 10), xr0, XVReg(i + 10));
                        xvfadd_s(XVReg(i + 10), xr0, XVReg(i + 10));
                    } else {
                        fma(useFma, VBETA, xr0, XVReg(i + 10), true);
                    }
                }
                //if (hasBias) { vaddps(XVReg(i + 10), VBIAS2, XVReg(i + 10)); }
                if (hasBias) { xvfadd_s(XVReg(i + 10), VBIAS2, XVReg(i + 10)); }
                if (isLoad2Unmasked) {
                    switch (i) {
                        case 0:
                            //vmovups(ptr[CO1 + 8 * SIZE], XVReg(i + 10));
                            uni_xvst(XVReg(i + 10), CO1, 8 * SIZE);
                            break;
                        case 1:
                            //vmovups(ptr[CO1 + LDC + 8 * SIZE], XVReg(i + 10));
                            add_d(X_TMP_1, CO1, LDC);
                            uni_xvst(XVReg(i + 10), X_TMP_1, 8 * SIZE);
                            break;
                        case 2:
                            //vmovups(ptr[CO1 + LDC * 2 + 8 * SIZE], XVReg(i + 10));
                            add_d(X_TMP_1, CO1, LDC);
                            add_d(X_TMP_1, X_TMP_1, LDC);
                            uni_xvst(XVReg(i + 10), X_TMP_1, 8 * SIZE);
                            break;
                        case 3:
                            //vmovups(ptr[CO2 + 8 * SIZE], XVReg(i + 10));
                            uni_xvst(XVReg(i + 10), CO2, 8 * SIZE);
                            break;
                        case 4:
                            //vmovups(ptr[CO2 + LDC + 8 * SIZE], XVReg(i + 10));
                            add_d(X_TMP_1, CO2, LDC);
                            uni_xvst(XVReg(i + 10), X_TMP_1, 8 * SIZE);
                            break;
                        case 5:
                            //vmovups(ptr[CO2 + LDC * 2 + 8 * SIZE], XVReg(i + 10));
                            add_d(X_TMP_1, CO2, LDC);
                            add_d(X_TMP_1, X_TMP_1, LDC);
                            uni_xvst(XVReg(i + 10), X_TMP_1, 8 * SIZE);
                            break;
                    }
                } else {
                    switch (i) {
                        case 0:
                            //vmaskmovps(ptr[CO1 + 8 * SIZE], VMASK, XVReg(i + 10));
                            store_mask_words(XVReg(i + 10), CO1, 8 * SIZE, VMASK);
                            break;
                        case 1:
                            //vmaskmovps(ptr[CO1 + LDC + 8 * SIZE], VMASK,
                            //        XVReg(i + 10));
                            add_d(X_TMP_1, CO1, LDC);
                            store_mask_words(XVReg(i + 10), X_TMP_1, 8 * SIZE, VMASK);
                            break;
                        case 2:
                            //vmaskmovps(ptr[CO1 + LDC * 2 + 8 * SIZE], VMASK,
                            //        XVReg(i + 10));
                            add_d(X_TMP_1, CO1, LDC);
                            add_d(X_TMP_1, X_TMP_1, LDC);
                            store_mask_words(XVReg(i + 10), X_TMP_1, 8 * SIZE, VMASK);
                            break;
                        case 3:
                            //vmaskmovps(ptr[CO2 + 8 * SIZE], VMASK, XVReg(i + 10));
                            store_mask_words(XVReg(i + 10), CO2, 8 * SIZE, VMASK);
                            break;
                        case 4:
                            //vmaskmovps(ptr[CO2 + LDC + 8 * SIZE], VMASK,
                            //        XVReg(i + 10));
                            add_d(X_TMP_1, CO2, LDC);
                            store_mask_words(XVReg(i + 10), X_TMP_1, 8 * SIZE, VMASK);
                            break;
                        case 5:
                            //vmaskmovps(ptr[CO2 + LDC * 2 + 8 * SIZE], VMASK,
                            //        XVReg(i + 10));
                            add_d(X_TMP_1, CO2, LDC);
                            add_d(X_TMP_1, X_TMP_1, LDC);
                            store_mask_words(XVReg(i + 10), X_TMP_1, 8 * SIZE, VMASK);
                            break;
                    }
                }
            }
            //if (i == 2) add(CO1, rax);
            if (i == 2) add_d(CO1, CO1, t3);
        }
        //if (unroll_n >= 4) { add(CO2, rax); }
        if (unroll_n >= 4) { add_d(CO2, CO2, t3); }

        // Compute next address of B
        if (!isTransB) {
            //lea(rax, ptr[K * SIZE]);
            mov_imm(X_TMP_0, SIZE);
            mul_d(t3, K, X_TMP_0);
            switch (unroll_n) {
                case 1:
                    //add(BO1, LDB);
                    add_d(BO1, BO1, LDB);
                    //add(BO2, LDB);
                    add_d(BO2, BO2, LDB);
                    break;
                case 2:
                    //lea(BO1, ptr[BO1 + LDB * 2]);
                    add_d(BO1, BO1, LDB);
                    add_d(BO1, BO1, LDB);
                    //lea(BO2, ptr[BO2 + LDB * 2]);
                    add_d(BO2, BO2, LDB);
                    add_d(BO2, BO2, LDB);
                    break;
                case 3:
                    //lea(BO1, ptr[BO1 + LDB3]);
                    add_d(BO1, BO1, LDB3);
                    //lea(BO2, ptr[BO2 + LDB3]);
                    add_d(BO2, BO2, LDB3);
                    break;
                case 4:
                    //lea(BO1, ptr[BO1 + LDB * 4]);
                    slli_d(X_TMP_0, LDB, 2);
                    add_d(BO1, BO1, X_TMP_0);
                    //lea(BO2, ptr[BO2 + LDB * 4]);
                    add_d(BO2, BO2, X_TMP_0);
                    break;
                case 5:
                    //lea(BO1, ptr[BO1 + LDB * 4]);
                    slli_d(X_TMP_0, LDB, 2);
                    add_d(BO1, BO1, X_TMP_0);
                    //add(BO1, LDB);
                    add_d(BO1, BO1, LDB);
                    //lea(BO2, ptr[BO2 + LDB * 4]);
                    add_d(BO2, BO2, X_TMP_0);
                    //add(BO2, LDB);
                    add_d(BO2, BO2, LDB);
                    break;
                case 6:
                    //lea(BO1, ptr[BO1 + LDB3 * 2]);
                    add_d(BO1, BO1, LDB3);
                    add_d(BO1, BO1, LDB3);
                    //lea(BO2, ptr[BO2 + LDB3 * 2]);
                    add_d(BO2, BO2, LDB3);
                    add_d(BO2, BO2, LDB3);
                    break;
            }
            //sub(BO1, rax);
            sub_d(BO1, BO1, t3);
            //sub(BO2, rax);
            sub_d(BO2, BO2, t3);
        } else {
            //mov(rax, LDB);
            add_d(t3, LDB, zero);
            //imul(rax, K);
            mul_d(t3, t3, K);
            //sub(BO1, rax);
            sub_d(BO1, BO1, t3);
            //add(BO1, unroll_n * SIZE);
            add_imm(BO1, BO1, unroll_n * SIZE, X_TMP_0);
        }
    }

    void kernel_16x6(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, xr4, xr5, xr6, xr7, xr8, xr9, xr10, xr11,
                xr12, xr13, xr14, xr15, xr4, xr5, xr6, xr7, xr8, xr9,
                xr10, xr11, xr12, xr13, xr14, xr15);
    }

    void kernel_16x5(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, xr4, xr5, xr6, xr7, xr8, xr9, xr10, xr11,
                xr12, xr13, xr14, xr15, xr4, xr5, xr6, xr7, xr8, xr9,
                xr10, xr11, xr12, xr13, xr14, xr15);
    }

    void kernel_16x4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, xr4, xr5, xr6, xr7, xr8, xr9, xr10, xr11,
                xr12, xr13, xr14, xr15, xr4, xr5, xr6, xr7, xr8, xr9,
                xr10, xr11, xr12, xr13, xr14, xr15);
    }

    void kernel_16x3(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, xr4, xr5, xr6, xr7, xr8, xr9, xr10,
                xr11, xr12, xr13, xr14, xr15, xr7, xr8, xr9, xr7, xr8,
                xr9, xr13, xr14, xr15, xr13, xr14, xr15);
    }

    void kernel_16x2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_16x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_16x1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_16x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_8x6(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, xr4, xr5, xr6, xr7, xr8, xr9, xr10,
                xr11, xr12, xr13, xr14, xr15, xr10, xr11, xr12, xr13,
                xr14, xr15, xr10, xr11, xr12, xr13, xr14, xr15);
    }

    void kernel_8x5(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x6(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy);
    }

    void kernel_8x4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x6(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy);
    }

    void kernel_8x3(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, xr4, xr5, xr6, xr7, xr8, xr9, xr10,
                xr11, xr12, xr13, xr14, xr15, xr7, xr8, xr9, xr7, xr8,
                xr9, xr13, xr14, xr15, xr13, xr14, xr15);
    }

    void kernel_8x2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_8x1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    // Function for packing if needed
    void do_pack(int unroll_m, bool isLoad1Unmasked, bool isLoad2Unmasked) {
        std::vector<Label> labels(6);

        int regIdx;
        XReg reg(0);

        //mov(BO1, A);
        add_d(BO1, A, zero);
        //lea(AO1, ptr[rsp + 256 + OFFSET * SIZE]);
        addi_d(AO1, sp, 256 + OFFSET * SIZE);

        if (isTransA) {
            //lea(BO2, ptr[BO1 + LDA * 4]);
            slli_d(X_TMP_0, LDA, 2);
            add_d(BO2, BO1, X_TMP_0);
            //lea(CO1, ptr[LDA + LDA * 2]);
            sub_d(CO1, X_TMP_0, LDA);
            //vmovupd(xr7, STRIDE);
            xvld(xr7, STRIDE.getXReg(), STRIDE.getOffset());
        }

        //mov(LL, K);
        add_d(LL, K, zero);
        //sar(LL, 2);
        srai_d(LL, LL, 2);
        //jle(labels[1], T_NEAR);
        bge(zero, LL, labels[1]);
        //align(16);

        L(labels[0]);
        if (!isTransA) {
            for (int i = 0; i < 4; i++) {
                regIdx = (i % 2 == 0) ? 4 : 6;
                if (isLoad1Unmasked) {
                    //vmovups(XVReg(regIdx), ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(XVReg(regIdx), BO1, (0 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(XVReg(regIdx), VMASK,
                    //        ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(XVReg(regIdx), BO1, (0 * 8 - OFFSET) * SIZE);
                    xvand_v(XVReg(regIdx), XVReg(regIdx), VMASK);
                }
                if (unroll_m > 8) {
                    if (isLoad2Unmasked) {
                        //vmovups(XVReg(regIdx + 1),
                        //        ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                        uni_xvld(XVReg(regIdx + 1),
                                    BO1, (1 * 8 - OFFSET) * SIZE);
                    } else {
                        //vmaskmovps(XVReg(regIdx + 1), VMASK,
                        //        ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                        uni_xvld(XVReg(regIdx + 1), BO1, (1 * 8 - OFFSET) * SIZE);
                        xvand_v(XVReg(regIdx + 1), XVReg(regIdx + 1), VMASK);
                    }
                }
                //add(BO1, LDA);
                add_d(BO1, BO1, LDA);

                //vmovups(ptr[AO1 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                //        XVReg(regIdx));
                uni_xvst(XVReg(regIdx), AO1, (unroll_m * i + 0 * 8 - OFFSET) * SIZE);

                if (unroll_m > 8) {
                    //vmovups(ptr[AO1 + (unroll_m * i + 1 * 8 - OFFSET) * SIZE],
                    //        XVReg(regIdx + 1));
                    uni_xvst(XVReg(regIdx + 1),
                                AO1, (unroll_m * i + 1 * 8 - OFFSET) * SIZE);
                }
            }

        } else {
            if (isLoad1Unmasked) {
                for (int i = 0; i < 2; i++) {
                    reg = (i % 2 == 0) ? BO1 : BO2;
                    //vmovups(vr0, ptr[reg + (0 * 8 - OFFSET) * SIZE]);
                    vld(vr0, reg, (0 * 8 - OFFSET) * SIZE);

                    //vmovups(vr1, ptr[reg + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                    add_d(X_TMP_0, reg, LDA);
                    vld(vr1, X_TMP_0, (0 * 8 - OFFSET) * SIZE);

                    //lea(BO2, ptr[reg + LDA * 2]);
                    add_d(X_TMP_0, LDA, LDA);
                    add_d(BO2, reg, X_TMP_0);

                    //vunpcklps(vr4, vr0, vr1);
                    vilvl_w(vr4, vr0, vr1);

                    //vunpckhps(vr5, vr0, vr1);
                    vilvh_w(vr5, vr1, vr0);

                    //vmovups(vr0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                    vld(vr0, BO2, (0 * 8 - OFFSET) * SIZE);

                    //vmovups(vr1, ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                    add_d(X_TMP_0, BO2, LDA);
                    vld(vr1, X_TMP_0, (0 * 8 - OFFSET) * SIZE);

                    //lea(BO2, ptr[BO2 + LDA * 2]);
                    add_d(X_TMP_0, LDA, LDA);
                    add_d(BO2, BO2, X_TMP_0);

                    //vunpcklps(vr6, vr0, vr1);
                    vilvl_w(vr6, vr0, vr1);

                    //vunpckhps(vr2, vr0, vr1);
                    vilvh_w(vr2, vr1, vr0);

                    //vunpcklpd(vr0, vr4, vr6);
                    vilvl_d(vr0, vr6, vr4);

                    //vunpckhpd(vr1, vr4, vr6);
                    vilvh_d(vr1, vr6, vr4);

                    //vmovups(ptr[AO1 + (unroll_m * 0 + i * 4 - OFFSET) * SIZE],
                    //        vr0);
                    vst(vr0, AO1, (unroll_m * 0 + i * 4 - OFFSET) * SIZE);

                    //vmovups(ptr[AO1 + (unroll_m * 1 + i * 4 - OFFSET) * SIZE],
                    //        vr1);
                    vst(vr1, AO1, (unroll_m * 1 + i * 4 - OFFSET) * SIZE);

                    //vunpcklpd(vr0, vr5, vr2);
                    vilvl_d(vr0, vr2, vr5);

                    //vunpckhpd(vr1, vr5, vr2);
                    vilvh_d(vr1, vr2, vr5);

                    //vmovups(ptr[AO1 + (unroll_m * 2 + i * 4 - OFFSET) * SIZE],
                    //        vr0);
                    vst(vr0, AO1, (unroll_m * 2 + i * 4 - OFFSET) * SIZE);

                    //vmovups(ptr[AO1 + (unroll_m * 3 + i * 4 - OFFSET) * SIZE],
                    //        vr1);
                    vst(vr1, AO1, (unroll_m * 3 + i * 4 - OFFSET) * SIZE);
                }
            } else if (is_avx2) {
                for (int i = 0; i < 2; i++) {
                    //vmovaps(vr4, vr3);
                    vbsll_v(vr4, vr3, 0);

                    //vgatherqps(vr0,
                    //        ptr[BO1 + xr7 + ((2 * i) - OFFSET) * SIZE], vr4);
                    vgatherqps(vr0, BO1, xr7, ((2 * i) - OFFSET) * SIZE, vr4);

                    //vmovaps(vr4, vr3);
                    vbsll_v(vr4, vr3, 0);

                    //vgatherqps(vr1,
                    //        ptr[BO1 + xr7 + ((2 * i + 1) - OFFSET) * SIZE],
                    //        vr4);
                    vgatherqps(vr1, BO1, xr7, ((2 * i + 1) - OFFSET) * SIZE, vr4);

                    //vmovups(ptr[AO1 + (unroll_m * (2 * i) + 0 * 4 - OFFSET) * SIZE],
                    //        vr0);
                    uni_xvst(vr0, AO1, (unroll_m * (2 * i) + 0 * 4 - OFFSET) * SIZE);

                    //vmovups(ptr[AO1 + (unroll_m * (2 * i + 1) + 0 * 4 - OFFSET) * SIZE],
                    //        vr1);
                    uni_xvst(vr1, AO1, (unroll_m * (2 * i + 1) + 0 * 4 - OFFSET) * SIZE);
                }

                //lea(BO2, ptr[BO1 + LDA * 4]);
                slli_d(X_TMP_0, LDA, 2);
                add_d(BO2, BO1, X_TMP_0);

                for (int i = 0; i < 2; i++) {
                    //vextractf128(vr4, xr3, 1);
                    xvpermi_q(xr4, xr3, 0x31);

                    //vgatherqps(vr0,
                    //        ptr[BO2 + xr7 + ((2 * i) - OFFSET) * SIZE], vr4);
                    vgatherqps(vr0, BO2, xr7, ((2 * i) - OFFSET) * SIZE, vr4);
                    
                    //vextractf128(vr4, xr3, 1);
                    xvpermi_q(xr4, xr3, 0x31);
                    
                    //vgatherqps(vr1,
                    //        ptr[BO2 + xr7 + ((2 * i + 1) - OFFSET) * SIZE],
                    //        vr4);
                    vgatherqps(vr1, BO2, xr7, ((2 * i + 1) - OFFSET) * SIZE, vr4);

                    //vmovups(ptr[AO1 + (unroll_m * (2 * i) + 1 * 4 - OFFSET) * SIZE], vr0);
                    uni_xvst(vr0, AO1, (unroll_m * (2 * i) + 1 * 4 - OFFSET) * SIZE);

                    //vmovups(ptr[AO1 + (unroll_m * (2 * i + 1) + 1 * 4 - OFFSET) * SIZE], vr1);
                    uni_xvst(vr1, AO1, (unroll_m * (2 * i + 1) + 1 * 4 - OFFSET) * SIZE);
                }

                //lea(BO2, ptr[BO2 + LDA * 4]);
                slli_d(X_TMP_0, LDA, 2);
                add_d(BO2, BO2, X_TMP_0);

            } /* else {
                //vxorps(vr4, vr4, vr4);
                //lea(BO2, ptr[BO1 + LDA * 4]);

                auto el_cp = [&](int section, int ld_step) {
                    RegExp src_addr = section == 0 ? BO1 : BO2;
                    if (ld_step == 1 || ld_step == 2)
                        src_addr = src_addr + LDA * ld_step;
                    else if (ld_step == 3)
                        src_addr = src_addr + CO1;
                    src_addr = src_addr - OFFSET * SIZE;

                    //vmovups(Xmm(ld_step % 2), ptr[src_addr]);
                    vld(VReg(ld_step % 2), zero, src_addr);

                    RegExp dst_addr
                            = AO1 + (ld_step + section * 4 - OFFSET) * SIZE;
                    for (int off = 0; off < 4; ++off)
                        pextrd(ptr[dst_addr + unroll_m * off * SIZE],
                                Xmm(ld_step % 2), off);
                };

                el_cp(0, 0);
                //cmp(M, 4 * 0 + 0 + 1);
                mov_imm(X_TMP_0, 4 * 0 + 0 + 1);
                //je(labels[4], T_NEAR);
                beq(M, X_TMP_0, labels[4]);
                el_cp(0, 1);
                //cmp(M, 4 * 0 + 1 + 1);
                mov_imm(X_TMP_0, 4 * 0 + 1 + 1);
                //je(labels[4], T_NEAR);
                beq(M, X_TMP_0, labels[4]);
                el_cp(0, 2);
                //cmp(M, 4 * 0 + 2 + 1);
                mov_imm(X_TMP_0, 4 * 0 + 2 + 1);
                //je(labels[4], T_NEAR);
                beq(M, X_TMP_0, labels[4]);
                el_cp(0, 3);
                //cmp(M, 4 * 0 + 3 + 1);
                mov_imm(X_TMP_0, 4 * 0 + 3 + 1);
                //je(labels[4], T_NEAR);
                beq(M, X_TMP_0, labels[4]);
                el_cp(1, 0);
                //cmp(M, 4 * 1 + 0 + 1);
                mov_imm(X_TMP_0, 4 * 1 + 0 + 1);
                //je(labels[4], T_NEAR);
                beq(M, X_TMP_0, labels[4]);
                el_cp(1, 1);
                //cmp(M, 4 * 1 + 1 + 1);
                mov_imm(X_TMP_0, 4 * 1 + 1 + 1);
                //je(labels[4], T_NEAR);
                beq(M, X_TMP_0, labels[4]);
                el_cp(1, 2);
                L(labels[4]);

                //lea(BO2, ptr[BO2 + LDA * 4]);
            }*/

            if (unroll_m >= 16) {
                assert(is_avx2);

                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        //vmovups(vr0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        uni_xvld(vr0, BO2, (0 * 8 - OFFSET) * SIZE);

                        //vmovups(vr1, ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        add_d(X_TMP_0, BO2, LDA);
                        uni_xvld(vr1, X_TMP_0, (0 * 8 - OFFSET) * SIZE);

                        //lea(BO2, ptr[BO2 + LDA * 2]);
                        add_d(X_TMP_0, LDA, LDA);
                        add_d(BO2, BO2, X_TMP_0);

                        //vunpcklps(vr4, vr0, vr1);
                        vilvl_w(vr4, vr0, vr1);

                        //vunpckhps(vr5, vr0, vr1);
                        vilvh_w(vr5, vr1, vr0);
                        
                        //vmovups(vr0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        uni_xvld(vr0, BO2, (0 * 8 - OFFSET) * SIZE);
                        
                        //vmovups(vr1, ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        add_d(X_TMP_0, BO2, LDA);
                        uni_xvld(vr1, X_TMP_0, (0 * 8 - OFFSET) * SIZE);

                        if (i == 0){
                            //lea(BO2, ptr[BO2 + LDA * 2]);
                            add_d(X_TMP_0, LDA, LDA);
                            add_d(BO2, BO2, X_TMP_0);
                        }
                        //vunpcklps(vr6, vr0, vr1);
                        vilvl_w(vr6, vr0, vr1);

                        //vunpckhps(vr2, vr0, vr1);
                        vilvh_w(vr2, vr1, vr0);

                        //vunpcklpd(vr0, vr4, vr6);
                        vilvl_d(vr0, vr6, vr4);

                        //vunpckhpd(vr1, vr4, vr6);
                        vilvh_d(vr1, vr6, vr4);

                        //vmovups(ptr[AO1 + (unroll_m * 0 + (i + 2) * 4 - OFFSET) * SIZE],
                        //        vr0);
                        uni_xvst(vr0, AO1, (unroll_m * 0 + (i + 2) * 4 - OFFSET) * SIZE);
                        
                        //vmovups(ptr[AO1 + (unroll_m * 1 + (i + 2) * 4 - OFFSET) * SIZE],
                        //        vr1);
                        uni_xvst(vr1, AO1, (unroll_m * 1 + (i + 2) * 4 - OFFSET) * SIZE);
                        
                        //vunpcklpd(vr0, vr5, vr2);
                        vilvl_d(vr0, vr2, vr5);

                        //vunpckhpd(vr1, vr5, vr2);
                        vilvh_d(vr1, vr2, vr5);
                        
                        //vmovups(ptr[AO1 + (unroll_m * 2 + (i + 2) * 4 - OFFSET) * SIZE],
                        //        vr0);
                        uni_xvst(vr0, AO1, (unroll_m * 2 + (i + 2) * 4 - OFFSET) * SIZE);
                        
                        //vmovups(ptr[AO1 + (unroll_m * 3 + (i + 2) * 4 - OFFSET) * SIZE],
                        //        vr1);
                        uni_xvst(vr1, AO1, (unroll_m * 3 + (i + 2) * 4 - OFFSET) * SIZE);
                    }
                } else {
                    for (int i = 0; i < 2; i++) {
                        //vmovaps(vr4, vr3);
                        vbsll_v(vr4, vr3, 0);

                        //vgatherqps(vr0,
                        //        ptr[BO2 + xr7 + ((2 * i) - OFFSET) * SIZE],
                        //        vr4);
                        vgatherqps(vr1, BO2, xr7, ((2 * i) - OFFSET) * SIZE, vr4);

                        //vmovaps(vr4, vr3);
                        vbsll_v(vr4, vr3, 0);

                        //vgatherqps(vr1,
                        //        ptr[BO2 + xr7 + ((2 * i + 1) - OFFSET) * SIZE],
                        //        vr4);
                        vgatherqps(vr1, BO2, xr7, ((2 * i + 1) - OFFSET) * SIZE, vr4);

                        //vmovups(ptr[AO1 + (unroll_m * (2 * i) + 2 * 4 - OFFSET) * SIZE],
                        //        vr0);
                        uni_xvst(vr0, AO1, (unroll_m * (2 * i) + 2 * 4 - OFFSET) * SIZE);

                        //vmovups(ptr[AO1 + (unroll_m * (2 * i + 1) + 2 * 4 - OFFSET) * SIZE],
                        //        vr1);
                        uni_xvst(vr1, AO1, (unroll_m * (2 * i + 1) + 2 * 4 - OFFSET) * SIZE);
                    }

                    //lea(BO2, ptr[BO2 + LDA * 4]);
                    slli_d(X_TMP_0, LDA, 2);
                    add_d(BO2, BO2, X_TMP_0);

                    for (int i = 0; i < 2; i++) {
                        //vextractf128(vr4, xr3, 1);
                        xvpermi_q(xr4, xr3, 0x31);

                        //vgatherqps(vr0,
                        //        ptr[BO2 + xr7 + ((2 * i) - OFFSET) * SIZE],
                        //        vr4);
                        vgatherqps(vr0, BO2, xr7, ((2 * i) - OFFSET) * SIZE, vr4);
                        
                        //vextractf128(vr4, xr3, 1);
                        xvpermi_q(xr4, xr3, 0x31);

                        //vgatherqps(vr1,
                        //        ptr[BO2 + xr7 + ((2 * i + 1) - OFFSET) * SIZE],
                        //        vr4);
                        vgatherqps(vr1, BO2, xr7, ((2 * i + 1) - OFFSET) * SIZE, vr4);

                        //vmovups(ptr[AO1 + (unroll_m * (2 * i) + 3 * 4 - OFFSET) * SIZE],
                        //        vr0);
                        uni_xvst(vr0, AO1, (unroll_m * (2 * i) + 3 * 4 - OFFSET) * SIZE);

                        //vmovups(ptr[AO1 + (unroll_m * (2 * i + 1) + 3 * 4 - OFFSET) * SIZE],
                        //        vr1);
                        uni_xvst(vr1, AO1, (unroll_m * (2 * i + 1) + 3 * 4 - OFFSET) * SIZE);
                    }

                    //lea(BO2, ptr[BO2 + LDA * 4]);
                    slli_d(X_TMP_0, LDA, 2);
                    add_d(BO2, BO2, X_TMP_0);
                }
            }
            //add(BO1, (4 * SIZE));
            addi_d(BO1, BO1, (4 * SIZE));
        }

        //add(AO1, unroll_m * 4 * SIZE);
        add_imm(AO1, AO1, unroll_m * 4 * SIZE, X_TMP_0);
        //sub(LL, 1);
        addi_d(LL, LL, -1);
        //jg(labels[0], T_NEAR);
        blt(zero, LL, labels[0]);
        //align(16);

        L(labels[1]);
        //mov(LL, K);
        add_d(LL, K, zero);
        //and_(LL, 3);
        andi(LL, LL, 3);
        //jle(labels[3], T_NEAR);
        bge(zero, LL, labels[3]);
        //align(16);

        L(labels[2]);
        if (!isTransA) {
            if (isLoad1Unmasked) {
                //vmovups(xr4, ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                uni_xvld(xr4, BO1, (0 * 8 - OFFSET) * SIZE);
            } else {
                //vmaskmovps(xr4, VMASK, ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                uni_xvld(xr4, BO1, (0 * 8 - OFFSET) * SIZE);
                xvand_v(xr4, xr4, VMASK);
            }
            if (unroll_m > 8) {
                if (isLoad2Unmasked) {
                    //vmovups(xr5, ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                    uni_xvld(xr5, BO1, (1 * 8 - OFFSET) * SIZE);
                } else {
                    //vmaskmovps(xr5, VMASK, ptr[BO1 + (1 + 8 - OFFSET) * SIZE]);
                    uni_xvld(xr5, BO1, (1 * 8 - OFFSET) * SIZE);
                    xvand_v(xr5, xr5, VMASK);
                }
            }

            //add(BO1, LDA);
            add_d(BO1, BO1, LDA);
            //vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE], xr4);
            uni_xvst(xr4, AO1, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);

            if (unroll_m > 8) {
                //vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE],
                //        xr5);
                uni_xvst(xr5, AO1, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
            }
        } else {
            if (isLoad1Unmasked) {
                for (int i = 0; i < 2; i++) {
                    reg = (i % 2 == 0) ? BO1 : BO2;
                    //vmovss(Xmm(i + 1), ptr[reg + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(VReg(i + 1), reg, (0 * 8 - OFFSET) * SIZE);
                    //vmovss(vr0, ptr[reg + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                    add_d(X_TMP_1, reg, LDA);
                    uni_xvld(vr0, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                    //lea(BO2, ptr[reg + LDA * 2]);
                    add_d(BO2, X_TMP_1, LDA);

                    //vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                    vilvl_w(VReg(i + 1), VReg(i + 1), vr0);
                }
                //vunpcklpd(vr1, vr1, vr2);
                vilvl_d(vr1, vr2, vr1);

                //vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE],
                //        vr1);
                vst(vr1, AO1, (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE);

                for (int i = 0; i < 2; i++) {
                    //vmovss(Xmm(i + 1), ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                    uni_xvld(VReg(i + 1), BO2, (0 * 8 - OFFSET) * SIZE);
                    //vmovss(vr0, ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                    add_d(X_TMP_1, BO2, LDA);
                    uni_xvld(vr0, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                    //lea(BO2, ptr[BO2 + LDA * 2]);
                    add_d(BO2, X_TMP_1, LDA);

                    //vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                    vilvl_w(VReg(i + 1), VReg(i + 1), vr0);
                }
                //vunpcklpd(vr1, vr1, vr2);
                vilvl_d(vr1, vr2, vr1);

                //vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE],
                //        vr1);
                uni_xvst(vr1, AO1, (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE);
            } else if (is_avx2) {
                //vmovaps(vr4, vr3);
                vbsll_v(vr4, vr3, 0);

                //vgatherqps(
                //        vr1, ptr[BO1 + xr7 + (0 * 8 - OFFSET) * SIZE], vr4);
                vgatherqps(vr1, BO1, xr7, (0 * 8 - OFFSET) * SIZE, vr4);
                //lea(BO2, ptr[BO1 + LDA * 4]);
                slli_d(X_TMP_0, LDA, 2);
                add_d(BO1, BO1, X_TMP_0);

                //vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE],
                //        vr1);
                uni_xvst(vr1, AO1, (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE);

                //vextractf128(vr4, xr3, 1);
                xvpermi_q(xr4, xr3, 0x31);

                //vgatherqps(
                //        vr1, ptr[BO2 + xr7 + (0 * 8 - OFFSET) * SIZE], vr4);
                vgatherqps(vr1, BO2, xr7, (0 * 8 - OFFSET) * SIZE, vr4);
                //lea(BO2, ptr[BO2 + LDA * 4]);
                slli_d(X_TMP_0, LDA, 2);
                add_d(BO2, BO2, X_TMP_0);
                //vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE],
                //        vr1);
                uni_xvst(vr1, AO1, (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE);
            } /*else {
                //vxorps(vr4, vr4, vr4);
                //lea(BO2, ptr[BO1 + LDA * 4]);

                auto el_cp = [&](int section, int ld_step) {
                    RegExp src_addr = section == 0 ? BO1 : BO2;
                    if (ld_step == 1 || ld_step == 2)
                        src_addr = src_addr + LDA * ld_step;
                    else if (ld_step == 3)
                        src_addr = src_addr + CO1;
                    src_addr = src_addr - OFFSET * SIZE;

                    //vmovss(vr1, ptr[src_addr]);
                    RegExp dst_addr
                            = AO1 + (ld_step + section * 4 - OFFSET) * SIZE;
                    //movss(ptr[dst_addr], vr1);
                };

                el_cp(0, 0);
                //cmp(M, 4 * 0 + 0 + 1);
                mov_imm(X_TMP_0, 4 * 0 + 0 + 1);
                //je(labels[5], T_NEAR);
                beq(M, X_TMP_0, labels[5]);
                el_cp(0, 1);
                //cmp(M, 4 * 0 + 1 + 1);
                mov_imm(X_TMP_0, 4 * 0 + 1 + 1);
                //je(labels[5], T_NEAR);
                beq(M, X_TMP_0, labels[5]);
                el_cp(0, 2);
                //cmp(M, 4 * 0 + 2 + 1);
                mov_imm(X_TMP_0, 4 * 0 + 2 + 1);
                //je(labels[5], T_NEAR);
                beq(M, X_TMP_0, labels[5]);
                el_cp(0, 3);
                //cmp(M, 4 * 0 + 3 + 1);
                mov_imm(X_TMP_0, 4 * 0 + 3 + 1);
                //je(labels[5], T_NEAR);
                beq(M, X_TMP_0, labels[5]);
                el_cp(1, 0);
                //cmp(M, 4 * 1 + 0 + 1);
                mov_imm(X_TMP_0, 4 * 1 + 0 + 1);
                //je(labels[5], T_NEAR);
                beq(M, X_TMP_0, labels[5]);
                el_cp(1, 1);
                //cmp(M, 4 * 1 + 1 + 1);
                mov_imm(X_TMP_0, 4 * 1 + 1 + 1);
                //je(labels[5], T_NEAR);
                beq(M, X_TMP_0, labels[5]);
                el_cp(1, 2);
                L(labels[5]);

                //lea(BO2, ptr[BO2 + LDA * 4]);
            }*/

            if (unroll_m >= 16) {
                assert(is_avx2);

                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        //vmovss(Xmm(i + 1), ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        uni_xvld(VReg(i + 1), BO2, (0 * 8 - OFFSET) * SIZE);
                        //vmovss(vr0,
                        //        ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        add_d(X_TMP_1, BO2, LDA);
                        uni_xvld(vr0, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                        //lea(BO2, ptr[BO2 + LDA * 2]);
                        add_d(BO2, X_TMP_1, LDA);

                        //vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                        vilvl_w(VReg(i + 1), VReg(i + 1), vr0);
                    }
                    //vunpcklpd(vr1, vr1, vr2);
                    vilvl_d(vr1, vr2, vr1);
                } else {
                    //vmovaps(vr4, vr3);
                    vbsll_v(vr4, vr3, 0);

                    //vgatherqps(vr1, ptr[BO2 + xr7 + (0 * 8 - OFFSET) * SIZE],
                    //        vr4);
                    vgatherqps(vr1, BO2, xr7, (0 * 8 - OFFSET) * SIZE, vr4);
                    //lea(BO2, ptr[BO2 + LDA * 4]);
                    slli_d(X_TMP_0, LDA, 2);
                    add_d(BO2, BO2, X_TMP_0);
                }
                //vmovups(ptr[AO1 + (unroll_m * 0 + 2 * 4 - OFFSET) * SIZE],
                //        vr1);
                uni_xvst(vr1, AO1, (unroll_m * 0 + 2 * 4 - OFFSET) * SIZE);

                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        //vmovss(Xmm(i + 1), ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        uni_xvld(VReg(i + 1), BO2, (0 * 8 - OFFSET) * SIZE);
                        //vmovss(vr0,
                        //        ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        add_d(X_TMP_1, BO2, LDA);
                        uni_xvld(vr0, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                        //lea(BO2, ptr[BO2 + LDA * 2]);
                        add_d(BO2, X_TMP_1, LDA);

                        //vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                        vilvl_w(VReg(i + 1), vr0, VReg(i + 1));
                    }
                    //vunpcklpd(vr1, vr1, vr2);
                    vilvl_d(vr1, vr2, vr1);
                } else {
                    //vextractf128(vr4, xr3, 1);
                    xvpermi_q(xr4, xr3, 0x31);

                    //vgatherqps(vr1, ptr[BO2 + xr7 + (0 * 8 - OFFSET) * SIZE],
                    //        vr4);
                    vgatherqps(vr1, BO2, xr7, (0 * 8 - OFFSET) * SIZE, vr4);
                }
                //vmovups(ptr[AO1 + (unroll_m * 0 + 3 * 4 - OFFSET) * SIZE],
                //        vr1);
                uni_xvst(vr1, AO1, (unroll_m * 0 + 3 * 4 - OFFSET) * SIZE);
            }
            //add(BO1, SIZE);
            addi_d(BO1, BO1, SIZE);
        }

        //add(AO1, unroll_m * SIZE);
        addi_d(AO1, AO1, unroll_m * SIZE);
        //sub(LL, 1);
        addi_d(LL, LL, -1);
        //jg(labels[2], T_NEAR);
        blt(zero, LL, labels[2]);
        //align(16);

        L(labels[3]);
    }


    // High-level subroutine; does packing if needed, then splits C matrix.
    // Operates on chunks of 16 rows, 6 columns at a time (handling tail
    // cases appropriately).
    // Masking is used for tail cases where M is not divisible by 8.
    void subloop(int unroll_m, bool isLoad1Unmasked, bool isLoad2Unmasked) {
        std::vector<Label> labels(15);

        if (isTransA) { do_pack(unroll_m, isLoad1Unmasked, isLoad2Unmasked); }

        //mov(CO1, C);
        ld_d(CO1, C.getXReg(), C.getOffset());
        //lea(CO2, ptr[CO1 + LDC * 2]);
        add_d(CO2, CO1, LDC);
        add_d(CO2, CO2, LDC);
        //add(CO2, LDC);
        add_d(CO2, CO2, LDC);
        //add(C, unroll_m * SIZE);
        add_imm(X_TMP_0, CO1, unroll_m * SIZE, X_TMP_0);
        st_d(X_TMP_0, C.getXReg(), C.getOffset());
        //mov(BO1, B);
        add_d(BO1, B, zero);
        //if (!isTransB) { lea(BO2, qword[B + LDB3]); }
        if (!isTransB) { add_d(BO2, B, LDB3); }

        if (!isTransA) {
            //lea(AA, ptr[A + (unroll_m * 2 - 1 - OFFSET) * SIZE]);
            add_imm(AA, A, (unroll_m * 2 - 1 - OFFSET) * SIZE, X_TMP_0);
            //cmp(M, UNROLL_M);
            mov_imm(X_TMP_0, UNROLL_M);
            //jg(labels[13], T_NEAR);
            ld_d(X_TMP_1, M.getXReg(), M.getOffset());
            blt(X_TMP_0, X_TMP_1, labels[13]);

            //mov(AA, ORIG_A);
            ld_d(AA, ORIG_A.getXReg(), ORIG_A.getOffset());
            //lea(AA, ptr[AA + (unroll_m - 1 - OFFSET) * SIZE]);
            add_imm(AA, AA, (unroll_m - 1 - OFFSET) * SIZE, X_TMP_0);
            L(labels[13]);
        }

        //mov(LL, N);
        ld_d(LL, N.getXReg(), N.getOffset());
        //mov(I, LL);
        st_d(LL, I.getXReg(), I.getOffset());
        if (!isTransA) {
            // If N is too small, skip copy operation
            //cmp(LL, UNROLL_N * 3);
            mov_imm(X_TMP_0, UNROLL_N * 3);
            //jle(labels[7], T_NEAR);
            bge(X_TMP_0, LL, labels[7]);

            // If A is not aligned to cache line
            //cmp(FLAG, 0);
            //je(labels[7], T_NEAR);
            ld_d(X_TMP_0, FLAG.getXReg(), FLAG.getOffset());
            beqz(X_TMP_0, labels[7]);
        } else {
            //cmp(LL, UNROLL_N);
            mov_imm(X_TMP_0, UNROLL_N);
            //jl(labels[1], T_NEAR);
            blt(LL, X_TMP_0, labels[1]);
        }
        //align(16);

        if (!isTransA) {
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, true, true);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        true, true);
            }
        } else {
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, false, false);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            }
        }

        //sub(I, UNROLL_N);
        ld_d(X_TMP_0, I.getXReg(), I.getOffset());
        add_imm(X_TMP_1, X_TMP_0, -1 * UNROLL_N, X_TMP_1);
        st_d(X_TMP_1, I.getXReg(), I.getOffset());
        //cmp(I, UNROLL_N);
        mov_imm(X_TMP_0, UNROLL_N);
        //jl(labels[1], T_NEAR);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        blt(X_TMP_1, X_TMP_0, labels[1]);
        //align(16);

        L(labels[0]);
        if (unroll_m == 16) {
            kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                    false, false);
        } else {
            kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                    false, false);
        }
        //sub(I, UNROLL_N);
        ld_d(X_TMP_0, I.getXReg(), I.getOffset());
        add_imm(X_TMP_1, X_TMP_0, -1 * UNROLL_N, X_TMP_1);
        st_d(X_TMP_1, I.getXReg(), I.getOffset());
        //cmp(I, UNROLL_N);
        mov_imm(X_TMP_0, UNROLL_N);
        //jge(labels[0], T_NEAR);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bge(X_TMP_1, X_TMP_0, labels[0]);
        //align(16);

        L(labels[1]);
        //cmp(I, 1);
        mov_imm(X_TMP_0, 1);
        //jne(labels[2], T_NEAR);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[2]);
        if (unroll_m == 16) {
            kernel_16x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        //jmp(labels[14], T_NEAR);
        b(labels[14]);
        //align(16);

        L(labels[2]);
        //cmp(I, 2);
        mov_imm(X_TMP_0, 2);
        //jne(labels[3], T_NEAR);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[3]);
        if (unroll_m == 16) {
            kernel_16x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        //jmp(labels[14], T_NEAR);
        b(labels[14]);
        //align(16);

        L(labels[3]);
        //cmp(I, 3);
        mov_imm(X_TMP_0, 3);
        //jne(labels[4], T_NEAR);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[4]);
        if (unroll_m == 16) {
            kernel_16x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        //jmp(labels[14], T_NEAR);
        b(labels[14]);
        //align(16);

        L(labels[4]);
        //cmp(I, 4);
        mov_imm(X_TMP_0, 4);
        //jne(labels[5], T_NEAR);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[5]);
        if (unroll_m == 16) {
            kernel_16x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        //jmp(labels[14], T_NEAR);
        b(labels[14]);
        //align(16);

        L(labels[5]);
        //cmp(I, 5);
        mov_imm(X_TMP_0, 5);
        //jne(labels[14], T_NEAR);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[14]);
        if (unroll_m == 16) {
            kernel_16x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        //jmp(labels[14], T_NEAR);
        b(labels[14]);
        //align(16);

        if (!isTransA) {
            L(labels[7]);
            //cmp(I, UNROLL_N);
            mov_imm(X_TMP_0, UNROLL_N);
            //jl(labels[6], T_NEAR);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            blt(X_TMP_1, X_TMP_0, labels[6]);
            //align(16);

            L(labels[8]);
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, true, false);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        true, false);
            }
            //sub(I, UNROLL_N);
            ld_d(X_TMP_0, I.getXReg(), I.getOffset());
            add_imm(X_TMP_1, X_TMP_0, -1 * UNROLL_N, X_TMP_1);
            st_d(X_TMP_1, I.getXReg(), I.getOffset());
            //cmp(I, UNROLL_N);
            mov_imm(X_TMP_0, UNROLL_N);
            //jge(labels[8], T_NEAR);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bge(X_TMP_1, X_TMP_0, labels[8]);
            //align(16);

            L(labels[6]);
            //cmp(I, 1);
            mov_imm(X_TMP_0, 1);
            //jne(labels[9], T_NEAR);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[9]);
            if (unroll_m == 16) {
                kernel_16x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            //jmp(labels[14], T_NEAR);
            b(labels[14]);
            //align(16);

            L(labels[9]);
            //cmp(I, 2);
            mov_imm(X_TMP_0, 2);
            //jne(labels[10], T_NEAR);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[10]);
            if (unroll_m == 16) {
                kernel_16x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            //jmp(labels[14], T_NEAR);
            b(labels[14]);
            //align(16);

            L(labels[10]);
            //cmp(I, 3);
            mov_imm(X_TMP_0, 3);
            //jne(labels[11], T_NEAR);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[11]);
            if (unroll_m == 16) {
                kernel_16x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            //jmp(labels[14], T_NEAR);
            b(labels[14]);
            //align(16);

            L(labels[11]);
            //cmp(I, 4);
            mov_imm(X_TMP_0, 4);
            //jne(labels[12], T_NEAR);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[12]);
            if (unroll_m == 16) {
                kernel_16x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            //jmp(labels[14], T_NEAR);
            b(labels[14]);
            //align(16);

            L(labels[12]);
            //cmp(I, 5);
            mov_imm(X_TMP_0, 5);
            //jne(labels[14], T_NEAR);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[14]);
            if (unroll_m == 16) {
                kernel_16x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            //align(16);
        }

        L(labels[14]);
        // Compute address for A
        if (!isTransA) {
            //add(A, unroll_m * SIZE);
            add_imm(A, A, unroll_m * SIZE, X_TMP_0);
        } else {
            //mov(rax, LDA);
            add_d(t3, LDA, zero);
            //imul(rax, rax, unroll_m);
            mov_imm(X_TMP_0, unroll_m);
            mul_d(t3, t3, X_TMP_0);
            //add(A, rax);
            add_d(A, A, t3);
        }

        // Compute next address of BIAS
        //if (hasBias) { add(BIAS, unroll_m * SIZE); }
        if (hasBias) {
            ld_d(X_TMP_0, BIAS.getXReg(), BIAS.getOffset());
            add_imm(X_TMP_0, X_TMP_0, unroll_m * SIZE, X_TMP_1);
            st_d(X_TMP_0, BIAS.getXReg(), BIAS.getOffset());
        }
    }

    void generate() override ATTRIBUTE_OPTIMIZE {
        //assert(IMPLICATION(!is_avx2, mayiuse(avx)));

        preamble();

        Label buffer_in_ws, buffer_allocated;

        // Get the registers
        //mov(B, ARG_B); // x86 has 6 abi_params and loongarch has 8 so do not need load
        //mov(LDB, ARG_LDB);// x86 has 6 abi_params and loongarch has 8 so do not need load

        //mov(r15, ARG_BETA);
        ld_d(s3, ARG_BETA.getXReg(), ARG_BETA.getOffset());
        //mov(r12, ARG_C);
        ld_d(s4, ARG_C.getXReg(), ARG_C.getOffset());
        //if (hasBias) mov(r10, ARG_BIAS);
        if (hasBias) ld_d(s5, ARG_BIAS.getXReg(), ARG_BIAS.getOffset());
        //mov(LDC, ARG_LDC);
        ld_d(LDC, ARG_LDC.getXReg(), ARG_LDC.getOffset());
        //mov(rbp, rsp); // in loongarch sp in already saved to s6

        //vmovss(vr0, ptr[ARG_ALPHA]);
        vldrepl_w(vr0, ARG_ALPHA, 0);
        //vmovss(vr1, ptr[r15]);
        vldrepl_w(vr1, s3, 0);

//#ifdef _WIN32
//        mov(A, ARG_A);
//        mov(LDA, ARG_LDA);
//#endif

        //cmp(K, STACK_K_CAPACITY);
        mov_imm(X_TMP_0, STACK_K_CAPACITY);
        //jg(buffer_in_ws, T_NEAR);
        blt(X_TMP_0, K, buffer_in_ws);

        // Create buffer and align to 4kB page
        //lea(rax, ptr[K * SIZE]);
        slli_d(X_TMP_0, K, 2);
        //sal(rax, 4);
        slli_d(X_TMP_0, X_TMP_0, 2);
        //add(rax, 256);
        addi_d(X_TMP_0, X_TMP_0, 256);
        //sub(rsp, rax);
        sub_d(sp, sp, X_TMP_0);
        //and_(rsp, -PAGE_4K);
        mov_imm(X_TMP_1, -PAGE_4K);
        and_(sp, sp, X_TMP_1);
        //jmp(buffer_allocated, T_NEAR);
        b(buffer_allocated);

        L(buffer_in_ws);
        //mov(rsp, ARG_WS);
        ld_d(sp, ARG_WS.getXReg(), ARG_WS.getOffset());

        L(buffer_allocated);

        //mov(ORIG_SP, rbp);
        st_d(s6, ORIG_SP.getXReg(), ORIG_SP.getOffset());
        //mov(M, ARG_M);
        st_d(ARG_M, M.getXReg(), M.getOffset());
        //mov(N, ARG_N);
        st_d(ARG_N, N.getXReg(), N.getOffset());
        //mov(C, r12);
        st_d(s4, C.getXReg(), C.getOffset());
        //if (hasBias) mov(BIAS, r10);
        if (hasBias) st_d(s5, BIAS.getXReg(), BIAS.getOffset());
        //vmovss(ALPHA, vr0);
        vstelm_w(vr0, ALPHA.getXReg(), ALPHA.getOffset(), 0);
        //vmovss(BETA, vr1);
        vstelm_w(vr1, BETA.getXReg(), BETA.getOffset(), 0);
        //sub(A, -OFFSET * SIZE);
        addi_d(A, A, OFFSET * SIZE);
        //sub(B, -OFFSET * SIZE);
        addi_d(B, B, OFFSET * SIZE);
        //mov(ORIG_A, A);
        st_d(A, ORIG_A.getXReg(), ORIG_A.getOffset());
        //sal(LDA, BASE_SHIFT);
        slli_d(LDA, LDA, BASE_SHIFT);
        //sal(LDB, BASE_SHIFT);
        slli_d(LDB, LDB, BASE_SHIFT);
        //sal(LDC, BASE_SHIFT);
        slli_d(LDC, LDC, BASE_SHIFT);
        //lea(LDB3, ptr[LDB + LDB * 2]);
        add_d(LDB3, LDB, LDB);
        add_d(LDB3, LDB3, LDB);

        for (int i = 0; i < 8; i++) {
            //mov(dword[rsp + 88 + i * 4], i);
            addi_d(X_TMP_0, zero, i);
            st_d(X_TMP_0, sp, 88 + i * 4);
        }

        if (isTransA && is_avx2) {
            //movq(vr0, LDA);
            vxor_v(vr0, vr0, vr0);
            vinsgr2vr_d(vr0, LDA, 0);
            //vpbroadcastq(xr1, vr0);
            xvldrepl_d(xr1, LDA, 0);
            //vinsertf128(xr0, xr0, vr0, 1);
            xvpermi_q(xr0, xr0, 0x02);
            //vpermilpd(xr0, xr0, 5);
            xvpermi_d(xr0, xr0, 0b10110001);
            //vpaddq(xr1, xr1, xr1);
            xvadd_d(xr1, xr1, xr1);
            //vperm2f128(xr1, xr1, xr1, 8);
            xvpermi_q(xr1, xr1, 8);
            vxor_v(vr1, vr1, vr1); // vperm2f128 make the low 128bit to 0
            //vpaddq(xr0, xr0, xr1);
            xvadd_d(xr0, xr0, xr1);
            //vmovups(STRIDE, xr0);
            xvst(xr0, STRIDE.getXReg(), STRIDE.getOffset());
        }

        // Check A alignment and leading dimension; take copy-based path as
        // needed
        //mov(rax, LDA);
        add_d(X_TMP_0, LDA, zero);
        //or_(rax, A);
        or_(X_TMP_0, X_TMP_0, A);
        //and_(rax, 0x1f);
        andi(X_TMP_0, X_TMP_0, 0x1f);
        //mov(FLAG, rax);
        st_d(X_TMP_0, FLAG.getXReg(), FLAG.getOffset());
        std::vector<Label> labels(5);

        //cmp(M, UNROLL_M);
        mov_imm(X_TMP_1, UNROLL_M);
        //jl(labels[0], T_NEAR);
        ld_d(X_TMP_0, M.getXReg(), M.getOffset());        
        blt(X_TMP_0, X_TMP_1, labels[0]);
        //align(16);

        L(labels[1]);
        subloop(UNROLL_M, true, true);
        //sub(M, UNROLL_M);
        ld_d(X_TMP_0, M.getXReg(), M.getOffset());
        addi_d(X_TMP_0, X_TMP_0, -1 * UNROLL_M);
        st_d(X_TMP_0, M.getXReg(), M.getOffset());
        //cmp(M, UNROLL_M);
        mov_imm(X_TMP_1, UNROLL_M);
        //jge(labels[1], T_NEAR);
        ld_d(X_TMP_0, M.getXReg(), M.getOffset());
        bge(X_TMP_0, X_TMP_1, labels[1]);
        //align(16);

        L(labels[0]);
        //cmp(M, 0);
        //jle(labels[4], T_NEAR);
        ld_d(X_TMP_0, M.getXReg(), M.getOffset());
        bge(zero, X_TMP_0, labels[4]);

        if (UNROLL_M > 8) {
            //cmp(M, 8);
            mov_imm(X_TMP_1, 8);
            //jle(labels[2], T_NEAR);
            ld_d(X_TMP_0, M.getXReg(), M.getOffset());
            bge(X_TMP_1, X_TMP_0, labels[2]);

            //sub(M, 8);
            addi_d(X_TMP_0, X_TMP_0, -8);
            st_d(X_TMP_0, M.getXReg(), M.getOffset());
            //vbroadcastss(VMASK, M);
            xvldrepl_w(VMASK, M.getXReg(), M.getOffset());
            //vpcmpgtd(VMASK, VMASK, MASK);
            xvld(xr31, MASK.getXReg(), MASK.getOffset());
            xvslt_w(VMASK, xr31, VMASK);

            subloop(16, true, false);
            //jmp(labels[4], T_NEAR);
            b(labels[4]);
            //align(16);

            L(labels[2]);
            //cmp(M, 8);
            mov_imm(X_TMP_1, 8);
            //jne(labels[3], T_NEAR);
            ld_d(X_TMP_0, M.getXReg(), M.getOffset());
            bne(X_TMP_0, X_TMP_1, labels[3]);
            subloop(8, true, true);
            //jmp(labels[4], T_NEAR);
            b(labels[4]);
        }

        //align(16);

        L(labels[3]);
        //vbroadcastss(VMASK, M);
        xvldrepl_w(VMASK, M.getXReg(), M.getOffset());
        if (is_avx2) {
            //vpcmpgtd(VMASK, VMASK, MASK);
            xvld(xr31, MASK.getXReg(), MASK.getOffset());
            xvslt_w(VMASK, xr31, VMASK);
        } //else {
        //    auto xmask = Xmm(VMASK.getIdx());
        //    auto xmm_tmp = vr4;

        //    vextractf128(xmm_tmp, VMASK, 1);
        //    vpcmpgtd(xmask, xmask, MASK);
        //    vpcmpgtd(xmm_tmp, xmm_tmp, dword[rsp + 88 + 4 * 4]); // MASK + 4
        //    vinsertf128(VMASK, VMASK, xmm_tmp, 1);
        //}
        subloop(8, false, false);
        //align(16);

        L(labels[4]);
        // Restore original stack
        //mov(rsp, ORIG_SP);
        ld_d(sp, ORIG_SP.getXReg(), ORIG_SP.getOffset());

        //vzeroupper();
        postamble();
    }

private:
    const char isTransA;
    const char isTransB;
    const bool hasBias;
    const bool is_avx2;
    const int UNROLL_M;
    const int UNROLL_N;
    const bool isBeta0;
    const bool isBetaN;
    const int PREFETCHSIZEA;
    const int PREFETCHSIZEB;

    /* x86 registers define
    // Register allocation (for convenience)
    const Reg64 ARG_M = abi_param1;
    const Reg64 ARG_N = abi_param2;
    const Reg64 K = abi_param3;
    const Reg64 ARG_ALPHA = abi_param4;
#ifdef _WIN32
    const Address ARG_A = ptr[rsp + OFFSET_SHADOWSPACE + STACKSIZE];
    const Address ARG_LDA
            = qword[rsp + OFFSET_SHADOWSPACE + sizeof(float *) + STACKSIZE];
    const int stackOffset = OFFSET_SHADOWSPACE + sizeof(float *) + STACKSIZE;
    const Reg64 A = rsi;
    const Reg64 LDA = rdi;
#else
    const Reg64 ARG_A = r8;
    const Reg64 ARG_LDA = r9;
    const int stackOffset = STACKSIZE;
    const Reg64 A = ARG_A;
    const Reg64 LDA = ARG_LDA;
#endif
    const Address ARG_B = ptr[rsp + 8 + stackOffset];
    const Address ARG_LDB = ptr[rsp + 16 + stackOffset];
    const Address ARG_BETA = ptr[rsp + 24 + stackOffset];
    const Address ARG_C = ptr[rsp + 32 + stackOffset];
    const Address ARG_LDC = ptr[rsp + 40 + stackOffset];
    const Address ARG_BIAS = ptr[rsp + 48 + stackOffset];
    const Address ARG_WS = ptr[rsp + 56 + stackOffset];

    const Reg64 B = r11;
    const Reg64 LDB = rbx;
    const Reg64 LDC = r13;
    const Reg64 LL = rax;
    const Reg64 AO1 = abi_param2;
    const Reg64 BO1 = abi_param4;
    const Reg64 BO2 = rbp;
    const Reg64 CO1 = r14;
    const Reg64 CO2 = r15;
    const Reg64 LDB3 = r10;
    const Reg64 LDA4 = abi_param1;
    const Reg64 AA = r12;
    const Reg64 BIAS1 = abi_param1;

    const Address M = qword[rsp + 0];
    const Address N = qword[rsp + 8];
    const Address FLAG = qword[rsp + 16];
    const Address I = qword[rsp + 24];
    const Address C = qword[rsp + 32];
    const Address BIAS = qword[rsp + 40];
    const Address ALPHA = qword[rsp + 48];
    const Address BETA = qword[rsp + 64];
    const Address ORIG_A = qword[rsp + 80];
    const Address MASK = dword[rsp + 88];
    const Address STRIDE = qword[rsp + 120];
    const Address ORIG_SP = qword[rsp + 152];

    const XVReg VALPHA = xr1;
    const XVReg VBETA = xr2;
    const XVReg VMASK = xr3;
    const XVReg VBIAS1 = xr2;
    const XVReg VBIAS2 = xr4; */

    const XReg ARG_M = abi_param1;
    const XReg ARG_N = abi_param2;
    const XReg K = abi_param3;
    const XReg ARG_ALPHA = abi_param4;

    const XReg ARG_A = abi_param5;
    const XReg ARG_LDA = abi_param6;
    const int stackOffset = STACKSIZE;
    const XReg A = ARG_A;
    const XReg LDA = ARG_LDA;

    const XReg ARG_B = abi_param7; // loongarch has 8 abi_params so ARG_B is abi_param7
    const XReg ARG_LDB = abi_param8; // loongarch has 8 abi_params so ARG_LDB is abi_param8
    const Address ARG_BETA = ptr_a(sp, 8 + stackOffset); // from ARG_BETA the param in sp
    const Address ARG_C = ptr_a(sp, 16 + stackOffset);
    const Address ARG_LDC = ptr_a(sp, 24 + stackOffset);
    const Address ARG_BIAS = ptr_a(sp, 32 + stackOffset);
    const Address ARG_WS = ptr_a(sp, 40 + stackOffset);

    const XReg B = ARG_B;
    const XReg LDB = ARG_LDB;
    const XReg LDC = t2;
    const XReg LL = t3;
    const XReg AO1 = abi_param2;
    const XReg BO1 = abi_param4;
    const XReg BO2 = t4;
    const XReg CO1 = t5;
    const XReg CO2 = t6;
    const XReg LDB3 = t7;
    const XReg LDA4 = abi_param1;
    const XReg AA = t8;
    const XReg BIAS1 = abi_param1;

    const Address M = ptr_a(sp, 0);
    const Address N = ptr_a(sp, 8);
    const Address FLAG = ptr_a(sp, 16);
    const Address I = ptr_a(sp, 24);
    const Address C = ptr_a(sp, 32);
    const Address BIAS = ptr_a(sp, 40);
    const Address ALPHA = ptr_a(sp, 48);
    const Address BETA = ptr_a(sp, 64);
    const Address ORIG_A = ptr_a(sp, 80);
    const Address MASK = ptr_a(sp, 88);
    const Address STRIDE = ptr_a(sp, 120);
    const Address ORIG_SP = ptr_a(sp, 152);

    const XVReg VALPHA = xr1;
    const XVReg VBETA = xr2;
    const XVReg VMASK = xr3;
    const XVReg VBIAS1 = xr2;
    const XVReg VBIAS2 = xr4;
};

xbyak_gemm_t *get_xbyak_gemm(
        bool isTransA, bool isTransB, float beta, bool hasBias) {
    auto beta_idx = [](float beta) {
        return (beta == 0.0) ? 0 : (beta == 1.0 ? 1 : 2);
    };

    // Kernel table [isTransA][isTransB][hasBias][beta (0, 1, other)]
    static xbyak_gemm_t *kernel_table[2][2][2][3];
    static std::once_flag initialized;
    dnnl_status_t st = dnnl_success;
    std::call_once(initialized, [&] {
        for (bool isTransA : {false, true})
            for (bool isTransB : {false, true})
                for (bool hasBias : {false, true})
                    for (float beta : {0.0f, 1.0f, 2.0f}) {
                        // nocopy sgemm with bias for beta != 0.0 is not supported
                        if (hasBias && beta != 0.0) continue;
                        auto &kern = kernel_table[isTransA][isTransB][hasBias]
                                                 [beta_idx(beta)];

                        kern = new xbyak_gemm_t(
                                isTransA, isTransB, beta, hasBias);
                        if (kern->create_kernel() != dnnl_success) {
                            st = dnnl_runtime_error;
                            return;
                        }
                    }
    });

    return (st == dnnl_success)
            ? kernel_table[isTransA][isTransB][hasBias][beta_idx(beta)]
            : nullptr;
}

dnnl_status_t sgemm_nocopy_driver(const char *transa, const char *transb,
        dim_t m, dim_t n, dim_t k, const float *alpha, const float *a,
        dim_t lda, const float *b, dim_t ldb, const float *beta, float *c,
        dim_t ldc, const float *bias, float *ws) {

    bool isTransA = (*transa == 'T' || *transa == 't');
    bool isTransB = (*transb == 'T' || *transb == 't');

    dim_t Bm, sizeM, Bn, sizeN, Bk, sizeK;

    dim_t i, j;

    if ((m <= 0) || (n <= 0)) return dnnl_success;

    if ((k <= 0) || (alpha[0] == 0.)) {

        if (beta[0] == 0.) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[i + j * ldc] = 0.0;
        } else if (beta[0] != 1.) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[i + j * ldc] *= beta[0];
        }

        return dnnl_success;
    }

    assert(IMPLICATION(bias != nullptr, *beta == 0.0));

    // XXX: this happens on every thread...
    bool hasBias = (bias != nullptr);
    auto ker_bn = get_xbyak_gemm(isTransA, isTransB, *beta, hasBias);
    auto ker_b1 = get_xbyak_gemm(isTransA, isTransB, 1.0, false);
    auto ker_b0 = get_xbyak_gemm(isTransA, isTransB, 0.0, false);
    if (utils::any_null(ker_bn, ker_b1, ker_b0)) return dnnl_runtime_error;

    dim_t BM = 4032;
    dim_t BN = isTransA ? 96 : 48;
    dim_t BK = isTransB ? 96 : 256;
    const float *curA, *curB, *curBias = nullptr;
    float *curC;

    for (Bk = 0; Bk < k; Bk += sizeK) {
        sizeK = k - Bk;
        if (sizeK >= BK * 2)
            sizeK = BK;
        else {
            if (sizeK > BK) sizeK = (sizeK + 1) / 2;
        }

        for (Bm = 0; Bm < m; Bm += sizeM) {
            sizeM = m - Bm;
            if (sizeM >= BM * 2)
                sizeM = BM;
            else {
                if (sizeM > BM + BM / 2) sizeM = (sizeM + 1) / 2;
            }

            for (Bn = 0; Bn < n; Bn += sizeN) {
                sizeN = n - Bn;
                if (sizeN >= BN * 2)
                    sizeN = BN;
                else {
                    if (sizeN > BN + BN / 2) sizeN = (sizeN + 1) / 2;
                }

                if (!isTransA) {
                    curA = a + Bm + Bk * lda;
                } else {
                    curA = a + Bk + Bm * lda;
                }
                if (!isTransB) {
                    curB = b + Bk + Bn * ldb;
                } else {
                    curB = b + Bn + Bk * ldb;
                }
                curC = c + Bm + (size_t)Bn * ldc;
                if (bias != nullptr) {
                    if (Bk == 0) {
                        curBias = bias + Bm;
                    } else {
                        curBias = nullptr;
                    }
                }
                if (Bk == 0) {
                    if (*beta == 0.0 && bias == nullptr)
                        (*ker_b0)(sizeM, sizeN, sizeK, alpha, curA, lda, curB,
                                ldb, beta, curC, ldc, curBias, ws);
                    else
                        (*ker_bn)(sizeM, sizeN, sizeK, alpha, curA, lda, curB,
                                ldb, beta, curC, ldc, curBias, ws);
                } else {
                    (*ker_b1)(sizeM, sizeN, sizeK, alpha, curA, lda, curB, ldb,
                            beta, curC, ldc, curBias, ws);
                }
            }
        }
    }
    msan_unpoison_matrix(c, m, n, ldc, sizeof(*c));

    return dnnl_success;
}

} // namespace lasx_gemm_f32

dnnl_status_t jit_lasx_gemm_f32(int nthrs, const char *transa,
        const char *transb, const dim_t *p_m, const dim_t *p_n,
        const dim_t *p_k, const float *p_alpha, const float *A,
        const dim_t *p_lda, const float *B, const dim_t *p_ldb,
        const float *p_beta, float *C, const dim_t *p_ldc, const float *bias) {

    using namespace dnnl::impl::utils;
    using namespace lasx_gemm_f32;
    using namespace gemm_utils;

    if (*p_beta != 0 && bias)
        return ref_gemm(transa, transb, p_m, p_n, p_k, p_alpha, A, p_lda, B,
                p_lda, p_beta, C, p_ldc, bias);

    int nthr_max = dnnl_get_current_num_threads();
    int nthr_to_use = nstl::min(nthrs, nthr_max);

    dim_t m = *p_m;
    dim_t n = *p_n;
    dim_t k = *p_k;
    dim_t lda = *p_lda;
    dim_t ldb = *p_ldb;
    dim_t ldc = *p_ldc;
    float beta = *p_beta;
    dim_t MB, NB, KB;

    int nthr_m = 1, nthr_n = 1, nthr_k = 1, nthr_mn = 1;

    // Determine threading partitioning
    calc_nthr_nocopy_avx(
            m, n, k, nthr_to_use, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(IMPLICATION(!dnnl_thr_syncable(), nthr_k == 1));

    nthr_to_use = nthr_m * nthr_n * nthr_k;

    nthr_mn = nthr_m * nthr_n;

    unsigned char *ompstatus_ = nullptr;
    unsigned char volatile *ompstatus = nullptr;

    float *c_buffers = nullptr;
    float *ws_buffers = nullptr;

    if (nthr_k > 1) {
        ompstatus_ = (unsigned char *)malloc(
                nthr_to_use * CACHE_LINE_SIZE, CACHE_LINE_SIZE);
        if (!ompstatus_) return dnnl_out_of_memory;

        ompstatus = (unsigned char volatile *)ompstatus_;
        assert(ompstatus);

        for (int i = 0; i < nthr_to_use; i++)
            ompstatus[i * CACHE_LINE_SIZE] = 0;

        c_buffers = (float *)malloc(
                sizeof(*c_buffers) * nthr_m * nthr_n * (nthr_k - 1) * MB * NB,
                PAGE_4K);
        if (!c_buffers) {
            free(ompstatus_);
            return dnnl_out_of_memory;
        }
    }

    const size_t ws_elems_per_thr
            = (size_t)rnd_up(div_up(k, nthr_k), KB) * 16 + 64;
    const size_t ws_size_per_thr
            = rnd_up(ws_elems_per_thr * sizeof(float), PAGE_4K);
    if (k > STACK_K_CAPACITY) {
        ws_buffers = (float *)malloc(nthr_to_use * ws_size_per_thr, PAGE_4K);
        if (!ws_buffers) {
            free(ompstatus_);
            free(c_buffers);
            return dnnl_out_of_memory;
        }
    }

    if (nthr_to_use == 1) {
        auto status = sgemm_nocopy_driver(transa, transb, m, n, k, p_alpha, A,
                lda, B, ldb, p_beta, C, ldc, bias, ws_buffers);
        if (ws_buffers) free(ws_buffers);
        return status;
    }

    // Always use the maximum number of threads to avoid OMP overhead that can
    // occur due to change thread counts.
    int nthr_spawn = dnnl_thr_syncable() ? nthr_max : nthr_to_use;

    std::atomic<dnnl_status_t> st(dnnl_success);
    parallel(nthr_spawn, [&](int ithr, int nthr) {
        assert(nthr_spawn == nthr);
        MAYBE_UNUSED(nthr);

        int ithr_m, ithr_n, ithr_k, ithr_mn;
        dim_t m_from, m_to, myM;
        dim_t n_from, n_to, myN;
        dim_t k_from, k_to, myK;
        int cbase, ibase;
        const float *myA, *myB, *myBias = nullptr;
        float *myC = C, myBeta;
        float *ws = ws_buffers
                ? ws_buffers + ithr * ws_size_per_thr / sizeof(float)
                : nullptr;
        dim_t ld = ldc;

        int sum_later = (nthr < nthr_m * nthr_n * nthr_k);

        if (ithr < nthr_m * nthr_n * nthr_k) {

            ithr_mn = ithr % nthr_mn;
            ithr_m = ithr_mn % nthr_m;
            ithr_n = ithr_mn / nthr_m;
            ithr_k = ithr / nthr_mn;

            /* swap ithr_k for performance improvement */
            if (ithr_k == 0)
                ithr_k = nthr_k - 1;
            else if (ithr_k == nthr_k - 1)
                ithr_k = 0;

            m_from = MB * (ithr_m);
            m_to = MB * (ithr_m + 1);
            if (m_to > m) m_to = m;
            myM = m_to - m_from;

            n_from = NB * (ithr_n);
            n_to = NB * (ithr_n + 1);
            if (n_to > n) n_to = n;
            myN = n_to - n_from;

            k_from = KB * (ithr_k);
            k_to = KB * (ithr_k + 1);
            if (k_to > k) k_to = k;
            myK = k_to - k_from;

            cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);
            ibase = (ithr_m + nthr_m * ithr_n) * nthr_k;

            if ((myM > 0) && (myN > 0)) {

                if (*transa == 'N' || *transa == 'n') {
                    myA = &(A[m_from + k_from * lda]);
                } else {
                    myA = &(A[k_from + m_from * lda]);
                }
                if (*transb == 'N' || *transb == 'n') {
                    myB = &(B[k_from + n_from * ldb]);
                } else {
                    myB = &(B[n_from + k_from * ldb]);
                }
                if (ithr_k == 0) {
                    myC = &(C[m_from + n_from * ldc]);
                    myBeta = beta;
                    ld = ldc;
                    if (bias) myBias = &(bias[m_from]);
                } else {
                    myC = c_buffers + MB * NB * (cbase + ithr_k - 1);
                    myBeta = 0.0;
                    ld = MB;
                    myBias = nullptr;
                }

                dnnl_status_t st_thr = sgemm_nocopy_driver(transa, transb, myM,
                        myN, myK, p_alpha, myA, lda, myB, ldb, &myBeta, myC, ld,
                        myBias, ws);
                if (st_thr != dnnl_success) {
                    st = st_thr;
                    return;
                }

                if (nthr_k > 1 && !sum_later)
                    ompstatus[(ibase + ithr_k) * CACHE_LINE_SIZE] = 1;
            }

            if (nthr_k > 1 && !sum_later) {

                // sum matrices partitioned along K dimension
                dim_t n1, n2;

                partition_unit_diff(ithr_k, nthr_k, myN, &n1, &n2);

                if (ithr_k > 0) {

                    myC = c_buffers + MB * NB * (cbase + ithr_k - 1) + n1 * MB;
                    /* need to wait until main thread finishes */
                    while (ompstatus[ibase * CACHE_LINE_SIZE] != 1) {};

                    /* my cache is hot */
                    sum_two_matrices(myM, n2, myC, MB,
                            &C[m_from + (n_from + n1) * ldc], ldc);
                }

                for (int ik = 1; ik < nthr_k; ++ik) {
                    if (ik != ithr_k) {

                        myC = c_buffers + MB * NB * (cbase + ik - 1) + n1 * MB;

                        while (ompstatus[(ibase + ik) * CACHE_LINE_SIZE] != 1) {
                        };

                        sum_two_matrices(myM, n2, myC, MB,
                                &C[m_from + (n_from + n1) * ldc], ldc);
                    }
                }
            }
        }
    });
    CHECK(st);

    // handle C summation later
    if (nthr_k > 1 && ompstatus[0] == 0) {

        parallel(nthr_spawn, [&](int ithr, int nthr) {
            assert(nthr_spawn == nthr);
            MAYBE_UNUSED(nthr);

            int ithr_m, ithr_n, ithr_k, ithr_mn;
            dim_t m_from, m_to, myM;
            dim_t n_from, n_to, myN;
            int cbase;
            float *myC = C;

            if (ithr < nthr_m * nthr_n * nthr_k) {

                ithr_mn = ithr % nthr_mn;
                ithr_m = ithr_mn % nthr_m;
                ithr_n = ithr_mn / nthr_m;
                ithr_k = ithr / nthr_mn;

                /* swap ithr_k for performance improvement */
                if (ithr_k == 0)
                    ithr_k = nthr_k - 1;
                else if (ithr_k == nthr_k - 1)
                    ithr_k = 0;

                m_from = MB * (ithr_m);
                m_to = MB * (ithr_m + 1);
                if (m_to > m) m_to = m;
                myM = m_to - m_from;

                n_from = NB * (ithr_n);
                n_to = NB * (ithr_n + 1);
                if (n_to > n) n_to = n;
                myN = n_to - n_from;

                cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

                if (nthr_k > 1) {
                    // sum matrices partitioned along K dimension
                    dim_t n1, n2;

                    partition_unit_diff(ithr_k, nthr_k, myN, &n1, &n2);

                    if (ithr_k > 0) {

                        myC = c_buffers + MB * NB * (cbase + ithr_k - 1)
                                + n1 * MB;

                        /* my cache is hot */
                        sum_two_matrices(myM, n2, myC, MB,
                                &C[m_from + (n_from + n1) * ldc], ldc);
                    }

                    for (int ik = 1; ik < nthr_k; ++ik) {
                        if (ik != ithr_k) {

                            myC = c_buffers + MB * NB * (cbase + ik - 1)
                                    + n1 * MB;

                            sum_two_matrices(myM, n2, myC, MB,
                                    &C[m_from + (n_from + n1) * ldc], ldc);
                        }
                    }
                }
            }
        });
    }

    free(c_buffers);
    free(ompstatus_);
    free(ws_buffers);

    return dnnl_success;
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
