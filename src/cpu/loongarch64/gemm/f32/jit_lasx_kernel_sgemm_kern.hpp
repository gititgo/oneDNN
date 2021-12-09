/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CPU_LOONGARCH64_GEMM_F32_JIT_LASX_KERNEL_SGEMM_KERN_HPP
#define CPU_LOONGARCH64_GEMM_F32_JIT_LASX_KERNEL_SGEMM_KERN_HPP

#include "cpu/loongarch64/jit_generator.hpp"

#define MAX_UNROLL_M 48
#define MAX_UNROLL_N_BIN 3

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

class jit_lasx_kernel_sgemm_kern : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lasx_kernel_sgemm_kern);
    const int elt_size_ = 4;
    const int elt_size_bin_ = 2;
    //int nelt_per_vecreg_ = mayiuse(avx512_core) ? 16 : 8;
    int nelt_per_vecreg_ = 8;
    const int unroll_m_reg_ = 3;
    int unroll_m_ = unroll_m_reg_ * nelt_per_vecreg_;
    //const int unroll_n_ = mayiuse(avx512_core) ? 8 : 4;
    const int unroll_n_ = 4;
    const int unroll_k_ = 4;
    const int unroll_k_bin_ = 2;
    //const int unroll_m_bin_ = mayiuse(avx512_core) ? 6 : 5;
    const int unroll_m_bin_ = 5;
    //const int second_fetch_ = mayiuse(avx512_core) ? 32 : 34;
    const int second_fetch_ = 34;
    //unsigned int unroll_n_bin_ = mayiuse(avx512_core) ? 3 : 2;
    unsigned int unroll_n_bin_ = 2;
    bool beta_zero_;

    /* x86 registers define
    Xbyak::Reg64 M_ = rdi, N_ = rsi, K_ = rdx, A_ = r8, B_ = r9, C_ = r10,
                 LDC_ = r11;
    Xbyak::Reg64 I_ = r12, J_ = r13, AA_ = rcx, KK_ = K_, BO_ = rbp, CO1_ = r14,
                 CO2_ = r15;
    Xbyak::Reg64 AO_ = rbx, LL_ = rax; */
    Xbyak_loongarch::XReg M_ = a3, N_ = a4, K_ = a5, A_ = a6, B_ = a7, C_ = t3,
                 LDC_ = t4;
    Xbyak_loongarch::XReg I_ = t5, J_ = t6, AA_ = t7, KK_ = K_, BO_ = t8, CO1_ = t2,
                 CO2_ = t1;
    Xbyak_loongarch::XReg AO_ = a1, LL_ = a2;
    //int zmm_a_idx_ = 0, zmm_b_idx_ = mayiuse(avx512_core) ? 6 : 3,
    //    zmm_acc_idx_ = mayiuse(avx512_core) ? 8 : 4;
    int zmm_a_idx_ = 0, zmm_b_idx_ = 3, zmm_acc_idx_ = 4;
    //int nb_zmm_a_ = mayiuse(avx512_core) ? unroll_m_reg_ * 2 : unroll_m_reg_,
    //    nb_zmm_b_ = mayiuse(avx512_core) ? 2 : 1;
    int nb_zmm_a_ = unroll_m_reg_, nb_zmm_b_ = 1;

    //int addr_off_ = mayiuse(avx512_core) ? 128 : 32;
    int addr_off_ = 32;
    //int PREFETCHSIZEB_ = mayiuse(avx512_core) ? (-128 + 16 * 8) : 64;
    int PREFETCHSIZEB_ = 64;
    //int PREFETCHSIZEA_ = mayiuse(avx512_core) ? (-128 + 16 * 2)
    //                                          : (PREFETCHSIZEB_ * 2 + 16);
    int PREFETCHSIZEA_ = (PREFETCHSIZEB_ * 2 + 16);
    int off_ = 0, offb_ = 0;

    int next_acc(int idx, int um, int un) const;
    void prefetchB_beforeBload(int um, int un, int k_idx, int n_idx);
    void prefetchB_beforeFMA(int um, int un, int k_idx, int n_idx, int m_idx);
    void prefetchA_afterFMA(int um, int un, int k_idx, int n_idx, int m_idx);
    void prefetchA_afterBload(int um, int un, int k_idx, int n_idx);
    void prefetchB_afterFMA(int k_idx, int n_idx, int m_idx);
    void prefetchA_beforeFMA(int um, int un, int k_idx, int n_idx, int m_idx);
    void prefetchC_afterBload(int um, int un, int k_idx, int n_idx);
    void prefetchC_beforeKloop(int um);
    void generate() override ATTRIBUTE_OPTIMIZE;

    template <typename T_reg, typename T_desta, typename T_srca>
    void loadA_betweenFMAs(int um, int un, int k_idx, int n_idx, int m_idx,
            void (jit_generator::*aload)(
                    const T_desta &, const T_srca &, int32_t)) {
        //int next_zmm_a = mayiuse(avx512_core)
        //        ? unroll_m_reg_
        //        : std::max(1, um / nelt_per_vecreg_);
        int next_zmm_a = std::max(1, um / nelt_per_vecreg_);
        //if (!(mayiuse(avx512_core) || (um <= 8) || ((um == 16) && (un == 4)))) {
        if (!((um <= 8) || ((um == 16) && (un == 4)))) {
            if (n_idx == un - 1) {
                //(this->*aload)(T_reg(zmm_a_idx_ + m_idx
                //                       + (k_idx % (nb_zmm_a_ / unroll_m_reg_))
                //                               * next_zmm_a),
                //        ptr[AO_
                //                + elt_size_
                //                        * (m_idx * nelt_per_vecreg_
                //                                + um
                //                                        * (k_idx
                //                                                + nb_zmm_a_
                //                                                        / unroll_m_reg_)
                //                                - addr_off_)]);
                (this->*aload)(T_reg(zmm_a_idx_ + m_idx + (k_idx % (nb_zmm_a_ / unroll_m_reg_))
                                               * next_zmm_a), AO_, elt_size_
                                        * (m_idx * nelt_per_vecreg_ + um * (k_idx
                                                    + nb_zmm_a_ / unroll_m_reg_)
                                                - addr_off_));
            }
        }
    }

    template <typename T_reg, typename T_desta, typename T_srca>
    void loadA_after(int um, int un, int k_idx,
            void (jit_generator::*aload)(
                    const T_desta &, const T_srca &, int32_t)) {
        int i;
        //int next_zmm_a = mayiuse(avx512_core)
        //        ? unroll_m_reg_
        //        : std::max(1, um / nelt_per_vecreg_);
        int next_zmm_a = std::max(1, um / nelt_per_vecreg_);
        //if (mayiuse(avx512_core) || (um <= 8) || ((um == 16) && (un == 4))) {
        if ((um <= 8) || ((um == 16) && (un == 4))) {
            for (i = 0; i < std::max(um / nelt_per_vecreg_, 1); i++) {
                //(this->*aload)(T_reg(zmm_a_idx_ + i
                //                       + (k_idx % (nb_zmm_a_ / unroll_m_reg_))
                //                               * next_zmm_a),
                //        ptr[AO_
                //                + elt_size_
                //                        * (i * nelt_per_vecreg_
                //                                + um
                //                                        * (k_idx
                //                                                + nb_zmm_a_
                //                                                        / unroll_m_reg_)
                //                                - addr_off_)]);
                (this->*aload)(T_reg(zmm_a_idx_ + i + (k_idx % (nb_zmm_a_ / unroll_m_reg_))
                                * next_zmm_a), AO_, elt_size_
                                        * (i * nelt_per_vecreg_
                                                + um * (k_idx + nb_zmm_a_ / unroll_m_reg_)
                                                - addr_off_));
            }
        }
    }

    template <typename T_reg, typename T_desta, typename T_srca,
            typename T_destb, typename T_srcb>
    void k_loop_body(int cfetch, int um, int un,
            void (jit_generator::*aload)(
                    const T_desta &, const T_srca &, int32_t),
            void (jit_generator::*bload)(
                    const T_destb &, const T_srcb &, int32_t)) {
        Xbyak_loongarch::Label K_loop_body_label;
        int i, j, p, b_idx;
        //int addb_off = ((!mayiuse(avx512_core)) && (nb_zmm_b_ == 2)) ? 1 : 0;
        int addb_off = (nb_zmm_b_ == 2) ? 1 : 0;

        //int next_zmm_a = mayiuse(avx512_core)
        //        ? unroll_m_reg_
        //        : std::max(1, um / nelt_per_vecreg_);
        int next_zmm_a = std::max(1, um / nelt_per_vecreg_);

        off_ = 0, offb_ = 0;

        //if (mayiuse(avx512_core)) L_aligned(K_loop_body_label);

        if (cfetch) prefetchC_beforeKloop(um);

        //if (!mayiuse(avx512_core)) L_aligned(K_loop_body_label);
        L_aligned(K_loop_body_label);

        for (p = 0; p < unroll_k_; p++) {
            //if (mayiuse(avx512_core)) {
            //    if ((um == unroll_m_) && (p == unroll_k_ - 1)) {
            //        prefetcht2(ptr[AA_ - elt_size_ * 128]);
            //    }
            //}
            for (j = 0; j < un; j++) {

                //b_idx = mayiuse(avx512_core) ? j % nb_zmm_b_ : p % nb_zmm_b_;
                b_idx = p % nb_zmm_b_;

                //if (!mayiuse(avx512_core)) {
                    if ((um == unroll_m_) && (un == unroll_n_)) {
                        if ((j == un - 1) && (p == unroll_k_ - 1))
                            //sub(BO_, -un * unroll_k_ * elt_size_);
                            add_imm(BO_, BO_, un * unroll_k_ * elt_size_, X_TMP_0);
                    }
                //}

                for (i = 0; i < std::max(um / nelt_per_vecreg_, 1); i++) {

                    prefetchA_beforeFMA(um, un, p, j, i);
                    prefetchB_beforeFMA(um, un, p, j, i);

                    //vfmadd231ps(T_reg(zmm_acc_idx_ + i * unroll_n_ + j),
                    //        T_reg(zmm_b_idx_ + b_idx),
                    //        T_reg(i
                    //                + (p % (nb_zmm_a_ / unroll_m_reg_))
                    //                        * next_zmm_a
                    //                + zmm_a_idx_));
                    uni_fmadd_s(T_reg(zmm_acc_idx_ + i * unroll_n_ + j),
                            T_reg(zmm_b_idx_ + b_idx),
                            T_reg(i
                                    + (p % (nb_zmm_a_ / unroll_m_reg_))
                                            * next_zmm_a
                                    + zmm_a_idx_), T_reg(zmm_acc_idx_ + i * unroll_n_ + j));

                    loadA_betweenFMAs<T_reg, T_desta, T_srca>(
                            um, un, p, j, i, aload);

                    prefetchB_afterFMA(p, j, i);
                    prefetchA_afterFMA(um, un, p, j, i);
                }

                prefetchB_beforeBload(um, un, p, j);

                //if (!mayiuse(avx512_core) && (um == unroll_m_)
                if ((um == unroll_m_)
                        && (un == unroll_n_) && (j == un - 1)
                        && (p == unroll_k_ - 1)) {
                    //(this->*bload)(T_reg(zmm_b_idx_ + b_idx),
                    //        ptr[BO_
                    //                + elt_size_
                    //                        * (un * p - addr_off_
                    //                                + std::min(nb_zmm_b_, un)
                    //                                + j + addb_off)
                    //                - un * unroll_k_ * elt_size_]);
                    (this->*bload)(T_reg(zmm_b_idx_ + b_idx), BO_, elt_size_
                                            * (un * p - addr_off_
                                                    + std::min(nb_zmm_b_, un)
                                                    + j + addb_off)
                                    - un * unroll_k_ * elt_size_);
                } else {
                    //(this->*bload)(T_reg(zmm_b_idx_ + b_idx),
                    //        ptr[BO_
                    //                + elt_size_
                    //                        * (un * p - addr_off_
                    //                                + std::min(nb_zmm_b_, un)
                    //                                + j + addb_off)]);
                    (this->*bload)(T_reg(zmm_b_idx_ + b_idx), BO_,
                                    elt_size_ * (un * p - addr_off_
                                                    + std::min(nb_zmm_b_, un)
                                                    + j + addb_off));
                }

                prefetchA_afterBload(um, un, p, j);

                if (cfetch) prefetchC_afterBload(um, un, p, j);

                //if (mayiuse(avx512_core)) {
                //    if ((um == unroll_m_) && (p == unroll_k_ - 1)
                //            && (j == std::min(un - 1, 3)))
                //        lea(AA_, ptr[AA_ + elt_size_ * unroll_n_]);
                //}
            }

            //if (mayiuse(avx512_core)) {
            //    for (j = un; j < unroll_n_; j++) {
            //        if (um < unroll_m_) {
            //            if (((p % (nb_zmm_a_ / unroll_m_reg_) == 0)
            //                        && (j % 6 == 0))
            //                    || ((p % (nb_zmm_a_ / unroll_m_reg_) == 1)
            //                            && (j == 3))) {
            //                off_ += 16;
            //            }
            //        }
            //    }
            //}

            loadA_after<T_reg, T_desta, T_srca>(um, un, p, aload);
        }

        //if (mayiuse(avx512_core)) {
        //    lea(AO_, ptr[AO_ + um * unroll_k_ * elt_size_]);
        //    lea(BO_, ptr[BO_ + un * unroll_k_ * elt_size_]);
        //} else {
            if ((um != unroll_m_) || (un != unroll_n_))
                //sub(BO_, -un * unroll_k_ * elt_size_);
                add_imm(BO_, BO_, un * unroll_k_ * elt_size_, X_TMP_0);
            //sub(AO_, -um * unroll_k_ * elt_size_);
            add_imm(AO_, AO_, um * unroll_k_ * elt_size_, X_TMP_0);
        //}
        //sub(LL_, 1);
        addi_d(LL_, LL_, -1);

        //jg(K_loop_body_label, T_NEAR);
        blt(zero, LL_, K_loop_body_label);
    }

    template <typename T_reg, typename T_desta, typename T_srca,
            typename T_destb, typename T_srcb>
    void k_loop_remainder(int um, int un,
            void (jit_generator::*aload)(
                    const T_desta &, const T_srca &, int32_t),
            void (jit_generator::*bload)(
                    const T_destb &, const T_srcb &, int32_t)) {

        Xbyak_loongarch::Label K_loop_remainder_label;
        int i, j = 0;

        L_aligned(K_loop_remainder_label);

        for (j = 0; j < un; j++) {

            for (i = 0; i < std::max(um / nelt_per_vecreg_, 1); i++) {
                //vfmadd231ps(T_reg(zmm_acc_idx_ + i * unroll_n_ + j),
                //        T_reg(zmm_b_idx_ + (j % nb_zmm_b_)),
                //        T_reg(i + zmm_a_idx_));
                uni_fmadd_s(T_reg(zmm_acc_idx_ + i * unroll_n_ + j),
                        T_reg(zmm_b_idx_ + (j % nb_zmm_b_)),
                        T_reg(i + zmm_a_idx_), T_reg(zmm_acc_idx_ + i * unroll_n_ + j));

                //if (mayiuse(avx512_core)) {
                //    if (i == 0) {
                //        if (j % 3 == 0) {
                //            prefetcht0(ptr[AO_
                //                    + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                //            off_ += 16;
                //        }
                //        if (j == 1)
                //            prefetcht0(ptr[BO_ + elt_size_ * (PREFETCHSIZEB_)]);
                //    }
                //} else {
                    if ((um > nelt_per_vecreg_) && (j == un - 1)) {
                        (this->*aload)(T_reg(zmm_a_idx_ + i),
                                AO_, elt_size_ * (um - addr_off_
                                                        + nelt_per_vecreg_
                                                                * i));
                    }
                //}
            }

            (this->*bload)(T_reg(zmm_b_idx_ + (j % nb_zmm_b_)),
                    BO_, -1 * elt_size_
                                    * (addr_off_ - std::min(nb_zmm_b_, un)
                                            - j));
        }

        //if (mayiuse(avx512_core) && (un < 2))
        //    prefetcht0(ptr[BO_ + elt_size_ * (PREFETCHSIZEB_)]);

        //if (mayiuse(avx512_core)) {
        //    for (i = un; i < 8; i += 4) {
        //        prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
        //        off_ += 16;
        //    }
        //}

        //if (mayiuse(avx512_core) || (um <= nelt_per_vecreg_)) {
        if ((um <= nelt_per_vecreg_)) {
            for (i = 0; i < std::max(um / nelt_per_vecreg_, 1); i++) {
                (this->*aload)(T_reg(zmm_a_idx_ + i),
                        AO_, elt_size_ * (um - addr_off_
                                                + nelt_per_vecreg_ * i));
            }
        }

        //if (mayiuse(avx512_core)) {
        //    lea(AO_, ptr[AO_ + um * elt_size_]);
        //    lea(BO_, ptr[BO_ + un * elt_size_]);
        //} else {
            //sub(AO_, -um * elt_size_);
            add_imm(AO_, AO_, um * elt_size_, X_TMP_0);
            //sub(BO_, -un * elt_size_);
            add_imm(BO_, BO_, un * elt_size_, X_TMP_0);
        //}
        //sub(LL_, 1);
        addi_d(LL_, LL_, -1);

        //jg(K_loop_remainder_label, T_NEAR);
        blt(zero, LL_, K_loop_remainder_label);
    }

    template <typename T_reg, typename T_desta, typename T_srca,
            typename T_destb, typename T_srcb>
    void loop(int um, int un,
            void (jit_generator::*aload)(
                    const T_desta &, const T_srca &, int32_t),
            void (jit_generator::*bload)(
                    const T_destb &, const T_srcb &, int32_t)) {

        int i, j, k, acc_idx;
        Xbyak_loongarch::Label end_K_loop_label, end_main_K_loop_label;
        Xbyak_loongarch::Label K_loop_with_prefetch_label, K_loop_with_prefetch_rem_label;

        //Xbyak::Reg64 A_reg = (mayiuse(avx512_core))
        //        ? AO_
        //        : ((um == unroll_m_) && (un == unroll_n_)) ? A_ : AO_;
        Xbyak_loongarch::XReg A_reg = ((um == unroll_m_) && (un == unroll_n_)) ? A_ : AO_;

        //if (mayiuse(avx512_core) || (unroll_m_ != um) || (unroll_n_ != un))
        if ((unroll_m_ != um) || (unroll_n_ != un))
            //mov(AO_, A_);
            add_d(AO_, A_, zero);

        //if (!mayiuse(avx512_core)) {
        {

            nb_zmm_a_ = unroll_m_reg_;
            nb_zmm_b_ = 1;
            zmm_a_idx_ = 0;
            zmm_b_idx_ = zmm_a_idx_ + nb_zmm_a_;

            if (((um == 16) && (un == 4))
                    || ((um <= nelt_per_vecreg_) && (un != 2))) {
                nb_zmm_a_ = unroll_m_reg_ * 2;
                zmm_b_idx_
                        = zmm_a_idx_ + std::max(1, um / nelt_per_vecreg_) * 2;
            } else {
                zmm_b_idx_ = zmm_a_idx_ + unroll_m_reg_;
            }

            if (un == 1)
                nb_zmm_b_ = 2;
            else
                nb_zmm_b_ = 1;
        }

        zmm_acc_idx_ = zmm_b_idx_ + nb_zmm_b_;
        acc_idx = 0;

        //if (!mayiuse(avx512_core)) {
        {
            j = zmm_b_idx_;
            for (k = 0; k < nb_zmm_b_; k++) {
                //if (!mayiuse(avx512_core) && (un > 1)) {
                if ((un > 1)) {
                    acc_idx = next_acc(acc_idx, um, un);
                    uni_vpxor(T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx));
                    acc_idx++;
                }
                (this->*bload)(
                        T_reg(j), BO_, -1 * (addr_off_ - k) * elt_size_);
                j++;
            }
        }

        for (k = 0; k < nb_zmm_a_ / unroll_m_reg_; k++) {
            //if (mayiuse(avx512_core))
            //    j = zmm_a_idx_ + k * unroll_m_reg_;
            //else
                j = zmm_a_idx_ + k * std::max(1, um / nelt_per_vecreg_);

            for (i = nelt_per_vecreg_; i <= std::max(um, nelt_per_vecreg_);
                    i += nelt_per_vecreg_) {
                //if (!mayiuse(avx512_core)) {
                {
                    acc_idx = next_acc(acc_idx, um, un);
                    uni_vpxor(T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx));
                    acc_idx++;
                }
                (this->*aload)(T_reg(j),
                        A_reg, (um * k - addr_off_ + i - nelt_per_vecreg_)
                                        * elt_size_);
                j++;
            }
        }

        //if (mayiuse(avx512_core)) {
        //    j = zmm_b_idx_;
        //    for (k = 0; k < nb_zmm_b_; k++) {
        //        if (!mayiuse(avx512_core) && (un > 1)) {
        //            acc_idx = next_acc(acc_idx, um, un);
        //            vxorps(T_reg(zmm_acc_idx_ + acc_idx),
        //                    T_reg(zmm_acc_idx_ + acc_idx),
        //                    T_reg(zmm_acc_idx_ + acc_idx));
        //            acc_idx++;
        //        }
        //        (this->*bload)(
        //                T_reg(j), ptr[BO_ - (addr_off_ - k) * elt_size_]);
        //        j++;
        //    }
        //}

        //if (!mayiuse(avx512_core)) {
        {

            if (un > 1) {
                if ((um == unroll_m_)
                        || ((um <= nelt_per_vecreg_) && (un == unroll_n_)
                                && (um > 1))) {
                    acc_idx = next_acc(acc_idx, um, un);
                    uni_vpxor(T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx));
                    acc_idx++;
                    acc_idx = next_acc(acc_idx, um, un);
                    uni_vpxor(T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx));
                    acc_idx++;
                }
                //prefetcht0(ptr[CO1_ + elt_size_ * ((um - 1) % 16)]);
                uni_preld(0, CO1_, elt_size_ * ((um - 1) % 16));
                if ((un < unroll_n_) && (um == unroll_m_)) {
                    acc_idx = next_acc(acc_idx, um, un);
                    uni_vpxor(T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx));
                    acc_idx++;
                    acc_idx = next_acc(acc_idx, um, un);
                    uni_vpxor(T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx),
                            T_reg(zmm_acc_idx_ + acc_idx));
                    acc_idx++;
                }
                //prefetcht0(ptr[CO1_ + LDC_ + elt_size_ * ((um - 1) % 16)]);
                add_d(X_TMP_1, CO1_, LDC_);
                uni_preld(0, X_TMP_1, elt_size_ * ((um - 1) % 16));
                if (un == unroll_n_) {
                    if ((um == unroll_m_)
                            || ((um <= nelt_per_vecreg_) && (un == unroll_n_)
                                    && (um > 1))) {
                        acc_idx = next_acc(acc_idx, um, un);
                        uni_vpxor(T_reg(zmm_acc_idx_ + acc_idx),
                                T_reg(zmm_acc_idx_ + acc_idx),
                                T_reg(zmm_acc_idx_ + acc_idx));
                        acc_idx++;
                    }
                    //prefetcht0(ptr[CO2_ + elt_size_ * ((um - 1) % 16)]);
                    uni_preld(0, CO2_, elt_size_ * ((um - 1) % 16));
                    if ((um == unroll_m_)
                            || ((um <= nelt_per_vecreg_) && (un == unroll_n_)
                                    && (um > 1))) {
                        acc_idx = next_acc(acc_idx, um, un);
                        uni_vpxor(T_reg(zmm_acc_idx_ + acc_idx),
                                T_reg(zmm_acc_idx_ + acc_idx),
                                T_reg(zmm_acc_idx_ + acc_idx));
                        acc_idx++;
                    }
                    //prefetcht0(ptr[CO2_ + LDC_ + elt_size_ * ((um - 1) % 16)]);
                    add_d(X_TMP_1, CO2_, LDC_);
                    uni_preld(0, X_TMP_1, elt_size_ * ((um - 1) % 16));
                }

            } else {
                //prefetcht0(ptr[CO1_ + ((um - 1) % 16) * elt_size_]);
                uni_preld(0, CO1_, ((um - 1) % 16) * elt_size_);
                //if (um == unroll_m_) prefetcht0(ptr[CO1_ + 23 * elt_size_]);
                if (um == unroll_m_) uni_preld(0, CO1_, 23 * elt_size_);
            }

            for (i = zmm_acc_idx_ + acc_idx;
                    i <= std::min(15,
                            zmm_acc_idx_
                                    + (std::max(1, um / nelt_per_vecreg_) - 1)
                                            * unroll_n_
                                    + un - 1);
                    i++)
                uni_vpxor(T_reg(i), T_reg(i), T_reg(i));
        }

        //if (!((mayiuse(avx512_core) || (unroll_m_ != um) || (unroll_n_ != un))))
        if (!((unroll_m_ != um) || (unroll_n_ != un)))
            //mov(AO_, A_);
            add_d(AO_, A_, zero);

        //mov(LL_, KK_);
        add_d(LL_, KK_, zero);
        //sar(LL_, unroll_k_bin_);
        srai_d(LL_, LL_, unroll_k_bin_);
        //jle(end_main_K_loop_label, T_NEAR);
        bge(zero, LL_, end_main_K_loop_label);

        //if (mayiuse(avx512_core)
        //        || (!mayiuse(avx512_core) && (un == unroll_n_)
        //                && (um == unroll_m_))) {
        if ((un == unroll_n_) && (um == unroll_m_)) {
            //sub(LL_, second_fetch_);
            addi_d(LL_, LL_, -1 * second_fetch_);
            //jle(K_loop_with_prefetch_label, T_NEAR);
            bge(zero, LL_, K_loop_with_prefetch_label);
        }

        k_loop_body<T_reg, T_desta, T_srca, T_destb, T_srcb>(
                0, um, un, aload, bload);

        //if (mayiuse(avx512_core)
        //        || (!mayiuse(avx512_core) && (un == unroll_n_)
        //                && (um == unroll_m_))) {
        if ((un == unroll_n_) && (um == unroll_m_)) {
            L_aligned(K_loop_with_prefetch_label);
        }

        //if (mayiuse(avx512_core)) {
        //    lea(CO2_, ptr[CO1_ + (nelt_per_vecreg_ - 1) * elt_size_]);
        //    add(LL_, un);
        //    jle(K_loop_with_prefetch_rem_label, T_NEAR);
        //}

        //if (mayiuse(avx512_core)
        //        || (!mayiuse(avx512_core) && (un == unroll_n_)
        //                && (um == unroll_m_))) {
        if ((un == unroll_n_) && (um == unroll_m_)) {
            k_loop_body<T_reg, T_desta, T_srca, T_destb, T_srcb>(
                    1, um, un, aload, bload);
        }

        //if (mayiuse(avx512_core)) {
        //    L_aligned(K_loop_with_prefetch_rem_label);
        //    add(LL_, second_fetch_ - un);
        //    jle(end_main_K_loop_label, T_NEAR);

        //    k_loop_body<T_reg, T_desta, T_srca, T_destb, T_srcb>(
        //            0, um, un, aload, bload);
        //}

        L_aligned(end_main_K_loop_label);

        //if (!mayiuse(avx512_core)) {
        {
            if ((un == unroll_n_) && ((um == 16) || (um == 8))) {
                //prefetcht2(ptr[AA_ - 16 * elt_size_]);
                uni_preld(2, AA_, -1 * 16 * elt_size_);
            }
        }

        //mov(LL_, KK_);
        add_d(LL_, KK_, zero);
        //and_(LL_, 3);
        andi(LL_, LL_, 3);
        //je(end_K_loop_label, T_NEAR);
        beq(LL_, zero, end_K_loop_label);

        k_loop_remainder<T_reg, T_desta, T_srca, T_destb, T_srcb>(
                um, un, aload, bload);

        L_aligned(end_K_loop_label);
    }

    template <typename vec_reg_t, typename T_src>
    void update(int um, int un, int sepload, bool is_beta_zero,
            void (jit_generator::*load)(const vec_reg_t &,
                    const vec_reg_t &, const vec_reg_t &),
            void (jit_generator::*store)(
                    const vec_reg_t &, const Xbyak_loongarch::XReg &, int32_t),
            void (jit_generator::*sload)(
                    const vec_reg_t &, const T_src &, int32_t)) {

        int i, j, offAA = 16;
        Xbyak_loongarch::XReg reg_C(0);

        if ((um < unroll_m_) && (um >= nelt_per_vecreg_))
            offAA = 32 - (un / 2) * 16;

        //if (mayiuse(avx512_core))
        //    lea(CO2_, ptr[CO1_ + LDC_]);
        //else {
            if ((um == nelt_per_vecreg_) && (un == unroll_n_)) {
                //prefetcht2(ptr[AA_ + elt_size_ * offAA]);
                uni_preld(2, AA_, elt_size_ * offAA);
                offAA += 16;
            }
        //}
        for (j = 0; j < un; j++) {

            //if (mayiuse(avx512_core)) {
            //    reg_C = (j == 0) ? CO1_ : CO2_;
            //    if (j >= 2) { add(CO2_, LDC_); }
            //} else
                reg_C = (j < 2) ? CO1_ : CO2_;

            // if beta == 1 load C_ and add to accumulator
            if (!is_beta_zero) {
                if (sepload) {
                    for (i = 0; i < std::max(um / nelt_per_vecreg_, 1); i++) {
                        //if (!mayiuse(avx512_core) && (j % 2 == 1)) {
                        if ((j % 2 == 1)) {
                            //(this->*sload)(vec_reg_t(i),
                            //        ptr[reg_C + LDC_
                            //                + elt_size_ * i
                            //                        * nelt_per_vecreg_]);
                            add_d(X_TMP_1, reg_C, LDC_);
                            (this->*sload)(vec_reg_t(i),
                                    X_TMP_1, elt_size_ * i
                                                    * nelt_per_vecreg_);
                        } else {
                            //(this->*sload)(vec_reg_t(i),
                            //        ptr[reg_C
                            //                + elt_size_ * i
                            //                        * nelt_per_vecreg_]);
                            (this->*sload)(vec_reg_t(i),
                                    reg_C, elt_size_ * i * nelt_per_vecreg_);
                        }
                        (this->*load)(
                                vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                                vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                                vec_reg_t(i));
                    }
                } else {
                    for (i = 0; i < std::max(um / nelt_per_vecreg_, 1); i++) {
                        //(this->*load)(
                        //        vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                        //        vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                        //        ptr[reg_C + elt_size_ * i * nelt_per_vecreg_]);
                        uni_xvld(vec_reg_t(31), reg_C, elt_size_ * i * nelt_per_vecreg_);
                        (this->*load)(
                                vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                                vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                                vec_reg_t(31));
                    }
                }
            }

            //if (!mayiuse(avx512_core)) {
            {
                if (j > 0) {
                    //prefetcht2(ptr[AA_ + elt_size_ * offAA]);
                    uni_preld(2, AA_, elt_size_ * offAA);
                    offAA += 16;
                }
            }

            // store accumulated value in C_
            for (i = 0; i < std::max(um / nelt_per_vecreg_, 1); i++) {
                //if (!mayiuse(avx512_core) && (j % 2 == 1)) {
                if ((j % 2 == 1)) {
                    //(this->*store)(ptr[reg_C + LDC_
                    //                       + elt_size_ * i * nelt_per_vecreg_],
                    //        vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_));
                    add_d(X_TMP_1, reg_C, LDC_);
                    (this->*store)(vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_), X_TMP_1,
                                            elt_size_ * i * nelt_per_vecreg_);
                } else {
                    //(this->*store)(
                    //        ptr[reg_C + elt_size_ * i * nelt_per_vecreg_],
                    //        vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_));
                    (this->*store)(vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                            reg_C, elt_size_ * i * nelt_per_vecreg_);
                }
                //if (mayiuse(avx512_core)) {
                //    vpxorq(vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                //            vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_),
                //            vec_reg_t(zmm_acc_idx_ + j + i * unroll_n_));
                //}
            }

            //if (!mayiuse(avx512_core)) {
            {
                if ((um == unroll_m_) && (un == 1)) {
                    //prefetcht2(ptr[AA_ + elt_size_ * offAA]);
                    uni_preld(2, AA_, elt_size_ * offAA);
                    offAA += 16;
                }
            }

            //if (!mayiuse(avx512_core)) {
            {
                if (j == std::min(1, un - 1)) {
                    if (j == 0)
                        //add(CO1_, LDC_);
                        add_d(CO1_, CO1_, LDC_);
                    else {
                        //lea(CO1_, ptr[CO1_ + LDC_ * un]);
                        mov_imm(X_TMP_0, un);
                        mul_d(X_TMP_0, LDC_, X_TMP_0);
                        add_d(CO1_, CO1_, X_TMP_0);
                    }
                }
                if (j == (un - 1)) {
                    if (j == 0)
                        //add(CO2_, LDC_);
                        add_d(CO2_, CO2_, LDC_);
                    else {
                        //lea(CO2_, ptr[CO2_ + LDC_ * un]);
                        mov_imm(X_TMP_0, un);
                        mul_d(X_TMP_0, LDC_, X_TMP_0);
                        add_d(CO2_, CO2_, X_TMP_0);
                    }
                }
            }
        }

        //if (mayiuse(avx512_core)) lea(CO1_, ptr[CO2_ + LDC_]);

        //if (!mayiuse(avx512_core)) {
        {
            if ((um >= nelt_per_vecreg_) && (un < unroll_n_)) {
                //prefetcht2(ptr[AA_ + elt_size_ * offAA]);
                uni_preld(2, AA_, elt_size_ * offAA);
                offAA += 16;
            }
        }
    }

public:
    jit_lasx_kernel_sgemm_kern(bool beta_zero);
};
} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_LOONGARCH64_GEMM_F32_JIT_LASX_KERNEL_SGEMM_KERN_HPP
