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

#include "cpu/loongarch64/gemm/f32/jit_lasx_kernel_sgemm_kern.hpp"

#ifdef _WIN32
static const bool is_windows = true;
#else
static const bool is_windows = false;
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

int jit_lasx_kernel_sgemm_kern::next_acc(int idx, int um, int un) const {
    while (!(((idx / unroll_n_) < std::max(1, um / nelt_per_vecreg_))
            || ((idx % unroll_n_) < un)))
        idx++;
    return idx;
}

void jit_lasx_kernel_sgemm_kern::prefetchB_beforeBload(
        int um, int un, int k_idx, int n_idx) {
    //if (!mayiuse(avx512_core)) {
    {
        if ((n_idx == 0) && (k_idx == 0) && (un == unroll_n_) && (um != 16)) {
            //prefetcht0(ptr[BO_ + elt_size_ * (PREFETCHSIZEB_ + offb_)]);
            uni_preld(0, BO_, elt_size_ * (PREFETCHSIZEB_ + offb_));
            offb_ += 16;
        }
    }
}

void jit_lasx_kernel_sgemm_kern::prefetchB_beforeFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
    //if (!mayiuse(avx512_core)) {
    {
        if ((um == 16) || (un < unroll_n_)) {
            if ((k_idx + m_idx + n_idx) == 0) {
                //prefetcht0(ptr[BO_ + elt_size_ * (PREFETCHSIZEB_ + offb_)]);
                uni_preld(0, BO_, elt_size_ * (PREFETCHSIZEB_ + offb_));
                offb_ += 16;
            }
            if ((um == 16) && (un == 4) && (k_idx == 2)
                    && ((m_idx + n_idx) == 0)) {
                //prefetcht0(ptr[BO_ + elt_size_ * (PREFETCHSIZEB_ + offb_)]);
                uni_preld(0, BO_, elt_size_ * (PREFETCHSIZEB_ + offb_));
                offb_ += 16;
            }
        }
    }
}

void jit_lasx_kernel_sgemm_kern::prefetchA_afterFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
    //if (mayiuse(avx512_core)) {
    //    if ((um < unroll_m_) && (m_idx == 0)) {
    //        if (((k_idx % (nb_zmm_a_ / unroll_m_reg_) == 0) && (n_idx % 6 == 0))
    //                || ((k_idx % (nb_zmm_a_ / unroll_m_reg_) == 1)
    //                        && (n_idx == 3))) {
    //            prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
    //            off_ += 16;
    //        }
    //    }
    //} else {
        if (un == unroll_n_) {
            if (((um < nelt_per_vecreg_) && (n_idx == 0)
                        && (k_idx == std::min(2, nelt_per_vecreg_ / um - 1)))
                    || ((um == nelt_per_vecreg_) && (un == unroll_n_)
                            && (n_idx == 1) && (k_idx == 0))) {
                //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                off_ += 16;
            }
        }
    //}
}

void jit_lasx_kernel_sgemm_kern::prefetchA_afterBload(
        int um, int un, int k_idx, int n_idx) {
    //if (!mayiuse(avx512_core)) {
    {
        if ((um == unroll_m_) && (un == 2)) {
            if (k_idx % 3 == 0) {
                if (n_idx == 1) {
                    if (k_idx == 0) off_ += 16;
                    //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                    uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                    off_ += 16;
                }
                if ((k_idx == 0) && (n_idx == 0)) {
                    //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                    uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                    off_ += 16;
                }
            } else {
                if (n_idx == 1) {
                    //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                    uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                    off_ += 16;
                }
            }
        }
    }
}

void jit_lasx_kernel_sgemm_kern::prefetchB_afterFMA(
        int k_idx, int n_idx, int m_idx) {
    //if (mayiuse(avx512_core)) {
    //    if (((m_idx + (k_idx % (nb_zmm_a_ / unroll_m_reg_)) * unroll_m_reg_)
    //                == 0)
    //            && (n_idx == 1)) {
    //        prefetcht0(ptr[BO_
    //                + elt_size_
    //                        * (PREFETCHSIZEB_
    //                                + nelt_per_vecreg_ * k_idx
    //                                        / (nb_zmm_a_ / unroll_m_reg_))]);
    //    }
    //}
}

void jit_lasx_kernel_sgemm_kern::prefetchA_beforeFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
    //if (!mayiuse(avx512_core)) {
    {
        if ((um == unroll_m_) && (un == unroll_n_)) {
            if (((k_idx == 0) && (n_idx % 2 == 1) && (m_idx == 0))
                    || ((k_idx == 1) && (n_idx == 2) && (m_idx == 0))
                    || ((k_idx == 2) && (n_idx == 0) && (m_idx == 2))
                    || ((k_idx == 2) && (n_idx == 3) && (m_idx == 0))
                    || ((k_idx == 3) && (n_idx == 1) && (m_idx == 0))) {
                //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                off_ += 16;
            }
        }
        if ((um == unroll_m_) && (un == 1)) {
            if (m_idx == 2) {
                //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                off_ += 16;
            } else if ((m_idx == 0) && ((k_idx == 1) || (k_idx == 2))) {
                //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                off_ += 16;
            }
        }
        if ((um == 16) && (un == unroll_n_) && (m_idx == 0) && (n_idx == 2)) {
            //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
            uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
            off_ += 16;
        }
        if ((um == 8) && (un == unroll_n_) && (m_idx == 0) && (n_idx == 1)
                && (k_idx == 2)) {
            //prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
            uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
            off_ += 16;
        }
    }
}

void jit_lasx_kernel_sgemm_kern::prefetchC_afterBload(
        int um, int un, int k_idx, int n_idx) {
    //if (mayiuse(avx512_core)) {
    //    if (um == unroll_m_) {
    //        if (n_idx == std::min(1, un - 1)) {
    //            if (k_idx == unroll_k_ - 1)
    //                lea(CO2_, ptr[CO2_ + LDC_]);
    //            else
    //                prefetchw(ptr[CO2_ + elt_size_ * k_idx * nelt_per_vecreg_]);
    //        }
    //    }
    //}
}

void jit_lasx_kernel_sgemm_kern::prefetchC_beforeKloop(int um) {
    //if (mayiuse(avx512_core)) {
    //    if (um < unroll_m_) {
    //        prefetchw(ptr[CO2_ + elt_size_ * 0]);
    //        prefetchw(ptr[CO2_ + elt_size_ * 8]);
    //        if (um <= 16) prefetchw(ptr[CO2_ + elt_size_ * 16]);
    //        lea(CO2_, ptr[CO2_ + LDC_]);
    //    }
    //} else {
        //prefetcht2(ptr[AA_ - 16 * elt_size_]);
        uni_preld(2, AA_, -16 * elt_size_);

        //prefetcht0(ptr[CO1_ + 7 * elt_size_]);
        uni_preld(0, CO1_, 7 * elt_size_);
        //prefetcht0(ptr[CO1_ + LDC_ + 7 * elt_size_]);
        add_d(X_TMP_1, CO1_, LDC_);
        uni_preld(0, X_TMP_1, 7 * elt_size_);
        //prefetcht0(ptr[CO2_ + 7 * elt_size_]);
        uni_preld(0, CO2_, 7 * elt_size_);
        //prefetcht0(ptr[CO2_ + LDC_ + 7 * elt_size_]);
        add_d(X_TMP_1, CO2_, LDC_);
        uni_preld(0, X_TMP_1, 7 * elt_size_);

        //prefetcht0(ptr[CO1_ + 23 * elt_size_]);
        uni_preld(0, CO1_, 23 * elt_size_);
        //prefetcht0(ptr[CO1_ + LDC_ + 23 * elt_size_]);
        add_d(X_TMP_1, CO1_, LDC_);
        uni_preld(0, X_TMP_1, 23 * elt_size_);
        //prefetcht0(ptr[CO2_ + 23 * elt_size_]);
        uni_preld(0, CO2_, 23 * elt_size_);
        //prefetcht0(ptr[CO2_ + LDC_ + 23 * elt_size_]);
        add_d(X_TMP_1, CO2_, LDC_);
        uni_preld(0, X_TMP_1, 23 * elt_size_);

        //add(LL_, second_fetch_);
        addi_d(LL_, LL_, second_fetch_);

        //prefetcht2(ptr[AA_]);
        preld(0, AA_, 0);
    //}
}

void jit_lasx_kernel_sgemm_kern::generate() {

    int i, unroll_x, unroll_y, uy_bin, ux_bin;
    int C_off = is_windows ? 56 : 8;
    int LDC_off = is_windows ? 64 : 16;
    int sepload = 0;

    std::vector<Xbyak_loongarch::Label> unroll_x_label(MAX_UNROLL_M),
            unroll_y_label((MAX_UNROLL_N_BIN + 1) * MAX_UNROLL_M);
    std::vector<Xbyak_loongarch::Label> end_n_loop_label(MAX_UNROLL_M);
    Xbyak_loongarch::Label end_m_loop_label;

    preamble();

    //if (is_windows) {
    //    mov(M_, ptr[rcx]);
    //    mov(N_, ptr[rdx]);
    //    mov(K_, ptr[r8]);
    //    mov(A_, ptr[rsp + get_size_of_abi_save_regs() + 40]);
    //    mov(B_, ptr[rsp + get_size_of_abi_save_regs() + 48]);
    //} else {
        //mov(M_, ptr[M_]);
        ld_d(M_, M_, 0);
        //mov(N_, ptr[N_]);
        ld_d(N_, N_, 0);
        //mov(K_, ptr[K_]);
        ld_d(K_, K_, 0);
    //}

    //mov(C_, ptr[rsp + get_size_of_abi_save_regs() + C_off]);
    uni_ld_d(C_, sp, get_size_of_abi_save_regs() + C_off);
    //mov(LDC_, ptr[rsp + get_size_of_abi_save_regs() + LDC_off]);
    uni_ld_d(LDC_, sp, get_size_of_abi_save_regs() + LDC_off);

    //if (mayiuse(avx512_core)) {
    //    for (i = zmm_acc_idx_; i < unroll_m_reg_ * unroll_n_ + zmm_acc_idx_;
    //            i++)
    //        vpxorq(Xbyak::Zmm(i), Xbyak::Zmm(i), Xbyak::Zmm(i));
    //}

    //sub(A_, -addr_off_ * elt_size_);
    add_imm(A_, A_, addr_off_ * elt_size_, X_TMP_0);
    //sub(B_, -addr_off_ * elt_size_);
    add_imm(B_, B_, addr_off_ * elt_size_, X_TMP_0);

    //sal(LDC_, elt_size_bin_);
    slli_d(LDC_, LDC_, elt_size_bin_);

    for (unroll_x = unroll_m_, i = 0, ux_bin = unroll_m_bin_; unroll_x >= 1;
            unroll_x -= std::min(nelt_per_vecreg_, std::max(1, unroll_x / 2)),
        i++, ux_bin--) {

        if (unroll_x == unroll_m_) {
            //mov(J_, M_);
            add_d(J_, M_, zero);
            //cmp(J_, unroll_m_);
            mov_imm(X_TMP_1, unroll_m_);
            //jl(unroll_x_label[i + 1], T_NEAR);
            blt(J_, X_TMP_1, unroll_x_label[i + 1]);
            L_aligned(unroll_x_label[i]);
        } else {
            L_aligned(unroll_x_label[i]);
            //test(J_, unroll_x);
            andi(X_TMP_1, J_, unroll_x);
            if (unroll_x > 1)
                //jle(unroll_x_label[i + 1], T_NEAR);
                bge(zero, X_TMP_1, unroll_x_label[i + 1]);
            else
                //jle(end_m_loop_label, T_NEAR);
                bge(zero, X_TMP_1, end_m_loop_label);
        }

        //mov(AA_, KK_);
        add_d(AA_, KK_, zero);

        if ((1 << ux_bin) > unroll_x) {
            //imul(AA_, AA_, unroll_x * elt_size_);
            mov_imm(X_TMP_1, unroll_x * elt_size_);
            mul_d(AA_, AA_, X_TMP_1);
        }
        else
            //sal(AA_, elt_size_bin_ + ux_bin);
            slli_d(AA_, AA_, elt_size_bin_ + ux_bin);

        //add(AA_, A_);
        add_d(AA_, AA_, A_);
        //mov(CO1_, C_);
        add_d(CO1_, C_, zero);

        //if ((unroll_x == unroll_m_) || (!mayiuse(avx512_core)))
        //    lea(CO2_, ptr[C_ + LDC_ * 2]);
        mov_imm(X_TMP_1, 2);
        mul_d(X_TMP_1, LDC_, X_TMP_1);
        add_d(CO2_, C_, X_TMP_1);

        //add(C_, unroll_x * elt_size_);
        add_imm(C_, C_, unroll_x * elt_size_, X_TMP_0);
        //mov(BO_, B_);
        add_d(BO_, B_, zero);

        for (unroll_y = unroll_n_, uy_bin = unroll_n_bin_; unroll_y >= 1;
                unroll_y /= 2, uy_bin--) {

            if (unroll_y == unroll_n_) {
                //mov(I_, N_);
                add_d(I_, N_, zero);
                //sar(I_, uy_bin);
                srai_d(I_, I_, uy_bin);
                //jle(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin - 1],
                //        T_NEAR);
                bge(zero, I_, unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin - 1]);
                L_aligned(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin]);
            } else {
                L_aligned(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin]);
                //test(N_, unroll_y);
                andi(X_TMP_1, N_, unroll_y);
                if (uy_bin == 0)
                    //jle(end_n_loop_label[i], T_NEAR);
                    bge(zero, X_TMP_1, end_n_loop_label[i]);
                else
                    //jle(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin - 1],
                    //        T_NEAR);
                    bge(zero, X_TMP_1, unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin - 1]);
            }

            //if (!mayiuse(avx512_core))
            //    prefetcht2(ptr[AA_ - addr_off_ * elt_size_]);
            uni_preld(2, AA_, -1 * addr_off_ * elt_size_);

            switch (unroll_x) {
                case 8:
                    //if (mayiuse(avx512_core)) {
                    //    loop<Xbyak::Zmm, Xbyak::Zmm, Xbyak::Address, Xbyak::Xmm,
                    //            Xbyak::Operand>(unroll_x, unroll_y,
                    //            &Xbyak::CodeGenerator::vbroadcastf64x4,
                    //            &Xbyak::CodeGenerator::vbroadcastss);
                    //    update<Xbyak::Ymm, Xbyak::Operand>(unroll_x, unroll_y,
                    //            0, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                    //            &Xbyak::CodeGenerator::vmovups,
                    //            &Xbyak::CodeGenerator::vmovups);
                    //} else {
                        loop<Xbyak_loongarch::XVReg, Xbyak_loongarch::XVReg, Xbyak_loongarch::XReg, Xbyak_loongarch::XVReg,
                                Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                                //&Xbyak::CodeGenerator::vmovups,
                                &jit_generator::uni_xvld,
                                //&Xbyak::CodeGenerator::vbroadcastss);
                                &jit_generator::uni_xvldrepl_w);
                        update<Xbyak_loongarch::XVReg, Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                                //1, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                                1, beta_zero_, &jit_generator::uni_fadd_s,
                                //&Xbyak::CodeGenerator::vmovups,
                                &jit_generator::uni_xvst,
                                //&Xbyak::CodeGenerator::vmovups);
                                &jit_generator::uni_xvld);
                    //}
                    break;
                case 4:
                    //if (mayiuse(avx512_core)) {
                    //    loop<Xbyak::Zmm, Xbyak::Ymm, Xbyak::Address, Xbyak::Xmm,
                    //            Xbyak::Operand>(unroll_x, unroll_y,
                    //            &Xbyak::CodeGenerator::vbroadcastf32x4,
                    //            &Xbyak::CodeGenerator::vbroadcastss);
                    //    sepload = 0;
                    //} else {
                        loop<Xbyak_loongarch::VReg, Xbyak_loongarch::VReg, Xbyak_loongarch::XReg, Xbyak_loongarch::VReg,
                                Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                                //&Xbyak::CodeGenerator::vmovups,
                                &jit_generator::uni_xvld,
                                //&Xbyak::CodeGenerator::vbroadcastss);
                                &jit_generator::uni_xvldrepl_w);
                        sepload = 1;
                    //}

                    update<Xbyak_loongarch::VReg, Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                            //sepload, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                            sepload, beta_zero_, &jit_generator::uni_fadd_s,
                            //&Xbyak::CodeGenerator::vmovups,
                            &jit_generator::uni_xvst,
                            //&Xbyak::CodeGenerator::vmovups);
                            &jit_generator::uni_xvld);

                    break;
                case 2:
                    //if (mayiuse(avx512_core)) {
                    //    loop<Xbyak::Zmm, Xbyak::Ymm, Xbyak::Operand, Xbyak::Xmm,
                    //            Xbyak::Operand>(unroll_x, unroll_y,
                    //            &Xbyak::CodeGenerator::vbroadcastsd,
                    //            &Xbyak::CodeGenerator::vbroadcastss);
                    //} else {
                        loop<Xbyak_loongarch::VReg, Xbyak_loongarch::VReg, Xbyak_loongarch::XReg, Xbyak_loongarch::VReg,
                                Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                                //&Xbyak::CodeGenerator::vmovddup,
                                &jit_generator::uni_xvld,
                                //&Xbyak::CodeGenerator::vbroadcastss);
                                &jit_generator::uni_xvldrepl_w);
                    //}
                    //update<Xbyak::Xmm, Xbyak::Address>(unroll_x, unroll_y, 1,
                    update<Xbyak_loongarch::VReg, Xbyak_loongarch::XReg>(unroll_x, unroll_y, 1,
                            //beta_zero_, &Xbyak::CodeGenerator::vaddps,
                            beta_zero_, &jit_generator::uni_fadd_s,
                            //&Xbyak::CodeGenerator::vmovlps,
                            &jit_generator::uni_xvstelm_d0,
                            //&Xbyak::CodeGenerator::vmovsd);
                            &jit_generator::uni_xvld);
                    break;
                case 1:
                    //if (mayiuse(avx512_core)) {
                    //    loop<Xbyak::Zmm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                    //            Xbyak::Operand>(unroll_x, unroll_y,
                    //            &Xbyak::CodeGenerator::vbroadcastss,
                    //            &Xbyak::CodeGenerator::vbroadcastss);
                    //    sepload = 0;
                    //} else {
                        //loop<Xbyak::Xmm, Xbyak::Xmm, Xbyak::Address, Xbyak::Xmm,
                        loop<Xbyak_loongarch::VReg, Xbyak_loongarch::VReg, Xbyak_loongarch::XReg, Xbyak_loongarch::VReg,
                                //Xbyak::Address>(unroll_x, unroll_y,
                                Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                                //&Xbyak::CodeGenerator::vmovss,
                                &jit_generator::uni_xvldrepl_w,
                                //&Xbyak::CodeGenerator::vmovss);
                                &jit_generator::uni_xvldrepl_w);
                        sepload = 1;
                    //}
                    //update<Xbyak::Xmm, Xbyak::Address>(unroll_x, unroll_y,
                    update<Xbyak_loongarch::VReg, Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                            //sepload, beta_zero_, &Xbyak::CodeGenerator::vaddss,
                            sepload, beta_zero_, &jit_generator::uni_fadd_s,
                            //&Xbyak::CodeGenerator::vmovss,
                            &jit_generator::uni_xvstelm_w0,
                            //&Xbyak::CodeGenerator::vmovss);
                            &jit_generator::uni_xvldrepl_w);

                    break;
                default:
                    //if (mayiuse(avx512_core)) {
                    //    loop<Xbyak::Zmm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                    //            Xbyak::Operand>(unroll_x, unroll_y,
                    //            &Xbyak::CodeGenerator::vmovups,
                    //            &Xbyak::CodeGenerator::vbroadcastss);
                    //    update<Xbyak::Zmm, Xbyak::Operand>(unroll_x, unroll_y,
                    //            0, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                    //            &Xbyak::CodeGenerator::vmovups,
                    //            &Xbyak::CodeGenerator::vmovups);
                    //} else {
                        //loop<Xbyak::Ymm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                        loop<Xbyak_loongarch::XVReg, Xbyak_loongarch::XVReg, Xbyak_loongarch::XReg, Xbyak_loongarch::XVReg,
                                //Xbyak::Operand>(unroll_x, unroll_y,
                                Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                                //&Xbyak::CodeGenerator::vmovups,
                                &jit_generator::uni_xvld,
                                //&Xbyak::CodeGenerator::vbroadcastss);
                                &jit_generator::uni_xvldrepl_w);
                        //update<Xbyak::Ymm, Xbyak::Operand>(unroll_x, unroll_y,
                        update<Xbyak_loongarch::XVReg, Xbyak_loongarch::XReg>(unroll_x, unroll_y,
                                //1, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                                1, beta_zero_, &jit_generator::uni_fadd_s,
                                //&Xbyak::CodeGenerator::vmovups,
                                &jit_generator::uni_xvst,
                                //&Xbyak::CodeGenerator::vmovups);
                                &jit_generator::uni_xvld);
                    //}

                    break;
            }

            //if (mayiuse(avx512_core)) {
            //    sub(AA_, -16 * elt_size_);
            //} else {
                if ((unroll_y != unroll_n_) || (unroll_x <= 4)) {
                    if (unroll_x == unroll_m_)
                        //sub(AA_, -16 * elt_size_);
                        add_imm(AA_, AA_, 16 * elt_size_, X_TMP_0);
                    else
                        //sub(AA_, -32 * elt_size_);
                        add_imm(AA_, AA_, 32 * elt_size_, X_TMP_0);
                } else
                    //sub(AA_, -48 * elt_size_);
                    add_imm(AA_, AA_, 48 * elt_size_, X_TMP_0);
            //}

            if (unroll_y == unroll_n_) {
                //dec(I_);
                addi_d(I_, I_, -1);
                //jg(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin], T_NEAR);
                blt(zero, I_, unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin]);
            }
        }

        L_aligned(end_n_loop_label[i]);

        //mov(A_, AO_);
        add_d(A_, AO_, zero);

        if (unroll_x == unroll_m_) {
            //sub(J_, unroll_x);
            add_imm(J_, J_, -1 * unroll_x, X_TMP_0);
            //cmp(J_, unroll_x);
            mov_imm(X_TMP_1, unroll_x);
            //jge(unroll_x_label[i], T_NEAR);
            bge(J_, X_TMP_1, unroll_x_label[i]);
        }
    }

    L_aligned(end_m_loop_label);

    postamble();
}

jit_lasx_kernel_sgemm_kern::jit_lasx_kernel_sgemm_kern(bool beta_zero)
    : jit_generator(nullptr, 65536) {

    beta_zero_ = beta_zero;
}
} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
