/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "cpu/loongarch64/cpu_isa_traits.hpp"
#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/gemm/f32/jit_lasx_gemv_t_f32_kern.hpp"

//#ifdef _WIN32
//static const bool is_windows = true;
//#else
//static const bool is_windows = false;
//#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace Xbyak_loongarch;

// Convert between vector register lengths.
//static inline Xmm make_xmm(const Xmm &v) {
static inline VReg make_xmm(const XVReg &v) {
    return VReg(v.getIdx());
}

/* just use load_bytes and store_bytes in loongarch
// Load vector register data for x, y or A.
void jit_lasx_gemv_t_f32_kern::v_load(
        const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems) {
    switch (nelems) {
        case 1: vmovss(make_xmm(dst), src); break;
        case 2: vmovsd(make_xmm(dst), src); break;
        case 4: vmovups(make_xmm(dst), src); break;
        default:
            assert(nelems >= 8);
            vmovups(dst, src);
            break;
    }
}

// Store vector register data for x, y or A.
void jit_lasx_gemv_t_f32_kern::v_store(
        const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems) {
    switch (nelems) {
        case 1: vmovss(dst, make_xmm(src)); break;
        case 2: vmovsd(dst, make_xmm(src)); break;
        case 4: vmovups(dst, make_xmm(src)); break;
        default:
            assert(nelems >= 8);
            vmovups(dst, src);
            break;
    }
} */

/* this is just vfmadd_s in loongarch
// Perform Hadamard product of 2 vectors and accumulate.
// Use FMA instruction, otherwise emulate.
void jit_lasx_gemv_t_f32_kern::dot_product(
        //const Xmm &dst, const Xmm &src1, const Xmm &src2) {
        const VReg &dst, const VReg &src1, const VReg &src2) {
    if (is_avx2_)
        vfmadd231ps(dst, src1, src2);
    else {
        vmulps(scratch_, src1, src2);
        vaddps(dst, dst, scratch_);
    }
} */

// Inner loop.
void jit_lasx_gemv_t_f32_kern::innerloop(int unroll_m, int unroll_n) {
    if ((unroll_m > M_UNROLL_) || (unroll_n > N_UNROLL_) || (unroll_m < 0)
            || (unroll_n < 0))
        return;

    int um_vecs = (unroll_m + 7) >> 3;

    // Load x.
    for (int i = 0; i < um_vecs; i++) {
        //auto x_mem = ptr[XO_ + size_ * (8 * i - offset_x_)];
        //v_load(x_regs_[i], x_mem, unroll_m);
        load_bytes(x_regs_[i], XO_, size_ * (8 * i - offset_x_), (unroll_m > 8 ? 8 : unroll_m) * size_);
    }
    //add(XO_, size_ * unroll_m);
    add_imm(XO_, XO_, size_ * unroll_m, X_TMP_0);

    //Reg64 LDA3 = rax;
    //lea(LDA3, ptr[LDA_ + LDA_ * 2]);
    slli_d(t2, LDA_, 1);
    add_d(t3, t2, LDA_);

    // Load A
    for (int j = 0; j < unroll_n; j++) {
        for (int i = 0; i < um_vecs; i++) {
            //Ymm a = a_regs_[i][j];
            XVReg a = a_regs_[i][j];

            //decltype(LDA_ * j) lda_mult = (j == 3) ? LDA3 : LDA_ * j;

            //auto a_mem = ptr[AO_ + lda_mult + size_ * (8 * i - offset_a_)];
            if (j > 0)
                add_d(X_TMP_1, AO_, (j == 1 ? LDA_ : (j == 2 ? t2 : t3)));
            //v_load(a, a_mem, unroll_m);
            load_bytes(a, X_TMP_1, size_ * (8 * i - offset_a_), (unroll_m > 8 ? 8 : unroll_m) * size_);
        }
    }

    //lea(AO_, ptr[AO_ + size_ * unroll_m]);
    add_imm(AO_, AO_, size_ * unroll_m, X_TMP_0);

    for (int j = 0; j < unroll_n; j++) {
        //Ymm acc = acc_[j];
        XVReg acc = acc_[j];

        for (int i = 0; i < um_vecs; i++) {
            //dot_product(acc, x_regs_[i], a_regs_[i][j]);
            xvfmadd_s(acc, x_regs_[i], a_regs_[i][j], acc);
        }
    }
}

// Outer loop.
void jit_lasx_gemv_t_f32_kern::outerloop(
        int unroll_x, int unroll_y, Label *&cur_outerloop_label) {
    if ((unroll_x > M_UNROLL_) || (unroll_y > N_UNROLL_) || (unroll_y < 0)
            || (unroll_x < 0))
        return;

    Label label_m_loop, label_n_loop, label_m_remainder_loops[5];

    L(*cur_outerloop_label);
    cur_outerloop_label++;
    if (unroll_y >= N_UNROLL_) {
        //mov(I_, N_);
        add_d(I_, N_, zero);
        //cmp(I_, unroll_y);
        mov_imm(X_TMP_0, unroll_y);
        //jl(*cur_outerloop_label, T_NEAR); // Jump to next outerloop label.
        blt(I_, X_TMP_0, *cur_outerloop_label);
    } else {
        //test(I_, unroll_y);
        andi(X_TMP_0, I_, unroll_y);
        //jle(*cur_outerloop_label, T_NEAR);
        bge(zero, X_TMP_0, *cur_outerloop_label);
    }

    L_aligned(label_n_loop);
    {

        //mov(YO_, Y_);
        add_d(YO_, Y_, zero);
        //lea(Y_, ptr[YO_ + INCY_ * unroll_y]);
        mov_imm(X_TMP_0, unroll_y);
        mul_d(X_TMP_1, INCY_, X_TMP_0);
        add_d(Y_, YO_, X_TMP_1);

        //mov(AO_, A_);
        add_d(AO_, A_, zero);
        //lea(A_, ptr[AO_ + LDA_ * unroll_y]);
        mul_d(X_TMP_1, LDA_, X_TMP_0);
        add_d(A_, AO_, X_TMP_1);

        //mov(XO_, X_);
        add_d(XO_, X_, zero);

        for (int i = 0; i < unroll_y; i++) {
            auto acc = acc_[i];
            //vxorps(acc, acc, acc);
            xvxor_v(acc, acc, acc);
        }

        //mov(J_, M_);
        add_d(J_, M_, zero);
        //cmp(J_, unroll_x);
        mov_imm(X_TMP_0, unroll_x);
        //jl(label_m_remainder_loops[0], T_NEAR);
        blt(J_, X_TMP_0, label_m_remainder_loops[0]);

        L_aligned(label_m_loop);
        {
            innerloop(unroll_x, unroll_y);
            //sub(J_, unroll_x);
            add_imm(J_, J_, -1 * unroll_x, X_TMP_0);
            //cmp(J_, unroll_x);
            mov_imm(X_TMP_0, unroll_x);
            //jge(label_m_loop, T_NEAR);
            bge(J_, X_TMP_0, label_m_loop);
        }

        //align(16);

        // Update y.
        for (int j = 0; j < unroll_y; j++) {
            //Ymm acc = acc_[j];
            XVReg acc = acc_[j];

            //vhaddps(acc, acc, acc);
            xvpickev_w(xr30, acc, acc);
            xvpickod_w(xr31, acc, acc);
            xvfadd_s(acc, xr31, xr30);
            //vperm2f128(scratch_, acc, acc, 0x1);
            xvpermi_q(scratch_, acc, 0x1);
            //vaddps(acc, acc, scratch_);
            xvfadd_s(acc, acc, scratch_);
            //vhaddps(acc, acc, acc);
            xvpickev_w(xr30, acc, acc);
            xvpickod_w(xr31, acc, acc);
            xvfadd_s(acc, xr31, xr30);
        }
        for (int j = 0; j < unroll_y; j++) {
            // TODO Handle negative increments
            XVReg y = y_regs_[j];
            XVReg acc = acc_[j];

            //imul(YO2_, INCY_, j);
            mov_imm(X_TMP_0, j);
            mul_d(YO2_, INCY_, X_TMP_0);
            //lea(YO2_, ptr[YO_ + YO2_]);
            add_d(YO2_, YO_, YO2_);
            //auto y_mem = ptr[YO2_];

            //v_load(y, y_mem, 1);
            load_bytes(y, YO2_, 0, 4);

            //if (is_avx2_) {
            //    vfmadd231ss(make_xmm(y), make_xmm(alpha_), make_xmm(acc));
            //} else {
            //    vmulps(make_xmm(scratch_), make_xmm(alpha_), make_xmm(acc));
            //    vaddps(make_xmm(y), make_xmm(y), make_xmm(scratch_));
            //}
            // x86 only calc the 4 byte we calc 16 byte(we only store 4 byte)
            vfmadd_s(make_xmm(y), make_xmm(alpha_), make_xmm(acc), make_xmm(y));

            //v_store(y_mem, y, 1);
            store_bytes(y, YO2_, 0, 4);
        }

        int label_idx = 0;
        for (int ux = 8; ux > 0; ux >>= 1) {
            L(label_m_remainder_loops[label_idx++]);
            if (unroll_x > ux) {
                //test(J_, ux);
                andi(X_TMP_0, J_, ux);
                //jle(label_m_remainder_loops[label_idx], T_NEAR);
                bge(zero, X_TMP_0, label_m_remainder_loops[label_idx]);

                for (int i = 0; i < unroll_y; i++) {
                    auto acc = acc_[i];
                    //vxorps(acc, acc, acc);
                    xvxor_v(acc, acc, acc);
                }

                innerloop(ux, unroll_y);

                //align(16);

                // Update y.
                for (int j = 0; j < unroll_y; j++) {
                    XVReg acc = acc_[j];

                    //vhaddps(acc, acc, acc);
                    xvpickev_w(xr30, acc, acc);
                    xvpickod_w(xr31, acc, acc);
                    xvfadd_s(acc, xr31, xr30);
                    //vperm2f128(scratch_, acc, acc, 0x1);
                    xvpermi_q(scratch_, acc, 0x1);
                    //vaddps(acc, acc, scratch_);
                    xvfadd_s(acc, acc, scratch_);
                    //vhaddps(acc, acc, acc);
                    xvpickev_w(xr30, acc, acc);
                    xvpickod_w(xr31, acc, acc);
                    xvfadd_s(acc, xr31, xr30);
                }
                for (int j = 0; j < unroll_y; j++) {
                    // TODO Handle negative increments
                    XVReg y = y_regs_[j];
                    XVReg acc = acc_[j];

                    //imul(YO2_, INCY_, j);
                    mov_imm(X_TMP_0, j);
                    mul_d(YO2_, INCY_, X_TMP_0);
                    //lea(YO2_, ptr[YO_ + YO2_]);
                    add_d(YO2_, YO_, YO2_);
                    //auto y_mem = ptr[YO2_];

                    //v_load(y, y_mem, 1);
                    load_bytes(y, YO2_, 0, 4);

                    //if (is_avx2_) {
                    //    vfmadd231ss(
                    //            make_xmm(y), make_xmm(alpha_), make_xmm(acc));
                    //} else {
                    //    vmulps(make_xmm(scratch_), make_xmm(alpha_),
                    //            make_xmm(acc));
                    //    vaddps(make_xmm(y), make_xmm(y), make_xmm(scratch_));
                    //}
                    // x86 only calc the 4 byte we calc 16 byte(we only store 4 byte)
                    vfmadd_s(make_xmm(y), make_xmm(alpha_), make_xmm(acc), make_xmm(y));

                    //v_store(y_mem, y, 1);
                    store_bytes(y, YO2_, 0, 4);
                }
            }
        }
        L(label_m_remainder_loops[label_idx]);

        if (unroll_y >= N_UNROLL_) {
            //sub(I_, unroll_y);
            add_imm(I_, I_, -1 * unroll_y, X_TMP_0);
            //cmp(I_, unroll_y);
            mov_imm(X_TMP_0, unroll_y);
            //jge(label_n_loop);
            bge(I_, X_TMP_0, label_n_loop);
        }
    }

    //align(16);
}

void jit_lasx_gemv_t_f32_kern::generate() {
    // Prologue
    preamble();

    //movss(make_xmm(alpha_), qword[ALPHA_]);
    vldrepl_d(make_xmm(alpha_), ALPHA_, 0);

    //if (is_windows) {
    //    mov(LDA_, arg_lda_);
    //    mov(X_, arg_x_);
    //}

    //mov(Y_, arg_y_); // Y_is abi_param8
    //mov(INCY_, arg_incy_);
    ld_d(INCY_, sp, get_size_of_abi_save_regs()); // INCY_ is 9 param in sp

    //sub(A_, -offset_a_ * size_);
    add_imm(A_, A_, offset_a_ * size_, X_TMP_0);
    //sub(X_, -offset_x_ * size_);
    add_imm(X_, X_, offset_x_ * size_, X_TMP_0);

    //mov(M_, qword[M_]);
    ld_d(M_, M_, 0);
    //mov(N_, qword[N_]);
    ld_d(N_, N_, 0);
    //mov(LDA_, qword[LDA_]);
    ld_d(LDA_, LDA_, 0);
    //mov(INCY_, qword[INCY_]);
    ld_d(INCY_, INCY_, 0);

    //lea(LDA_, ptr[LDA_ * size_]);
    slli_d(LDA_, LDA_, 2);
    //lea(INCY_, ptr[INCY_ * size_]);
    slli_d(INCY_, INCY_, 2);

    Label outerloop_labels[4];
    Label *cur_outerloop_label = &outerloop_labels[0];

    // Main n loop.
    outerloop(M_UNROLL_, N_UNROLL_, cur_outerloop_label);

    // n remainder loops.
    for (int un = 2; un > 0; un >>= 1)
        if (N_UNROLL_ > un) outerloop(M_UNROLL_, un, cur_outerloop_label);

    L(*cur_outerloop_label);

    // Epilogue.
    postamble();
}

// Function signature: gemv(*m, *n, *alpha, *a, *lda, *x, *incx, *y, *incy)
jit_lasx_gemv_t_f32_kern::jit_lasx_gemv_t_f32_kern()
    : jit_generator(nullptr, 100000) {
    //, arg_lda_(0)
    //, arg_x_(0)
    //, arg_incx_(0)
    //, arg_y_(0)
    //, arg_incy_(0) {
    //is_avx2_ = mayiuse(avx2);

    // this already inited in hpp
    //M_ = abi_param1;
    //N_ = abi_param2;
    //ALPHA_ = abi_param3;
    //A_ = abi_param4;
    //LDA_ = is_windows ? rdi : r8;
    //X_ = is_windows ? rsi : r9;
    //INCY_ = r10;
    //Y_ = r11;

    //J_ = r12;
    //I_ = r13;

    //AO_ = r14;
    //XO_ = r15;

    //YO_ = rbx;
    //YO2_ = rbp;

    // Assign vector registers
    //for (int i = 0; i < (N_UNROLL_); i++)
    //    y_regs_[i] = Ymm(i);

    //int rn = 0;
    //for (int i = 0; i < (M_UNROLL_ >> 3); i++)
    //    for (int j = 0; j < N_UNROLL_; j++)
    //        a_regs_[i][j] = Ymm(rn++);

    //x_regs_[0] = ymm8;
    //x_regs_[1] = ymm9;

    //alpha_ = ymm10;
    //scratch_ = ymm11;

    //for (int i = 0; i < (N_UNROLL_); i++)
    //    acc_[i] = Ymm(12 + i);

    // Assign stack variables.
    //auto args_offset = get_size_of_abi_save_regs() + 8 + (is_windows ? 48 : 0);

    //arg_lda_ = ptr[rsp + (args_offset - 16)];
    //arg_x_ = ptr[rsp + (args_offset - 8)];
    //arg_incx_ = ptr[rsp + (args_offset + 0)]; // Assumed 1 for A transpose.
    //arg_y_ = ptr[rsp + (args_offset + 8)];
    //arg_incy_ = ptr[rsp + (args_offset + 16)]; // Assumed 1 for A non-transpose.
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
