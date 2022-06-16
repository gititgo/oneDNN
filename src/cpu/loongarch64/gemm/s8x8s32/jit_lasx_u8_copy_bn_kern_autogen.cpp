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

#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/gemm/s8x8s32/common_u8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

jit_lasx_u8_copy_bn_kern::jit_lasx_u8_copy_bn_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_u8_copy_bn_kern::generate() {

#ifndef _WIN32
#define M a0//rdi
#define N a1//rsi
#define A a2//rdx
#define LDA a3//rcx
#define ALPHA a4//r8
#define B a5//r9

#define I t4//rax
#define A1 t5//r10
#define A2 a4//r8
#define LDA3 t6//r11
#define TM t7
#define TM0 t8


#else

//#define M rcx
//#define N rdx
//#define A r8
//#define LDA r9
//#define ALPHA rax
//#define B rdi

//#define I rax
//#define A1 rsi
//#define A2 r10
//#define LDA3 r11

//#define ARG_ALPHA 40 + stacksize + rsp
//#define ARG_B 48 + stacksize + rsp

#endif

    inLocalLabel();
    {
        std::vector<Xbyak_loongarch::Label> labels(24);

        preamble();
#ifdef _WIN32
        //auto stacksize = get_size_of_abi_save_regs();
        //mov(ALPHA, ptr[ARG_ALPHA]);
        //mov(B, ptr[ARG_B]);
#endif

        //mov(N, qword[N]);
        ld_d(N, N, 0);
        //mov(M, qword[M]);
        ld_d(M, M, 0);
        //mov(LDA, qword[LDA]);
        ld_d(LDA, LDA, 0);
        //sub(A, -128);
        addi_d(A, A, 128);
        //sub(B, -128);
        addi_d(B, B, 128);
        //lea(LDA3, ptr[LDA + LDA * 2]);
        add_d(LDA3, LDA, LDA);
        add_d(LDA3, LDA3, LDA);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jl(labels[3], T_NEAR);
        blt(N, TM, labels[3]);
        //align(4);

        L(labels[6]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(TM, A1, LDA);
        add_d(A2, TM, LDA);
        //lea(I, ptr[A1 + LDA * 4]);
        add_d(I, TM, LDA3);
        //mov(A, I);
        add_d(A, I, zero);
        //mov(I, M);
        //sar(I, 0x4);
        srai_d(I, M, 0x4);
        //jle(labels[22], T_NEAR);
        bge(zero, I, labels[22]);
        //align(4);

        L(labels[19]);
        //movdqu(xmm0, xword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        add_d(TM, A1, LDA);
        vld(vr1, TM, -0x80);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //movdqu(xmm2, xword[A2 - 0x80]);
        vld(vr2, A2, -0x80);
        //movdqu(xmm3, xword[A2 + LDA * 1 - 0x80]);
        add_d(TM, A2, LDA);
        vld(vr3, TM, -0x80);
        //sub(A2, -16);
        addi_d(A2, A2, 16);
        //movdqa(xmm4, xmm0);
        vbsll_v(vr4, vr0, 0);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //punpckhdq(xmm4, xmm1);
        vilvh_w(vr4, vr1, vr4);
        //movdqa(xmm5, xmm2);
        vbsll_v(vr5, vr2, 0);
        //punpckldq(xmm2, xmm3);
        vilvl_w(vr2, vr3, vr2);
        //punpckhdq(xmm5, xmm3);
        vilvh_w(vr5, vr3, vr5);
        //movdqa(xmm1, xmm0);
        vbsll_v(vr1, vr0, 0);
        //punpcklqdq(xmm0, xmm2);
        vilvl_d(vr0, vr2, vr0);
        //punpckhqdq(xmm1, xmm2);
        vilvh_d(vr1, vr2, vr1);
        //movdqa(xmm3, xmm4);
        vbsll_v(vr3, vr4, 0);
        //punpcklqdq(xmm4, xmm5);
        vilvl_d(vr4, vr5, vr4);
        //punpckhqdq(xmm3, xmm5);
        vilvh_d(vr3, vr5, vr3);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //movdqu(xword[B - 0x70], xmm1);
        vst(vr1, B, -0x70);
        //movdqu(xword[B - 0x60], xmm4);
        vst(vr4, B, -0x60);
        //movdqu(xword[B - 0x50], xmm3);
        vst(vr3, B, -0x50);
        //sub(B, -64);
        addi_d(B, B, 64);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[19], T_NEAR);
        blt(zero, I, labels[19]);
        //align(4);

        L(labels[22]);
        //test(M, 0x8);
        andi(TM, M, 0x8);
        //jle(labels[23], T_NEAR);
        bge(zero, TM, labels[23]);
        //movq(xmm0, qword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        add_d(TM, A1, LDA);
        vld(vr1, TM, -0x80);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //movq(xmm2, qword[A2 - 0x80]);
        vld(vr2, A2, -0x80);
        //movq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        add_d(TM, A2, LDA);
        vld(vr3, TM, -0x80);
        //sub(A2, -8);
        addi_d(A2, A2, 8);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //punpckldq(xmm2, xmm3);
        vilvl_w(vr2, vr3, vr2);
        //movdqa(xmm1, xmm0);
        vbsll_v(vr1, vr0, 0);
        //punpcklqdq(xmm0, xmm2);
        vilvl_d(vr0, vr2, vr0);
        //punpckhqdq(xmm1, xmm2);
        vilvh_d(vr1, vr2, vr1);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //movdqu(xword[B - 0x70], xmm1);
        vst(vr1, B, -0x70);
        //sub(B, -32);
        addi_d(B, B, 32);
        //align(4);

        L(labels[23]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[0], T_NEAR);
        bge(zero, TM, labels[0]);
        //movd(xmm0, dword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        add_d(TM, A1, LDA);
        vld(vr1, TM, -0x80);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //movd(xmm2, dword[A2 - 0x80]);
        vld(vr2, A2, -0x80);
        //movd(xmm3, dword[A2 + LDA * 1 - 0x80]);
        add_d(TM, A2, LDA);
        vld(vr3, TM, -0x80);
        //sub(A2, -4);
        addi_d(A2, A2, 4);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //punpckldq(xmm2, xmm3);
        vilvl_w(vr2, vr3, vr2);
        //punpcklqdq(xmm0, xmm2);
        vilvl_d(vr0, vr2, vr0);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[0]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[1], T_NEAR);
        bge(zero, TM, labels[1]);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM0, 0x0);
        //mov(ax, word[A1 + LDA * 1 - 0x80]);
        add_d(TM, A1, LDA);
        ld_h(TM0, TM, -0x80);
        //sub(A1, -2);
        addi_d(A1, A1, 2);
        //pinsrw(xmm0, eax, 0x1);
        vinsgr2vr_h(vr0, TM0, 0x1);
        //mov(ax, word[A2 - 0x80]);
        ld_h(TM0, A2, -0x80);
        //pinsrw(xmm0, eax, 0x2);
        vinsgr2vr_h(vr0, TM0, 0x2);
        //mov(ax, word[A2 + LDA * 1 - 0x80]);
        add_d(TM, A2, LDA);
        ld_h(TM0, TM, -0x80);
        //sub(A2, -2);
        addi_d(A2, A2, 2);
        //pinsrw(xmm0, eax, 0x3);
        vinsgr2vr_h(vr0, TM0, 0x3);
        //movq(qword[B - 0x80], xmm0);
        vstelm_d(vr0, B, -0x80, 0);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[1]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[2], T_NEAR);
        bge(zero, TM, labels[2]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //pinsrb(xmm0, eax, 0x0);
        vinsgr2vr_b(vr0, TM0, 0x0);
        //mov(al, byte[A1 + LDA * 1 - 0x80]);
        add_d(TM, A1, LDA);
        ld_b(TM0, TM, -0x80);
        //pinsrb(xmm0, eax, 0x1);
        vinsgr2vr_b(vr0, TM0, 0x1);
        //mov(al, byte[A2 - 0x80]);
        ld_b(TM0, A2, -0x80);
        //pinsrb(xmm0, eax, 0x2);
        vinsgr2vr_b(vr0, TM0, 0x2);
        //mov(al, byte[A2 + LDA * 1 - 0x80]);
        add_d(TM, A2, LDA);
        ld_b(TM0, TM, -0x80);
        //pinsrb(xmm0, eax, 0x3);
        vinsgr2vr_b(vr0, TM0, 0x3);
        //movd(dword[B - 0x80], xmm0);
        vstelm_w(vr0, B, -0x80, 0);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[2]);
        //sub(N, 0x4);
        addi_d(N, N, -0x4);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jge(labels[6], T_NEAR);
        bge(N, TM, labels[6]);
        //align(4);

        L(labels[3]);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jl(labels[12], T_NEAR);
        blt(N, TM, labels[12]);
        //align(4);

        L(labels[4]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //lea(I, ptr[A1 + LDA * 2]);
        add_d(I, A2, LDA);
        //mov(A, I);
        add_d(A, I, zero);
        //mov(I, M);
        //sar(I, 0x4);
        srai_d(I, M, 0x4);
        //jle(labels[7], T_NEAR);
        bge(zero, I, labels[7]);
        //align(4);

        L(labels[5]);
        //movdqu(xmm0, xword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //movdqu(xmm1, xword[A2 - 0x80]);
        vld(vr1, A2, -0x80);
        //sub(A2, -16);
        addi_d(A2, A2, 16);
        //movdqa(xmm2, xmm0);
        vbsll_v(vr2, vr0, 0);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //punpckhdq(xmm2, xmm1);
        vilvh_w(vr2, vr1, vr2);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //movdqu(xword[B - 0x70], xmm2);
        vst(vr2, B, -0x70);
        //sub(B, -32);
        addi_d(B, B, 32);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[5], T_NEAR);
        blt(zero, I, labels[5]);
        //align(4);

        L(labels[7]);
        //test(M, 0x8);
        andi(TM, M, 0x8);
        //jle(labels[8], T_NEAR);
        bge(zero, TM, labels[8]);
        //movq(xmm0, qword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //movq(xmm1, qword[A2 - 0x80]);
        vld(vr1, A2, -0x80);
        //sub(A2, -8);
        addi_d(A2, A2, 8);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[8]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[9], T_NEAR);
        bge(zero, TM, labels[9]);
        //movd(xmm0, dword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //movd(xmm1, dword[A2 - 0x80]);
        vld(vr1, A2, -0x80);
        //sub(A2, -4);
        addi_d(A2, A2, 4);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //movq(qword[B - 0x80], xmm0);
        vstelm_d(vr0, B, -0x80, 0);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[9]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[10], T_NEAR);
        bge(zero, TM, labels[10]);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //sub(A1, -2);
        addi_d(A1, A1, 2);
        //pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM0, 0x0);
        //mov(ax, word[A2 - 0x80]);
        ld_h(TM0, A2, -0x80);
        //sub(A2, -2);
        addi_d(A2, A2, 2);
        //pinsrw(xmm0, eax, 0x1);
        vinsgr2vr_h(vr0, TM0, 0x1);
        //movd(dword[B - 0x80], xmm0);
        vstelm_w(vr0, B, -0x80, 0);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[10]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[11], T_NEAR);
        bge(zero, TM, labels[11]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //mov(byte[B - 0x80], al);
        st_b(TM0, B, -0x80);
        //mov(al, byte[A2 - 0x80]);
        ld_b(TM0, A2, -0x80);
        //mov(byte[B - 0x7f], al);
        st_b(TM0, B, -0x7f);
        //sub(B, -2);
        addi_d(B, B, 2);
        //align(4);

        L(labels[11]);
        //sub(N, 0x2);
        addi_d(N, N, -0x2);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jge(labels[4], T_NEAR);
        bge(N, TM, labels[4]);
        //align(4);

        L(labels[12]);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jl(labels[21], T_NEAR);
        blt(N, TM, labels[21]);
        //align(4);

        L(labels[13]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //add(A, LDA);
        add_d(A, A, LDA);
        //mov(I, M);
        //sar(I, 0x4);
        srai_d(I, M, 0x4);
        //jle(labels[15], T_NEAR);
        bge(zero, I, labels[15]);
        //align(4);

        L(labels[14]);
        //movdqu(xmm0, xword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[14], T_NEAR);
        blt(zero, I, labels[14]);
        //align(4);

        L(labels[15]);
        //test(M, 0x8);
        andi(TM, M, 0x8);
        //jle(labels[16], T_NEAR);
        bge(zero, TM, labels[16]);
        //movq(xmm0, qword[A1 - 0x80]);
        ld_d(TM, A1, -0x80);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //movq(qword[B - 0x80], xmm0);
        st_d(TM, B, -0x80);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[16]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[17], T_NEAR);
        bge(zero, TM, labels[17]);
        //movd(xmm0, dword[A1 - 0x80]);
        ld_w(TM, A1, -0x80);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //movd(dword[B - 0x80], xmm0);
        st_w(TM, B, -0x80);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[17]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[18], T_NEAR);
        bge(zero, TM, labels[18]);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, -0x80);
        //mov(word[B - 0x80], ax);
        st_h(TM, B, -0x80);
        //sub(A1, -2);
        addi_d(A1, A1, 2);
        //sub(B, -2);
        addi_d(B, B, 2);
        //align(4);

        L(labels[18]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[20], T_NEAR);
        bge(zero, TM, labels[20]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, -0x80);
        //mov(byte[B - 0x80], al);
        st_b(TM, B, -0x80);
        //sub(B, -1);
        addi_d(B, B, 1);
        //align(4);

        L(labels[20]);
        //sub(N, 0x1);
        addi_d(N, N, -0x1);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jge(labels[13], T_NEAR);
        bge(N, TM, labels[13]);
        //align(4);

        L(labels[21]);

        postamble();
    }
    outLocalLabel();

#undef M
#undef N
#undef A
#undef LDA
#undef ALPHA
#undef B
#undef I
#undef A1
#undef A2
#undef LDA3
#ifdef _WIN32
#undef ARG_ALPHA
#undef ARG_B
#endif
}

} //namespace loongarch64
} //namespace cpu
} //namespace impl
} //namespace dnnl
