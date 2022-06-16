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

jit_lasx_u8_copy_an_kern::jit_lasx_u8_copy_an_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_u8_copy_an_kern::generate() {

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
        std::vector<Xbyak_loongarch::Label> labels(34);
        preamble();

#ifdef _WIN32
        //auto stacksize = get_size_of_abi_save_regs();
        //mov(ALPHA, ptr[ARG_ALPHA]);
        //mov(B, ptr[ARG_B]);
#endif

        //mov(M, qword[M]);
        ld_d(M, M, 0);
        //mov(N, qword[N]);
        ld_d(N, N, 0);
        //mov(LDA, qword[LDA]);
        ld_d(LDA, LDA, 0);
        //lea(LDA3, ptr[LDA + LDA * 2]);
        add_d(LDA3, LDA, LDA);
        add_d(LDA3, LDA3, LDA);
        //sub(A, -128);
        addi_d(A, A, 128);
        //sub(B, -128);
        addi_d(B, B, 128);
        //cmp(N, 0x10);
        mov_imm(TM, 0x10);
        //jl(labels[0], T_NEAR);
        blt(N, TM, labels[0]);
        //align(4);

        L(labels[4]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //add(A, 0x10);
        addi_d(A, A, 0x10);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[31], T_NEAR);
        bge(zero, I, labels[31]);
        //align(4);

        L(labels[12]);
        //movdqu(xmm0, xword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movdqu(xmm1, xword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movdqu(xmm2, xword[A1 - 0x80]);
        vld(vr2, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movdqu(xmm3, xword[A1 - 0x80]);
        vld(vr3, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movdqa(xmm4, xmm0);
        vbsll_v(vr4, vr0, 0);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpckhbw(xmm4, xmm1);
        vilvh_b(vr4, vr1, vr4);
        //movdqa(xmm1, xmm2);
        vbsll_v(vr1, vr2, 0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //punpckhbw(xmm1, xmm3);
        vilvh_b(vr1, vr3, vr1);
        //movdqa(xmm3, xmm0);
        vbsll_v(vr3, vr0, 0);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //punpckhwd(xmm3, xmm2);
        vilvh_h(vr3, vr2, vr3);
        //movdqa(xmm2, xmm4);
        vbsll_v(vr2, vr4, 0);
        //punpcklwd(xmm4, xmm1);
        vilvl_h(vr4, vr1, vr4);
        //punpckhwd(xmm2, xmm1);
        vilvh_h(vr2, vr1, vr2);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //movdqu(xword[B - 0x70], xmm3);
        vst(vr3, B, -0x70);
        //movdqu(xword[B - 0x60], xmm4);
        vst(vr4, B, -0x60);
        //movdqu(xword[B - 0x50], xmm2);
        vst(vr2, B, -0x50);
        //sub(B, -64);
        addi_d(B, B, 64);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[12], T_NEAR);
        blt(zero, I, labels[12]);
        //align(4);

        L(labels[31]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[32], T_NEAR);
        bge(zero, TM, labels[32]);
        //movdqu(xmm0, xword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movdqu(xmm1, xword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movdqa(xmm2, xmm0);
        vbsll_v(vr2, vr0, 0);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpckhbw(xmm2, xmm1);
        vilvh_b(vr2, vr1, vr2);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //movdqu(xword[B - 0x70], xmm2);
        vst(vr2, B, -0x70);
        //sub(B, -32);
        addi_d(B, B, 32);
        //align(4);

        L(labels[32]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[33], T_NEAR);
        bge(zero, TM, labels[33]);
        //movdqu(xmm0, xword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[33]);
        //sub(N, 0x10);
        addi_d(N, N, -0x10);
        //cmp(N, 0x10);
        mov_imm(TM, 0x10);
        //jge(labels[4], T_NEAR);
        bge(N, TM, labels[4]);
        //align(4);

        L(labels[0]);
        //cmp(N, 0x8);
        mov_imm(TM, 0x8);
        //jl(labels[8], T_NEAR);
        blt(N, TM, labels[8]);
        //align(4);

        L(labels[1]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //add(A, 0x8);
        addi_d(A, A, 0x8);
        //mov(I, M);
        //sar(I, 0x3);
        srai_d(I, M, 0x3);
        //jle(labels[3], T_NEAR);
        bge(zero, I, labels[3]);
        //align(4);

        L(labels[2]);
        //movq(xmm0, qword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm1, qword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm2, qword[A1 - 0x80]);
        vld(vr2, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm3, qword[A1 - 0x80]);
        vld(vr3, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //movdqa(xmm1, xmm0);
        vbsll_v(vr1, vr0, 0);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //punpckhwd(xmm1, xmm2);
        vilvh_h(vr1, vr2, vr1);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //movdqu(xword[B - 0x70], xmm1);
        vst(vr1, B, -0x70);
        //movq(xmm0, qword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm1, qword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm2, qword[A1 - 0x80]);
        vld(vr2, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm3, qword[A1 - 0x80]);
        vld(vr3, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //movdqa(xmm1, xmm0);
        vbsll_v(vr1, vr0, 0);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //punpckhwd(xmm1, xmm2);
        vilvh_h(vr1, vr2, vr1);
        //movdqu(xword[B - 0x60], xmm0);
        vst(vr0, B, -0x60);
        //movdqu(xword[B - 0x50], xmm1);
        vst(vr1, B, -0x50);
        //sub(B, -64);
        addi_d(B, B, 64);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[2], T_NEAR);
        blt(zero, I, labels[2]);
        //align(4);

        L(labels[3]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[5], T_NEAR);
        bge(zero, TM, labels[5]);
        //movq(xmm0, qword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm1, qword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm2, qword[A1 - 0x80]);
        vld(vr2, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm3, qword[A1 - 0x80]);
        vld(vr3, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //movdqa(xmm1, xmm0);
        vbsll_v(vr1, vr0, 0);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //punpckhwd(xmm1, xmm2);
        vilvh_h(vr1, vr2, vr1);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //movdqu(xword[B - 0x70], xmm1);
        vst(vr1, B, -0x70);
        //sub(B, -32);
        addi_d(B, B, 32);
        //align(4);

        L(labels[5]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[6], T_NEAR);
        bge(zero, TM, labels[6]);
        //movq(xmm0, qword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(xmm1, qword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[6]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[7], T_NEAR);
        bge(zero, TM, labels[7]);
        //movq(xmm0, qword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movq(qword[B - 0x80], xmm0);
        vstelm_d(vr0, B, -0x80, 0);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[7]);
        //sub(N, 0x8);
        addi_d(N, N, -0x8);
        //cmp(N, 0x8);
        mov_imm(TM, 0x8);
        //jge(labels[1], T_NEAR);
        bge(N, TM, labels[1]);
        //align(4);

        L(labels[8]);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jl(labels[16], T_NEAR);
        blt(N, TM, labels[16]);
        //align(4);

        L(labels[9]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //add(A, 0x4);
        addi_d(A, A, 0x4);
        //mov(I, M);
        //sar(I, 0x3);
        srai_d(I, M, 0x3);
        //jle(labels[11], T_NEAR);
        bge(zero, I, labels[11]);
        //align(4);

        L(labels[10]);
        //movd(xmm0, dword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm1, dword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm2, dword[A1 - 0x80]);
        vld(vr2, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm3, dword[A1 - 0x80]);
        vld(vr3, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //movd(xmm0, dword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm1, dword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm2, dword[A1 - 0x80]);
        vld(vr2, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm3, dword[A1 - 0x80]);
        vld(vr3, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //movdqu(xword[B - 0x70], xmm0);
        vst(vr0, B, -0x70);
        //sub(B, -32);
        addi_d(B, B, 32);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[10], T_NEAR);
        blt(zero, I, labels[10]);
        //align(4);

        L(labels[11]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[13], T_NEAR);
        bge(zero, TM, labels[13]);
        //movd(xmm0, dword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm1, dword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm2, dword[A1 - 0x80]);
        vld(vr2, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm3, dword[A1 - 0x80]);
        vld(vr3, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[13]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[14], T_NEAR);
        bge(zero, TM, labels[14]);
        //movd(xmm0, dword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //movd(xmm1, dword[A1 - 0x80]);
        vld(vr1, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //movq(qword[B - 0x80], xmm0);
        vstelm_d(vr0, B, -0x80, 0);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[14]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[15], T_NEAR);
        bge(zero, TM, labels[15]);
        //movd(xmm0, dword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //movd(dword[B - 0x80], xmm0);
        vstelm_w(vr0, B, -0x80, 0);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[15]);
        //sub(N, 0x4);
        addi_d(N, N, -0x4);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jge(labels[9], T_NEAR);
        bge(N, TM, labels[9]);
        //align(4);

        L(labels[16]);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jl(labels[23], T_NEAR);
        blt(N, TM, labels[23]);
        //align(4);

        L(labels[17]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //add(A, 0x2);
        addi_d(A, A, 0x2);
        //mov(LDA3, M);
        //sar(LDA3, 0x3);
        srai_d(LDA3, M, 0x3);
        //jle(labels[19], T_NEAR);
        bge(zero, LDA3, labels[19]);
        //align(4);

        L(labels[18]);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm1, eax, 0x0);
        vinsgr2vr_h(vr1, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm2, eax, 0x0);
        vinsgr2vr_h(vr2, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm3, eax, 0x0);
        vinsgr2vr_h(vr3, TM0, 0x0);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm1, eax, 0x0);
        vinsgr2vr_h(vr1, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm2, eax, 0x0);
        vinsgr2vr_h(vr2, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm3, eax, 0x0);
        vinsgr2vr_h(vr3, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm4, eax, 0x0);
        vinsgr2vr_h(vr4, TM0, 0x0);
        //punpcklbw(xmm1, xmm2);
        vilvl_b(vr1, vr2, vr1);
        //punpcklbw(xmm3, xmm4);
        vilvl_b(vr3, vr4, vr3);
        //punpcklwd(xmm1, xmm3);
        vilvl_h(vr1, vr3, vr1);
        //punpcklqdq(xmm0, xmm1);
        vilvl_d(vr0, vr1, vr0);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //dec(LDA3);
        addi_d(LDA3, LDA3, -1);
        //jg(labels[18], T_NEAR);
        blt(zero, LDA3, labels[18]);
        //align(4);

        L(labels[19]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[20], T_NEAR);
        bge(zero, TM, labels[20]);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm1, eax, 0x0);
        vinsgr2vr_h(vr1, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm2, eax, 0x0);
        vinsgr2vr_h(vr2, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm3, eax, 0x0);
        vinsgr2vr_h(vr3, TM0, 0x0);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        //punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        //movq(qword[B - 0x80], xmm0);
        vstelm_d(vr0, B, -0x80, 0);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[20]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[21], T_NEAR);
        bge(zero, TM, labels[21]);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM0, 0x0);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrw(xmm1, eax, 0x0);
        vinsgr2vr_h(vr1, TM0, 0x0);
        //punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        //movd(dword[B - 0x80], xmm0);
        vstelm_w(vr0, B, -0x80, 0);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[21]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[22], T_NEAR);
        bge(zero, TM, labels[22]);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //mov(word[B - 0x80], ax);
        st_h(TM0, B, -0x80);
        //sub(B, -2);
        addi_d(B, B, 2);
        //align(4);

        L(labels[22]);
        //sub(N, 0x2);
        addi_d(N, N, -0x2);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jge(labels[17], T_NEAR);
        bge(N, TM, labels[17]);
        //align(4);

        L(labels[23]);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jl(labels[30], T_NEAR);
        blt(N, TM, labels[30]);
        //align(4);

        L(labels[24]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //add(A, 0x1);
        addi_d(A, A, 0x1);
        //mov(LDA3, M);       
        //sar(LDA3, 0x3);
        srai_d(LDA3, M, 0x3);
        //jle(labels[26], T_NEAR);
        bge(zero, LDA3, labels[26]);
        //align(4);

        L(labels[25]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x0);
        vinsgr2vr_b(vr0, TM0, 0x0);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x1);
        vinsgr2vr_b(vr0, TM0, 0x1);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x2);
        vinsgr2vr_b(vr0, TM0, 0x2);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x3);
        vinsgr2vr_b(vr0, TM0, 0x3);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x4);
        vinsgr2vr_b(vr0, TM0, 0x4);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x5);
        vinsgr2vr_b(vr0, TM0, 0x5);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x6);
        vinsgr2vr_b(vr0, TM0, 0x6);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x7);
        vinsgr2vr_b(vr0, TM0, 0x7);
        //movq(qword[B - 0x80], xmm0);
        vstelm_d(vr0, B, -0x80, 0);
        //sub(B, -8);
        addi_d(B, B, 8);
        //dec(LDA3);
        addi_d(LDA3, LDA3, -1);
        //jg(labels[25], T_NEAR);
        blt(zero, LDA3, labels[25]);
        //align(4);

        L(labels[26]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[27], T_NEAR);
        bge(zero, TM, labels[27]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x0);
        vinsgr2vr_b(vr0, TM0, 0x0);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x1);
        vinsgr2vr_b(vr0, TM0, 0x1);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x2);
        vinsgr2vr_b(vr0, TM0, 0x2);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //pinsrb(xmm0, eax, 0x3);
        vinsgr2vr_b(vr0, TM0, 0x3);
        //movd(dword[B - 0x80], xmm0);
        vstelm_w(vr0, B, -0x80, 0);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[27]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[28], T_NEAR);
        bge(zero, TM, labels[28]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //mov(byte[B - 0x80], al);
        st_b(TM0, B, -0x80);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //add(A1, LDA);
        add_d(A1, A1, LDA);
        //mov(byte[B - 0x7f], al);
        st_b(TM0, B, -0x7f);
        //sub(B, -2);
        addi_d(B, B, 2);
        //align(4);

        L(labels[28]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[29], T_NEAR);
        bge(zero, TM, labels[29]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //mov(byte[B - 0x80], al);
        st_b(TM0, B, -0x80);
        //sub(B, -1);
        addi_d(B, B, 1);
        //align(4);

        L(labels[29]);
        //sub(N, 0x1);
        addi_d(N, N, -0x1);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jge(labels[24], T_NEAR);
        bge(N, TM, labels[24]);
        //align(4);

        L(labels[30]);

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
