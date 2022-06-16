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

jit_lasx_u8_copy_sum_bn_kern::jit_lasx_u8_copy_sum_bn_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_u8_copy_sum_bn_kern::generate() {

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

//#define ARG_BIAS 24 + stacksize + rsp
#define ARG_BIAS (stacksize)

//#else

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
//#define ARG_BIAS 72 + stacksize + rsp


#endif

    inLocalLabel();
    {
        std::vector<Xbyak_loongarch::Label> labels(24);

        preamble();
        auto stacksize = get_size_of_abi_save_regs();
        
// #ifdef _WIN32
//         mov(ALPHA, ptr[ARG_ALPHA]);
//         mov(B, ptr[ARG_B]);
// #endif

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
        //jl(labels[6], T_NEAR);
        blt(N, TM, labels[6]);
        //align(4);

        L(labels[2]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA);
        add_d(A2, A2, LDA);
        //lea(I, ptr[A1 + LDA * 4]);
        add_d(I, A2, LDA);
        add_d(I, I, LDA);
        //mov(A, I);
        add_d(A, I, zero);
        //pxor(xmm7, xmm7);
        vxor_v(vr7, vr7, vr7);
        //mov(I, M);
        //sar(I, 0x4);
        srai_d(I, M, 0x4);
        //jle(labels[0], T_NEAR);
        bge(zero, I, labels[0]);
        //align(4);

        L(labels[11]);
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
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //movhlps(xmm6, xmm0);
        vextrins_d(vr6, vr0, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //pmovzxbw(xmm5, xmm1);
        vext2xv_hu_bu(xr5, xr1);
        //movhlps(xmm6, xmm1);
        vextrins_d(vr6, vr1, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x70], xmm1);
        vst(vr1, B, -0x70);
        //pmovzxbw(xmm5, xmm4);
        vext2xv_hu_bu(xr5, xr4);
        //movhlps(xmm6, xmm4);
        vextrins_d(vr6, vr4, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x60], xmm4);
        vst(vr4, B, -0x60);
        //pmovzxbw(xmm5, xmm3);
        vext2xv_hu_bu(xr5, xr3);
        //movhlps(xmm6, xmm3);
        vextrins_d(vr6, vr3, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x50], xmm3);
        vst(vr3, B, -0x50);
        //sub(B, -64);
        addi_d(B, B, 64);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[11], T_NEAR);
        blt(zero, I, labels[11]);
        //align(4);

        L(labels[0]);
        //test(M, 0x8);
        andi(TM, M, 0x8);
        //jle(labels[1], T_NEAR);
        bge(zero, TM, labels[1]);
        //movq(xmm0, qword[A1 - 0x80]);
        load_bytes(vr0, A1, -0x80, 8);
        //movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        add_d(TM, A1, LDA);
        load_bytes(vr1, TM, -0x80, 8);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //movq(xmm2, qword[A2 - 0x80]);
        load_bytes(vr2, A2, -0x80, 8);
        //movq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        add_d(TM, A2, LDA);
        load_bytes(vr3, TM, -0x80, 8);
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
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //movhlps(xmm6, xmm0);
        vextrins_d(vr6, vr0, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //pmovzxbw(xmm5, xmm1);
        vext2xv_hu_bu(xr5, xr1);
        //movhlps(xmm6, xmm1);
        vextrins_d(vr6, vr1, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x70], xmm1);
        vst(vr1, B, -0x70);
        //sub(B, -32);
        addi_d(B, B, 32);
        //align(4);

        L(labels[1]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[3], T_NEAR);
        bge(zero, TM, labels[3]);
        //movd(xmm0, dword[A1 - 0x80]);
        load_bytes(vr0, A1, -0x80, 4);
        //movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        add_d(TM, A1, LDA);
        load_bytes(vr1, TM, -0x80, 4);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //movd(xmm2, dword[A2 - 0x80]);
        load_bytes(vr2, A2, -0x80, 4);
        //movd(xmm3, dword[A2 + LDA * 1 - 0x80]);
        add_d(TM, A2, LDA);
        load_bytes(vr3, TM, -0x80, 4);
        //sub(A2, -4);
        addi_d(A2, A2, 4);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //punpckldq(xmm2, xmm3);
        vilvl_w(vr2, vr3, vr2);
        //punpcklqdq(xmm0, xmm2);
        vilvl_d(vr0, vr2, vr0);
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //movhlps(xmm6, xmm0);
        vextrins_d(vr6, vr0, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[3]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[4], T_NEAR);
        bge(zero, TM, labels[4]);
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
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movq(qword[B - 0x80], xmm0);
        store_bytes(vr0, B, -0x80, 8);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[4]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[5], T_NEAR);
        bge(zero, TM, labels[5]);
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
        //pmovzxbd(xmm5, xmm0);
        vext2xv_wu_bu(xr5, xr0);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movd(dword[B - 0x80], xmm0);
        store_bytes(vr0, B, -0x80, 4);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[5]);
        //mov(A1, qword[ARG_BIAS]);
        ld_d(A1, sp, ARG_BIAS);
        //movdqu(xword[A1], xmm7);
        vst(vr7, A1, 0);
        //add(qword[ARG_BIAS], 0x10);
        addi_d(A1, A1, 0x10);
        st_d(A1, sp, ARG_BIAS);
        //sub(N, 0x4);
        addi_d(N, N, -0x4);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jge(labels[2], T_NEAR);
        bge(N, TM, labels[2]);
        //align(4);

        L(labels[6]);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jl(labels[15], T_NEAR);
        blt(N, TM, labels[15]);
        //align(4);

        L(labels[7]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //lea(I, ptr[A1 + LDA * 2]);
        add_d(I, A2, LDA);
        //mov(A, I);
        add_d(A, I, zero);
        //pxor(xmm7, xmm7);
        vxor_v(vr7, vr7, vr7);
        //mov(I, M);
        //sar(I, 0x4);
        srai_d(I, M, 0x4);
        //jle(labels[9], T_NEAR);
        bge(zero, I, labels[9]);
        //align(4);

        L(labels[8]);
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
        //pshufd(xmm6, xmm0, 0xd8);
        vshuf4i_w(vr6, vr0, 0xd8);
        //pmovzxbw(xmm5, xmm6);
        vext2xv_hu_bu(xr5, xr6);
        //movhlps(xmm6, xmm6);
        vextrins_d(vr6, vr6, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //pshufd(xmm6, xmm2, 0xd8);
        vshuf4i_w(vr6, vr2, 0xd8);
        //pmovzxbw(xmm5, xmm6);
        vext2xv_hu_bu(xr5, xr6);
        //movhlps(xmm6, xmm6);
        vextrins_d(vr6, vr6, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x70], xmm2);
        vst(vr2, B, -0x70);
        //sub(B, -32);
        addi_d(B, B, 32);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[8], T_NEAR);
        blt(zero, I, labels[8]);
        //align(4);

        L(labels[9]);
        //test(M, 0x8);
        andi(TM, M, 0x8);
        //jle(labels[10], T_NEAR);
        bge(zero, TM, labels[10]);
        //movq(xmm0, qword[A1 - 0x80]);
        load_bytes(vr0, A1, -0x80, 8);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //movq(xmm1, qword[A2 - 0x80]);
        load_bytes(vr1, A2, -0x80, 8);
        //sub(A2, -8);
        addi_d(A2, A2, 8);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //pshufd(xmm6, xmm0, 0xd8);
        vshuf4i_w(vr6, vr0, 0xd8);
        //pmovzxbw(xmm5, xmm6);
        vext2xv_hu_bu(xr5, xr6);
        //movhlps(xmm6, xmm6);
        vextrins_d(vr6, vr6, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);
        
        L(labels[10]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[12], T_NEAR);
        bge(zero, TM, labels[12]);
        //movd(xmm0, dword[A1 - 0x80]);
        load_bytes(vr0, A1, -0x80, 4);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //movd(xmm1, dword[A2 - 0x80]);
        load_bytes(vr1, A2, -0x80, 4);
        //sub(A2, -4);
        addi_d(A2, A2, 4);
        //punpckldq(xmm0, xmm1);
        vilvl_w(vr0, vr1, vr0);
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movq(qword[B - 0x80], xmm0);
        store_bytes(vr0, B, -0x80, 8);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[12]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[13], T_NEAR);
        bge(zero, TM, labels[13]);
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
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movd(dword[B - 0x80], xmm0);
        store_bytes(vr0, B, -0x80, 4);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[13]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[14], T_NEAR);
        bge(zero, TM, labels[14]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //pinsrb(xmm0, eax, 0x0);
        vinsgr2vr_b(vr0, TM0, 0x0);
        //mov(byte[B - 0x80], al);
        st_b(TM0, B, -0x80);
        //mov(al, byte[A2 - 0x80]);
        ld_b(TM0, A2, -0x80);
        //pinsrb(xmm0, eax, 0x1);
        vinsgr2vr_b(vr0, TM0, 0x1);
        //mov(byte[B - 0x7f], al);
        st_b(TM0, B, -0x7f);
        //sub(B, -2);
        addi_d(B, B, 2);
        //pmovzxbd(xmm5, xmm0);
        vext2xv_wu_bu(xr5, xr0);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //align(4);

        L(labels[14]);
        //mov(A1, qword[ARG_BIAS]);
        ld_d(A1, sp, ARG_BIAS);
        //movq(qword[A1], xmm7);
        store_bytes(vr7, A1, 0, 8);
        //add(qword[ARG_BIAS], 0x8);
        addi_d(A1, A1, 0x8);
        st_d(A1, sp, ARG_BIAS);
        //sub(N, 0x2);
        addi_d(N, N, -0x2);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jge(labels[7], T_NEAR);
        bge(N, TM, labels[7]);
        //align(4);

        L(labels[15]);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jl(labels[23], T_NEAR);
        blt(N, TM, labels[23]);
        //align(4);

        L(labels[16]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //add(A, LDA);
        add_d(A, A, LDA);
        //pxor(xmm7, xmm7);
        vxor_v(vr7, vr7, vr7);
        //mov(I, M);       
        //sar(I, 0x4);
        srai_d(I, M, 0x4);
        //jle(labels[18], T_NEAR);
        bge(zero, I, labels[18]);
        //align(4);

        L(labels[17]);
        //movdqu(xmm0, xword[A1 - 0x80]);
        vld(vr0, A1, -0x80);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //movhlps(xmm6, xmm0);
        vextrins_d(vr6, vr0, 0x01);
        //pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, -0x80);
        //sub(B, -16);
        addi_d(B, B, 16);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[17], T_NEAR);
        blt(zero, I, labels[17]);
        //align(4);

        L(labels[18]);
        //test(M, 0x8);
        andi(TM, M, 0x8);
        //jle(labels[19], T_NEAR);
        bge(zero, TM, labels[19]);
        //movq(xmm0, qword[A1 - 0x80]);
        load_bytes(vr0, A1, -0x80, 8);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movq(qword[B - 0x80], xmm0);
        store_bytes(vr0, B, -0x80, 8);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[19]);
        //test(M, 0x4);
        andi(TM, M, 0x4);
        //jle(labels[20], T_NEAR);
        bge(zero, TM, labels[20]);
        //movd(xmm0, dword[A1 - 0x80]);
        load_bytes(vr0, A1, -0x80, 4);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //movd(dword[B - 0x80], xmm0);
        store_bytes(vr0, B, -0x80, 4);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[20]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[21], T_NEAR);
        bge(zero, TM, labels[21]);
        //mov(ax, word[A1 - 0x80]);
        ld_h(TM0, A1, -0x80);
        //pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM0, 0x0);
        //pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        //phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        //pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //mov(word[B - 0x80], ax);
        st_h(TM0, B, -0x80);
        //sub(A1, -2);
        addi_d(A1, A1, 2);
        //sub(B, -2);
        addi_d(B, B, 2); 
        //align(4);

        L(labels[21]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[22], T_NEAR);
        bge(zero, TM, labels[22]);
        //mov(al, byte[A1 - 0x80]);
        ld_b(TM0, A1, -0x80);
        //pinsrb(xmm0, eax, 0x0);
        vinsgr2vr_b(vr0, TM0, 0x0);
        //pmovzxbd(xmm5, xmm0);
        vext2xv_wu_bu(xr5, xr0);
        //paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        //mov(byte[B - 0x80], al);
        st_b(TM0, B, -0x80);
        //sub(B, -1);
        addi_d(B, B, 1);
        //align(4);
        
        L(labels[22]);
        //mov(A1, qword[ARG_BIAS]);
        ld_d(A1, sp, ARG_BIAS);
        //movd(dword[A1], xmm7);
        store_bytes(vr7, A1, 0, 4);
        //add(qword[ARG_BIAS], 0x4);
        addi_d(A1, A1, 0x4);
        st_d(A1, sp, ARG_BIAS);
        //sub(N, 0x1);
        addi_d(N, N, -0x1);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jge(labels[16], T_NEAR);
        bge(N, TM, labels[16]);
        //align(4);

        L(labels[23]);
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
#undef ARG_BIAS
}

} //namespace x64
} //namespace cpu
} //namespace impl
} //namespace dnnl
