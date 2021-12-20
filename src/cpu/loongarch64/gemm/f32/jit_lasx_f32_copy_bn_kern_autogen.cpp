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

#include "cpu/loongarch64/gemm/f32/common_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

jit_lasx_f32_copy_bn_kern::jit_lasx_f32_copy_bn_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_f32_copy_bn_kern::generate() {

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
#define TM s1
#define TM0 s0
#define LDA2 t7
#define LDA4 t8

#else
//#define M rcx
//#define N rdx
//#define A r8
//#define LDA r9
//#define ALPHA rsi
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
        std::vector<Xbyak_loongarch::Label> labels(50);

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
        //sub(A, 0x0);
        //sub(B, -128);
        addi_d(B, B, 128);
        //shl(LDA, 0x2);
        slli_d(LDA, LDA, 0x2);
        //lea(LDA3, ptr[LDA + LDA * 2]);
        add_d(LDA2, LDA, LDA);
        add_d(LDA3, LDA2, LDA);
        add_d(LDA4, LDA3, LDA);
        //vbroadcastss(xr6, dword[ALPHA]);
        xvldrepl_w(xr6, ALPHA, 0);
        //vpcmpeqb(vr3, vr3, vr3);
        vseq_b(vr3, vr3, vr3);
        //vpsrld(vr3, vr3, 0x17);
        vsrli_w(vr3, vr3, 0x17);
        //vpslld(vr3, vr3, 0x19);
        vslli_w(vr3, vr3, 0x19);
        //vpsrld(vr3, vr3, 0x2);
        vsrli_w(vr3, vr3, 0x2);
        //vpcmpeqb(vr4, vr4, vr4);
        vseq_b(vr4, vr4, vr4);
        //vpslld(vr4, vr4, 0x1f);
        vslli_w(vr4, vr4, 0x1f);
        //vperm2f128(xr4, xr4, xr4, 0x20);
        xvpermi_q(xr4, xr4, 0x20);
        //vucomiss(vr6, vr3);
        vfcmp_cne_s(vr31, vr3, vr6);
        vpickve2gr_w(TM, vr31, 0);
        //jne(labels[36], T_NEAR);
        bnez(TM, labels[36]);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jl(labels[47], T_NEAR);
        blt(N, TM, labels[47]);
        //align(4);

        L(labels[23]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x4);
        //add(A, I);
        add_d(A, A, LDA4);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[0], T_NEAR);
        bge(zero, I, labels[0]);
        //align(4);

        L(labels[14]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vmovups(vr1, xword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vmovups(vr2, xword[A1 + LDA * 2]);
        vldx(vr2, A1, LDA2);
        //vmovups(vr3, xword[A1 + LDA3 * 1]);
        vldx(vr3, A1, LDA3);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr5, vr0, vr1);
        vilvh_w(vr5, vr1, vr0);
        //vunpcklps(vr1, vr2, vr3);
        vilvl_w(vr1, vr3, vr2);
        //vunpckhps(vr3, vr2, vr3);
        vilvh_w(vr3, vr3, vr2);
        //vunpcklpd(vr0, vr4, vr1);
        vilvl_d(vr0, vr1, vr4);
        //vunpckhpd(vr1, vr4, vr1);
        vilvh_d(vr1, vr1, vr4);
        //vunpcklpd(vr2, vr5, vr3);
        vilvl_d(vr2, vr3, vr5);
        //vunpckhpd(vr3, vr5, vr3);
        vilvh_d(vr3, vr3, vr5);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //vmovups(xword[B - 0x70], vr1);
        vst(vr1, B, -0x70);
        //vmovups(xword[B - 0x60], vr2);
        vst(vr2, B, -0x60);
        //vmovups(xword[B - 0x50], vr3);
        vst(vr3, B, -0x50);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -64);
        addi_d(B, B, 64);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[14], T_NEAR);
        blt(zero, I, labels[14]);
        //align(4);

        L(labels[0]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[49], T_NEAR);
        bge(zero, TM, labels[49]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vmovsd(vr1, qword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vmovhps(vr0, vr0, qword[A1 + LDA * 2]);
        ldx_d(TM0, A1, LDA2);
        vinsgr2vr_d(vr0, TM0, 1);
        //vmovhps(vr1, vr1, qword[A1 + LDA3 * 1]);
        ldx_d(TM0, A1, LDA3);
        vinsgr2vr_d(vr1, TM0, 1);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr1, vr0, vr1);
        vilvh_w(vr1, vr1, vr0);
        //vunpcklpd(vr0, vr4, vr1);
        vilvl_d(vr0, vr1, vr4);
        //vunpckhpd(vr1, vr4, vr1);
        vilvh_d(vr1, vr1, vr4);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //vmovups(xword[B - 0x70], vr1);
        vst(vr1, B, -0x70);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -32);
        addi_d(B, B, 32);
        //align(4);

        L(labels[49]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[48], T_NEAR);
        bge(zero, TM, labels[48]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vmovss(vr1, dword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vmovss(vr2, dword[A1 + LDA * 2]);
        vldx(vr2, A1, LDA2);
        //vmovss(vr3, dword[A1 + LDA3 * 1]);
        vldx(vr3, A1, LDA3);
        //vunpcklps(vr2, vr2, vr3);
        vilvl_w(vr2, vr3, vr2);
        //vunpcklpd(vr0, vr0, vr2);
        vilvl_d(vr0, vr2, vr0);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[48]);
        //sub(N, 0x4);
        addi_d(N, N, -0x4);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jge(labels[23], T_NEAR);
        bge(N, TM, labels[23]);
        //align(4);

        L(labels[47]);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jl(labels[42], T_NEAR);
        blt(N, TM, labels[42]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x2);
        //add(A, I);
        add_d(A, A, LDA2);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[45], T_NEAR);
        bge(zero, I, labels[45]);
        //align(4);

        L(labels[46]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vmovups(vr1, xword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr1, vr0, vr1);
        vilvh_w(vr1, vr1, vr0);
        //vmovaps(vr0, vr4);
        vbsll_v(vr0, vr4, 0);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //vmovhps(qword[B - 0x78], vr0);
        vstelm_d(vr0, B, -0x78, 1);
        //vmovlps(qword[B - 0x70], vr1);
        vstelm_d(vr1, B, -0x70, 0);
        //vmovhps(qword[B - 0x68], vr1);
        vstelm_d(vr1, B, -0x68, 1);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -32);
        addi_d(B, B, 32);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[46], T_NEAR);
        blt(zero, I, labels[46]);
        //align(4);

        L(labels[45]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[44], T_NEAR);
        bge(zero, TM, labels[44]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vmovsd(vr1, qword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //vmovhps(qword[B - 0x78], vr0);
        vstelm_d(vr0, B, -0x78, 1);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[44]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[43], T_NEAR);
        bge(zero, TM, labels[43]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vmovss(vr1, dword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[43]);
        //sub(N, 0x2);
        addi_d(N, N, -0x2);
        //align(4);

        L(labels[42]);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jl(labels[37], T_NEAR);
        blt(N, TM, labels[37]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x1);
        //add(A, I);
        add_d(A, A, LDA);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[40], T_NEAR);
        bge(zero, I, labels[40]);
        //align(4);

        L(labels[41]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vpshufd(vr1, vr0, 0x55);
        //vpshufd(vr2, vr0, 0xaa);
        //vpshufd(vr3, vr0, 0xff);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //vmovss(dword[B - 0x7c], vr1);
        vstelm_w(vr0, B, -0x7c, 1);
        //vmovss(dword[B - 0x78], vr2);
        vstelm_w(vr0, B, -0x78, 2);
        //vmovss(dword[B - 0x74], vr3);
        vstelm_w(vr0, B, -0x74, 3);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -16);
        addi_d(B, B, 16);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[41], T_NEAR);
        blt(zero, I, labels[41]);
        //align(4);

        L(labels[40]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[39], T_NEAR);
        bge(zero, TM, labels[39]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vpshufd(vr1, vr0, 0x55);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //vmovss(dword[B - 0x7c], vr1);
        vstelm_w(vr0, B, -0x7c, 1);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[39]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[38], T_NEAR);
        bge(zero, TM, labels[38]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[38]);
        //sub(N, 0x1);
        addi_d(N, N, -0x1);
        //align(4);

        L(labels[37]);
        //jmp(labels[1], T_NEAR);
        b(labels[1]);
        //align(4);

        L(labels[36]);
        //vxorps(vr3, vr3, vr4);
        vxor_v(vr3, vr3, vr4);
        //vucomiss(vr6, vr3);
        vfcmp_cne_s(vr31, vr6, vr3);
        vpickve2gr_w(TM, vr31, 0);
        //jne(labels[18], T_NEAR);
        bnez(TM, labels[18]);
        //vmovaps(xr6, xr4);
        xvbsll_v(xr6, xr4, 0);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jl(labels[30], T_NEAR);
        blt(N, TM, labels[30]);
        //align(4);

        L(labels[35]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x4);
        //add(A, I);
        add_d(A, A, LDA4);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[33], T_NEAR);
        bge(zero, I, labels[33]);
        //align(4);

        L(labels[34]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vmovups(vr1, xword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vmovups(vr2, xword[A1 + LDA * 2]);
        vldx(vr2, A1, LDA2);
        //vmovups(vr3, xword[A1 + LDA3 * 1]);
        vldx(vr3, A1, LDA3);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr5, vr0, vr1);
        vilvh_w(vr5, vr1, vr0);
        //vunpcklps(vr1, vr2, vr3);
        vilvl_w(vr1, vr3, vr2);
        //vunpckhps(vr3, vr2, vr3);
        vilvh_w(vr3, vr3, vr2);
        //vunpcklpd(vr0, vr4, vr1);
        vilvl_d(vr0, vr1, vr4);
        //vunpckhpd(vr1, vr4, vr1);
        vilvh_d(vr1, vr1, vr4);
        //vunpcklpd(vr2, vr5, vr3);
        vilvl_d(vr2, vr3, vr5);
        //vunpckhpd(vr3, vr5, vr3);
        vilvh_d(vr3, vr3, vr5);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vxorps(vr1, vr6, vr1);
        vxor_v(vr1, vr6, vr1);
        //vxorps(vr2, vr6, vr2);
        vxor_v(vr2, vr6, vr2);
        //vxorps(vr3, vr6, vr3);
        vxor_v(vr3, vr6, vr3);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //vmovups(xword[B - 0x70], vr1);
        vst(vr1, B, -0x70);
        //vmovups(xword[B - 0x60], vr2);
        vst(vr2, B, -0x60);
        //vmovups(xword[B - 0x50], vr3);
        vst(vr3, B, -0x50);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -64);
        addi_d(B, B, 64);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[34], T_NEAR);
        blt(zero, I, labels[34]);
        //align(4);

        L(labels[33]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[32], T_NEAR);
        bge(zero, TM, labels[32]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vmovsd(vr1, qword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vmovhps(vr0, vr0, qword[A1 + LDA * 2]);
        ldx_d(TM0, A1, LDA2);
        vinsgr2vr_d(vr0, TM0, 1);
        //vmovhps(vr1, vr1, qword[A1 + LDA3 * 1]);
        ldx_d(TM0, A1, LDA3);
        vinsgr2vr_d(vr1, TM0, 1);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr1, vr0, vr1);
        vilvh_w(vr1, vr1, vr0);
        //vunpcklpd(vr0, vr4, vr1);
        vilvl_d(vr0, vr1, vr4);
        //vunpckhpd(vr1, vr4, vr1);
        vilvh_d(vr1, vr1, vr4);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vxorps(vr1, vr6, vr1);
        vxor_v(vr1, vr6, vr1);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //vmovups(xword[B - 0x70], vr1);
        vst(vr1, B, -0x70);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -32);
        addi_d(B, B, 32);
        //align(4);

        L(labels[32]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[31], T_NEAR);
        bge(zero, TM, labels[31]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vmovss(vr1, dword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vmovss(vr2, dword[A1 + LDA * 2]);
        vldx(vr2, A1, LDA2);
        //vmovss(vr3, dword[A1 + LDA3 * 1]);
        vldx(vr3, A1, LDA3);
        //vunpcklps(vr2, vr2, vr3);
        vilvl_w(vr2, vr3, vr2);
        //vunpcklpd(vr0, vr0, vr2);
        vilvl_d(vr0, vr2, vr0);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[31]);
        //sub(N, 0x4);
        addi_d(N, N, -0x4);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jge(labels[35], T_NEAR);
        bge(N, TM, labels[35]);
        //align(4);

        L(labels[30]);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jl(labels[25], T_NEAR);
        blt(N, TM, labels[25]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x2);
        //add(A, I);
        add_d(A, A, LDA2);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[28], T_NEAR);
        bge(zero, I, labels[28]);
        //align(4);

        L(labels[29]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vmovups(vr1, xword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr1, vr0, vr1);
        vilvh_w(vr1, vr1, vr0);
        //vmovaps(vr0, vr4);
        vbsll_v(vr0, vr4, 0);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vxorps(vr1, vr6, vr1);
        vxor_v(vr1, vr6, vr1);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //vmovhps(qword[B - 0x78], vr0);
        vstelm_d(vr0, B, -0x78, 1);
        //vmovlps(qword[B - 0x70], vr1);
        vstelm_d(vr1, B, -0x70, 0);
        //vmovhps(qword[B - 0x68], vr1);
        vstelm_d(vr1, B, -0x68, 1);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -32);
        addi_d(B, B, 32);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[29], T_NEAR);
        blt(zero, I, labels[29]);
        //align(4);

        L(labels[28]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[27], T_NEAR);
        bge(zero, TM, labels[27]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vmovsd(vr1, qword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //vmovhps(qword[B - 0x78], vr0);
        vstelm_d(vr0, B, -0x78, 1);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[27]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[26], T_NEAR);
        bge(zero, TM, labels[26]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vmovss(vr1, dword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[26]);
        //sub(N, 0x2);
        addi_d(N, N, -0x2);
        //align(4);

        L(labels[25]);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jl(labels[19], T_NEAR);
        blt(N, TM, labels[19]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x1);
        //add(A, I);
        add_d(A, A, LDA);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[22], T_NEAR);
        bge(zero, I, labels[22]);
        //align(4);

        L(labels[24]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vpshufd(vr1, vr0, 0x55);
        //vpshufd(vr2, vr0, 0xaa);
        //vpshufd(vr3, vr0, 0xff);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //vmovss(dword[B - 0x7c], vr1);
        vstelm_w(vr0, B, -0x7c, 1);
        //vmovss(dword[B - 0x78], vr2);
        vstelm_w(vr0, B, -0x78, 2);
        //vmovss(dword[B - 0x74], vr3);
        vstelm_w(vr0, B, -0x74, 3);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -16);
        addi_d(B, B, 16);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[24], T_NEAR);
        blt(zero, I, labels[24]);
        //align(4);

        L(labels[22]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[21], T_NEAR);
        bge(zero, TM, labels[21]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vpshufd(vr1, vr0, 0x55);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //vmovss(dword[B - 0x7c], vr1);
        vstelm_w(vr0, B, -0x7c, 1);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[21]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[20], T_NEAR);
        bge(zero, TM, labels[20]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vxorps(vr0, vr6, vr0);
        vxor_v(vr0, vr6, vr0);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[20]);
        //sub(N, 0x1);
        addi_d(N, N, -0x1);
        //align(4);

        L(labels[19]);
        //jmp(labels[1], T_NEAR);
        b(labels[1]);
        //align(4);

        L(labels[18]);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jl(labels[11], T_NEAR);
        blt(N, TM, labels[11]);
        //align(4);

        L(labels[17]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x4);
        //add(A, I);
        add_d(A, A, LDA4);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[15], T_NEAR);
        bge(zero, I, labels[15]);
        //align(4);

        L(labels[16]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vmovups(vr1, xword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vmovups(vr2, xword[A1 + LDA * 2]);
        vldx(vr2, A1, LDA2);
        //vmovups(vr3, xword[A1 + LDA3 * 1]);
        vldx(vr3, A1, LDA3);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr5, vr0, vr1);
        vilvh_w(vr5, vr1, vr0);
        //vunpcklps(vr1, vr2, vr3);
        vilvl_w(vr1, vr3, vr2);
        //vunpckhps(vr3, vr2, vr3);
        vilvh_w(vr3, vr3, vr2);
        //vunpcklpd(vr0, vr4, vr1);
        vilvl_d(vr0, vr1, vr4);
        //vunpckhpd(vr1, vr4, vr1);
        vilvh_d(vr1, vr1, vr4);
        //vunpcklpd(vr2, vr5, vr3);
        vilvl_d(vr2, vr3, vr5);
        //vunpckhpd(vr3, vr5, vr3);
        vilvh_d(vr3, vr3, vr5);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vmulps(vr1, vr6, vr1);
        vfmul_s(vr1, vr6, vr1);
        //vmulps(vr2, vr6, vr2);
        vfmul_s(vr2, vr6, vr2);
        //vmulps(vr3, vr6, vr3);
        vfmul_s(vr3, vr6, vr3);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //vmovups(xword[B - 0x70], vr1);
        vst(vr1, B, -0x70);
        //vmovups(xword[B - 0x60], vr2);
        vst(vr2, B, -0x60);
        //vmovups(xword[B - 0x50], vr3);
        vst(vr3, B, -0x50);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -64);
        addi_d(B, B, 64);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[16], T_NEAR);
        blt(zero, I, labels[16]);
        //align(4);

        L(labels[15]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[13], T_NEAR);
        bge(zero, TM, labels[13]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vmovsd(vr1, qword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vmovhps(vr0, vr0, qword[A1 + LDA * 2]);
        ldx_d(TM0, A1, LDA2);
        vinsgr2vr_d(vr0, TM0, 1);
        //vmovhps(vr1, vr1, qword[A1 + LDA3 * 1]);
        ldx_d(TM0, A1, LDA3);
        vinsgr2vr_d(vr1, TM0, 1);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr1, vr0, vr1);
        vilvh_w(vr1, vr1, vr0);
        //vunpcklpd(vr0, vr4, vr1);
        vilvl_d(vr0, vr1, vr4);
        //vunpckhpd(vr1, vr4, vr1);
        vilvh_d(vr1, vr1, vr4);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vmulps(vr1, vr6, vr1);
        vfmul_s(vr1, vr6, vr1);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //vmovups(xword[B - 0x70], vr1);
        vst(vr1, B, -0x70);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -32);
        addi_d(B, B, 32);
        //align(4);

        L(labels[13]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[12], T_NEAR);
        bge(zero, TM, labels[12]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vmovss(vr1, dword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vmovss(vr2, dword[A1 + LDA * 2]);
        vldx(vr2, A1, LDA2);
        //vmovss(vr3, dword[A1 + LDA3 * 1]);
        vldx(vr3, A1, LDA3);
        //vunpcklps(vr2, vr2, vr3);
        vilvl_w(vr2, vr3, vr2);
        //vunpcklpd(vr0, vr0, vr2);
        vilvl_d(vr0, vr2, vr0);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vmovups(xword[B - 0x80], vr0);
        vst(vr0, B, -0x80);
        //lea(A2, ptr[A1 + LDA * 4]);
        add_d(A2, A1, LDA4);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[12]);
        //sub(N, 0x4);
        addi_d(N, N, -0x4);
        //cmp(N, 0x4);
        mov_imm(TM, 0x4);
        //jge(labels[17], T_NEAR);
        bge(N, TM, labels[17]);
        //align(4);

        L(labels[11]);
        //cmp(N, 0x2);
        mov_imm(TM, 0x2);
        //jl(labels[6], T_NEAR);
        blt(N, TM, labels[6]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x2);
        //add(A, I);
        add_d(A, A, LDA2);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[9], T_NEAR);
        bge(zero, I, labels[9]);
        //align(4);

        L(labels[10]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vmovups(vr1, xword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr4, vr0, vr1);
        vilvl_w(vr4, vr1, vr0);
        //vunpckhps(vr1, vr0, vr1);
        vilvh_w(vr1, vr1, vr0);
        //vmovaps(vr0, vr4);
        vbsll_v(vr0, vr4, 0);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vmulps(vr1, vr6, vr1);
        vfmul_s(vr1, vr6, vr1);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //vmovhps(qword[B - 0x78], vr0);
        vstelm_d(vr0, B, -0x78, 1);
        //vmovlps(qword[B - 0x70], vr1);
        vstelm_d(vr1, B, -0x70, 0);
        //vmovhps(qword[B - 0x68], vr1);
        vstelm_d(vr1, B, -0x68, 1);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -32);
        addi_d(B, B, 32);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[10], T_NEAR);
        blt(zero, I, labels[10]);
        //align(4);

        L(labels[9]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[8], T_NEAR);
        bge(zero, TM, labels[8]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vmovsd(vr1, qword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //vmovhps(qword[B - 0x78], vr0);
        vstelm_d(vr0, B, -0x78, 1);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -16);
        addi_d(B, B, 16);
        //align(4);

        L(labels[8]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[7], T_NEAR);
        bge(zero, TM, labels[7]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vmovss(vr1, dword[A1 + LDA * 1]);
        vldx(vr1, A1, LDA);
        //vunpcklps(vr0, vr0, vr1);
        vilvl_w(vr0, vr1, vr0);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vmovlps(qword[B - 0x80], vr0);
        vstelm_d(vr0, B, -0x80, 0);
        //lea(A2, ptr[A1 + LDA * 2]);
        add_d(A2, A1, LDA2);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[7]);
        //sub(N, 0x2);
        addi_d(N, N, -0x2);
        //align(4);

        L(labels[6]);
        //cmp(N, 0x1);
        mov_imm(TM, 0x1);
        //jl(labels[1], T_NEAR);
        blt(N, TM, labels[1]);
        //mov(A1, A);
        add_d(A1, A, zero);
        //mov(I, LDA);
        //imul(I, I, 0x1);
        //add(A, I);
        add_d(A, A, LDA);
        //mov(I, M);
        //sar(I, 0x2);
        srai_d(I, M, 0x2);
        //jle(labels[4], T_NEAR);
        bge(zero, I, labels[4]);
        //align(4);

        L(labels[5]);
        //vmovups(vr0, xword[A1]);
        vld(vr0, A1, 0);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vpshufd(vr1, vr0, 0x55);
        //vpshufd(vr2, vr0, 0xaa);
        //vpshufd(vr3, vr0, 0xff);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //vmovss(dword[B - 0x7c], vr1);
        vstelm_w(vr0, B, -0x7c, 1);
        //vmovss(dword[B - 0x78], vr2);
        vstelm_w(vr0, B, -0x78, 2);
        //vmovss(dword[B - 0x74], vr3);
        vstelm_w(vr0, B, -0x74, 3);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -16);
        addi_d(A1, A1, 16);
        //sub(B, -16);
        addi_d(B, B, 16);
        //dec(I);
        addi_d(I, I, -1);
        //jg(labels[5], T_NEAR);
        blt(zero, I, labels[5]);
        //align(4);

        L(labels[4]);
        //test(M, 0x2);
        andi(TM, M, 0x2);
        //jle(labels[3], T_NEAR);
        bge(zero, TM, labels[3]);
        //vmovsd(vr0, qword[A1]);
        vldrepl_d(vr0, A1, 0);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vpshufd(vr1, vr0, 0x55);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //vmovss(dword[B - 0x7c], vr1);
        vstelm_w(vr0, B, -0x7c, 1);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -8);
        addi_d(A1, A1, 8);
        //sub(B, -8);
        addi_d(B, B, 8);
        //align(4);

        L(labels[3]);
        //test(M, 0x1);
        andi(TM, M, 0x1);
        //jle(labels[2], T_NEAR);
        bge(zero, TM, labels[2]);
        //vmovss(vr0, dword[A1]);
        vldrepl_w(vr0, A1, 0);
        //vmulps(vr0, vr6, vr0);
        vfmul_s(vr0, vr6, vr0);
        //vmovss(dword[B - 0x80], vr0);
        vstelm_w(vr0, B, -0x80, 0);
        //lea(A2, ptr[A1 + LDA * 1]);
        add_d(A2, A1, LDA);
        //sub(A1, -4);
        addi_d(A1, A1, 4);
        //sub(B, -4);
        addi_d(B, B, 4);
        //align(4);

        L(labels[2]);
        //sub(N, 0x1);
        addi_d(N, N, -0x1);
        //align(4);

        L(labels[1]);

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
#undef TM
#undef TM0
#undef LDA2
#undef LDA4
#ifdef _WIN32
//#undef ARG_ALPHA
//#undef ARG_B
#endif
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
