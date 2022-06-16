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

jit_lasx_u8_copy_sum_bt_kern::jit_lasx_u8_copy_sum_bt_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_u8_copy_sum_bt_kern::generate() {

#ifndef _WIN32
#define M a0    //rdi
#define N a1    //rsi
#define A a2    //rdx
#define LDA a3  //rcx
#define ALPHA a4    //r8
#define B a5    //r9

#define I t4    //rax
#define A1 t5   //r10
#define A2 a4   //r8
#define LDA3 t6 //r11
#define TM t7

// #define ARG_BIAS 24 + stacksize + rsp
#define ARG_BIAS (stacksize)

// #else

// #define M rcx
// #define N rdx
// #define A r8
// #define LDA r9
// #define ALPHA rax
// #define B rdi

// #define I rax
// #define A1 rsi
// #define A2 r10
// #define LDA3 r11

// #define ARG_ALPHA 40 + stacksize + rsp
// #define ARG_B 48 + stacksize + rsp
// #define ARG_BIAS 72 + stacksize + rsp


#endif

    inLocalLabel();
    {
        std::vector<Xbyak_loongarch::Label> labels(21);

        preamble();
        auto stacksize = get_size_of_abi_save_regs();
//#ifdef _WIN32
        // mov(ALPHA, ptr[ARG_ALPHA]);
        // mov(B, ptr[ARG_B]);
//#endif

        // mov(M, qword[M]);
        ld_d(M, M, 0);
        // mov(N, qword[N]);
        ld_d(N, N, 0);
        // mov(LDA, qword[LDA]);
        ld_d(LDA, LDA, 0);
        // lea(LDA3, ptr[LDA + LDA * 2]);
        add_d(LDA3, LDA, LDA);
        add_d(LDA3, LDA3, LDA);
        // sub(A, -128);
        addi_d(A, A, 128);
        // sub(B, -128);
        addi_d(B, B, 128);
        // cmp(N, 0x4);
        mov_imm(TM, 0x4);
        // jl(labels[2], T_NEAR);
        blt(N, TM, labels[2]);
        // align(4);

        L(labels[5]);
        // mov(A1, A);
        add_d(A1, A, zero);
        // add(A, 0x4);
        addi_d(A, A, 0x4);
        // pxor(xmm7, xmm7);
        vxor_v(vr7, vr7, vr7);
        // mov(I, M);
        // sar(I, 0x3);
        srai_d(I, M, 0x3);
        // jle(labels[19], T_NEAR);
        bge(zero, I, labels[19]);
        // align(4);

        L(labels[9]);
        // movd(xmm0, dword[A1 - 0x80]);
        load_bytes(vr0, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm1, dword[A1 - 0x80]);
        load_bytes(vr1, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm2, dword[A1 - 0x80]);
        load_bytes(vr2, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm3, dword[A1 - 0x80]);
        load_bytes(vr3, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        // punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        // punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // movhlps(xmm6, xmm0);
        vextrins_d(vr6, vr0, 0x01);
        // pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        // phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, - 0x80);
        // movd(xmm0, dword[A1 - 0x80]);
        load_bytes(vr0, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm1, dword[A1 - 0x80]);
        load_bytes(vr1, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm2, dword[A1 - 0x80]);
        load_bytes(vr2, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm3, dword[A1 - 0x80]);
        load_bytes(vr3, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        // punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        // punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // movhlps(xmm6, xmm0);
        vextrins_d(vr6, vr0, 0x01);
        // pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        // phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movdqu(xword[B - 0x70], xmm0);
        vst(vr0, B, - 0x70);
        // sub(B, -32);
        addi_d(B, B, 32);
        // dec(I);
        addi_d(I, I, -1);
        // jg(labels[9], T_NEAR);
        blt(zero, I, labels[9]);
        // align(4);

        L(labels[19]);
        // test(M, 0x4);
        andi(TM, M, 0x4);
        // jle(labels[20], T_NEAR);
        bge(zero, TM, labels[20]); 
        // movd(xmm0, dword[A1 - 0x80]);
        load_bytes(vr0, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm1, dword[A1 - 0x80]);
        load_bytes(vr1, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm2, dword[A1 - 0x80]);
        load_bytes(vr2, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm3, dword[A1 - 0x80]);
        load_bytes(vr3, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        // punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        // punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // movhlps(xmm6, xmm0);
        vextrins_d(vr6, vr0, 0x01);
        // pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        // phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, - 0x80);
        // sub(B, -16);
        addi_d(B, B, 16);
        // align(4);

        L(labels[20]);
        // test(M, 0x2);
        andi(TM, M, 0x2);
        // jle(labels[0], T_NEAR);
        bge(zero, TM, labels[0]);
        // movd(xmm0, dword[A1 - 0x80]);
        load_bytes(vr0, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // movd(xmm1, dword[A1 - 0x80]);
        load_bytes(vr1, A1, - 0x80, 4);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movq(qword[B - 0x80], xmm0);
        store_bytes(vr0, B, - 0x80, 8);
        // sub(B, -8);
        addi_d(B, B, 8);
        // align(4);

        L(labels[0]);
        // test(M, 0x1);
        andi(TM, M, 0x1);
        // jle(labels[1], T_NEAR);
        bge(zero, TM, labels[1]);
        // movd(xmm0, dword[A1 - 0x80]);
        load_bytes(vr0, A1, - 0x80, 4);
        // pmovzxbd(xmm5, xmm0);
        vext2xv_wu_bu(xr5, xr0);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movd(dword[B - 0x80], xmm0);
        store_bytes(vr0, B, - 0x80, 4);
        // sub(B, -4);
        addi_d(B, B, 4);
        // align(4);

        L(labels[1]);
        // mov(A1, qword[ARG_BIAS]);
        ld_d(A1, sp, ARG_BIAS);
        // movdqu(xword[A1], xmm7);
        vst(vr7, A1, 0);
        // add(qword[ARG_BIAS], 0x10);
        addi_d(A1, A1, 0x10);
        st_d(A1, sp, ARG_BIAS);
        // sub(N, 0x4);
        addi_d(N, N, -0x4);
        // cmp(N, 0x4);
        mov_imm(TM, 0x4);
        // jge(labels[5], T_NEAR);
        bge(N, TM, labels[5]);
        // align(4);

        L(labels[2]);
        // cmp(N, 0x2);
        mov_imm(TM, 0x2);
        // jl(labels[11], T_NEAR);
        blt(N, TM, labels[11]);
        // align(4);

        L(labels[3]);
        // mov(A1, A);
        add_d(A1, A, zero);
        // add(A, 0x2);
        addi_d(A, A, 0x2);
        // pxor(xmm7, xmm7);
        vxor_v(vr7, vr7, vr7);
        // mov(LDA3, M);
        // sar(LDA3, 0x3);
        srai_d(LDA3, M, 0x3);
        // jle(labels[6], T_NEAR);
        bge(zero, LDA3, labels[6]);
        // align(4);

        L(labels[4]);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm1, eax, 0x0);
        vinsgr2vr_h(vr1, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm2, eax, 0x0);
        vinsgr2vr_h(vr2, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm3, eax, 0x0);
        vinsgr2vr_h(vr3, TM, 0x0);
        // punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        // punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        // punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm1, eax, 0x0);
        vinsgr2vr_h(vr1, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm2, eax, 0x0);
        vinsgr2vr_h(vr2, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm3, eax, 0x0);
        vinsgr2vr_h(vr3, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm4, eax, 0x0);
        vinsgr2vr_h(vr4, TM, 0x0);
        // punpcklbw(xmm1, xmm2);
        vilvl_b(vr1, vr2, vr1);
        // punpcklbw(xmm3, xmm4);
        vilvl_b(vr3, vr4, vr3);
        // punpcklwd(xmm1, xmm3);
        vilvl_h(vr1, vr3, vr1);
        // punpcklqdq(xmm0, xmm1);
        vilvl_d(vr0, vr1, vr0);
        // pshufd(xmm6, xmm0, 0xd8);
        vshuf4i_w(vr6, vr0, 0xd8);
        // pmovzxbw(xmm5, xmm6);
        vext2xv_hu_bu(xr5, xr6);
        // movhlps(xmm6, xmm6);
        vextrins_d(vr6, vr6, 0x01);
        // pmovzxbw(xmm6, xmm6);
        vext2xv_hu_bu(xr6, xr6);
        // phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movdqu(xword[B - 0x80], xmm0);
        vst(vr0, B, - 0x80);
        // sub(B, -16);
        addi_d(B, B, 16);
        // dec(LDA3);
        addi_d(LDA3, LDA3, -1);
        // jg(labels[4], T_NEAR);
        blt(zero, LDA3, labels[4]);
        // align(4);

        L(labels[6]);
        // test(M, 0x4);
        andi(TM, M, 0x4);
        // jle(labels[7], T_NEAR);
        bge(zero, TM, labels[7]);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm1, eax, 0x0);
        vinsgr2vr_h(vr1, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm2, eax, 0x0);
        vinsgr2vr_h(vr2, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm3, eax, 0x0);
        vinsgr2vr_h(vr3, TM, 0x0);
        // punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        // punpcklbw(xmm2, xmm3);
        vilvl_b(vr2, vr3, vr2);
        // punpcklwd(xmm0, xmm2);
        vilvl_h(vr0, vr2, vr0);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movq(qword[B - 0x80], xmm0);
        store_bytes(vr0, B, - 0x80, 8);
        // sub(B, -8);
        addi_d(B, B, 8);
        // align(4);


        L(labels[7]);
        // test(M, 0x2);
        andi(TM, M, 0x2);
        // jle(labels[8], T_NEAR);
        bge(zero, TM, labels[8]);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM, 0x0);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrw(xmm1, eax, 0x0);
        vinsgr2vr_h(vr1, TM, 0x0);
        // punpcklbw(xmm0, xmm1);
        vilvl_b(vr0, vr1, vr0);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movd(dword[B - 0x80], xmm0);
        store_bytes(vr0, B, - 0x80, 4);
        // sub(B, -4);
        addi_d(B, B, 4);
        // align(4);

        L(labels[8]);
        // test(M, 0x1);
        andi(TM, M, 0x1);
        // jle(labels[10], T_NEAR);
        bge(zero, TM, labels[10]);
        // mov(ax, word[A1 - 0x80]);
        ld_h(TM, A1, - 0x80);
        // pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM, 0x0);
        // pmovzxbd(xmm5, xmm0);
        vext2xv_wu_bu(xr5, xr0);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // mov(word[B - 0x80], ax);
        st_h(TM, B, - 0x80);
        // sub(B, -2);
        addi_d(B, B, 2);
        // align(4);

        L(labels[10]);
        // mov(A1, qword[ARG_BIAS]);
        ld_d(A1, sp, ARG_BIAS);
        // movq(qword[A1], xmm7);
        store_bytes(vr7, A1, 0, 8);
        // add(qword[ARG_BIAS], 0x8);
        addi_d(A1, A1, 0x8);
        st_d(A1, sp, ARG_BIAS);
        // sub(N, 0x2);
        addi_d(N, N, -0x2);
        // cmp(N, 0x2);
        mov_imm(TM, 0x2);
        // jge(labels[3], T_NEAR);
        bge(N, TM, labels[3]);
        // align(4);

        L(labels[11]);
        // cmp(N, 0x1);
        mov_imm(TM, 0x1);
        // jl(labels[18], T_NEAR);
        blt(N, TM, labels[18]);
        // align(4);

        L(labels[12]);
        // mov(A1, A);
        add_d(A1, A, zero);
        // add(A, 0x1);
        addi_d(A, A, 0x1);
        // pxor(xmm7, xmm7);
        vxor_v(vr7, vr7, vr7);
        // mov(LDA3, M);
        // sar(LDA3, 0x3);
        srai_d(LDA3, M, 0x3);
        // jle(labels[14], T_NEAR);
        bge(zero, LDA3, labels[14]);
        // align(4);

        L(labels[13]);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x0);
        vinsgr2vr_b(vr0, TM, 0x0);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x1);
        vinsgr2vr_b(vr0, TM, 0x1);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x2);
        vinsgr2vr_b(vr0, TM, 0x2);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x3);
        vinsgr2vr_b(vr0, TM, 0x3);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x4);
        vinsgr2vr_b(vr0, TM, 0x4);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x5);
        vinsgr2vr_b(vr0, TM, 0x5);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x6);
        vinsgr2vr_b(vr0, TM, 0x6);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x7);
        vinsgr2vr_b(vr0, TM, 0x7);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // phaddw(xmm5, xmm6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movq(qword[B - 0x80], xmm0);
        store_bytes(vr0, B, - 0x80, 8);
        // sub(B, -8);
        addi_d(B, B, 8);
        // dec(LDA3);
        addi_d(LDA3, LDA3, -1);
        // jg(labels[13], T_NEAR);
        blt(zero, LDA3, labels[13]);
        // align(4);

        L(labels[14]);
        // test(M, 0x4);
        andi(TM, M, 0x4);
        // jle(labels[15], T_NEAR);
        bge(zero, TM, labels[15]);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x0);
        vinsgr2vr_b(vr0, TM, 0x0);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x1);
        vinsgr2vr_b(vr0, TM, 0x1);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x2);
        vinsgr2vr_b(vr0, TM, 0x2);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x3);
        vinsgr2vr_b(vr0, TM, 0x3);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // movd(dword[B - 0x80], xmm0);
        store_bytes(vr0, B, - 0x80, 4);
        // sub(B, -4);
        addi_d(B, B, 4);
        // align(4);


        L(labels[15]);
        // test(M, 0x2);
        andi(TM, M, 0x2);
        // jle(labels[16], T_NEAR);
        bge(zero, TM, labels[16]);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x0);
        vinsgr2vr_b(vr0, TM, 0x0);
        // mov(byte[B - 0x80], al);
        st_b(TM, B, - 0x80);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // add(A1, LDA);
        add_d(A1, A1, LDA);
        // pinsrb(xmm0, eax, 0x1);
        vinsgr2vr_b(vr0, TM, 0x1);
        // pmovzxbw(xmm5, xmm0);
        vext2xv_hu_bu(xr5, xr0);
        // phaddw(xmm5, xmm5);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        // pmovzxwd(xmm5, xmm5);
        vext2xv_wu_hu(xr5, xr5);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // mov(byte[B - 0x7f], al);
        st_b(TM, B, - 0x7f);
        // sub(B, -2);
        addi_d(B, B, 2);
        // align(4);

        L(labels[16]);
        // test(M, 0x1);
        andi(TM, M, 0x1);
        // jle(labels[17], T_NEAR);
        bge(zero, TM, labels[17]);
        // mov(al, byte[A1 - 0x80]);
        ld_b(TM, A1, - 0x80);
        // pinsrw(xmm0, eax, 0x0);
        vinsgr2vr_h(vr0, TM, 0x0);
        // pmovzxbd(xmm5, xmm0);
        vext2xv_wu_bu(xr5, xr0);
        // paddd(xmm7, xmm5);
        vadd_w(vr7, vr7, vr5);
        // mov(byte[B - 0x80], al);
        st_b(TM, B, - 0x80);
        // sub(B, -1);
        addi_d(B, B, 1);
        // align(4);


        L(labels[17]);
        // mov(A1, qword[ARG_BIAS]);
        ld_d(A1, sp, ARG_BIAS);
        // movd(dword[A1], xmm7);
        store_bytes(vr7, A1, 0, 4);
        // add(qword[ARG_BIAS], 0x4);
        addi_d(A1, A1, 0x4);
        st_d(A1, sp, ARG_BIAS);
        // sub(N, 0x1);
        addi_d(N, N, -0x1);
        // cmp(N, 0x1);
        mov_imm(TM, 0x1);
        // jge(labels[12], T_NEAR);
        bge(N, TM, labels[12]);
        // align(4);

        L(labels[18]);
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
#ifdef _WIN32
// #undef ARG_ALPHA
// #undef ARG_B
#endif
// #undef ARG_BIAS
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
