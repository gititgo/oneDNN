/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#ifndef CPU_LOONGARCH64_GEMM_S8X8S32_JIT_LASX_GEMM_S8U8S32_KERN_HPP
#define CPU_LOONGARCH64_GEMM_S8X8S32_JIT_LASX_GEMM_S8U8S32_KERN_HPP

#include "cpu/loongarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

class jit_lasx_gemm_s8u8s32_kern : public jit_generator {
public:
    jit_lasx_gemm_s8u8s32_kern(bool beta_zero, bool enable_offset_c,
            bool enable_offset_r, int unroll_m);
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lasx_gemm_s8u8s32_kern);

protected:
    bool beta_zero_;
    bool enable_offset_c_, enable_offset_r_;
    bool vnni_;
    int unroll_m_;

    // use uni_preld instead in loongarch
    // void prefetch_a(const Xbyak::Address &src) { prefetcht0(src); }
    // void prefetch_b(const Xbyak::Address &src) { prefetcht0(src); }
    // void prefetch_c(const Xbyak::Address &src) { prefetchw(src); }
    // void prefetch_x(const Xbyak::Address &src) { prefetcht1(src); }

    // void c_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);
    // void c_store(const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems);
    void c_load(const Xbyak_loongarch::XVReg &dst, const Xbyak_loongarch::XReg &src, 
                int64_t offset, int nelems);
    void c_store(const Xbyak_loongarch::XVReg &dst, const Xbyak_loongarch::XReg &src, 
                int64_t offset, int nelems);
    void dot_product(const Xbyak_loongarch::XVReg &dst, 
                    const Xbyak_loongarch::XVReg &b, 
                    const Xbyak_loongarch::XVReg &a);
    
    void kernel_loop(int unroll_m, int unroll_n, bool cfetch);
    void remainder_kernel(int unroll_m, int unroll_n, int unroll_k, int bwidth);
    void innerloop(int unroll_m, int unroll_n);
    void outerloop(int unroll_x, int unroll_y, Xbyak_loongarch::Label *&outerloop_label);

    void generate() override ATTRIBUTE_OPTIMIZE;

private:
    static const int IGEMM_UNROLL_N_ = 4;

    static const int size_ = 4;
    static const int isize_ = 1;

    // Prefetch configuration
    static const int prefetch_size_a_ = 704;
    static const int prefetch_size_b_ = 384;

    static const int offset_a_ = 128, offset_b_ = 128;
    static const int max_unroll_m_ = 24, max_unroll_n_ = 4;

    // Integer register assignments
    //Xbyak::Reg64 M_, N_, K_, A_, B_, C_, LDC_, I_, J_, LoopCount_;
    Xbyak_loongarch::XReg M_ = abi_param1;  //rdi
    Xbyak_loongarch::XReg N_ = abi_param2;  //rsi
    Xbyak_loongarch::XReg K_ = abi_param3;  //rdx
    Xbyak_loongarch::XReg A_ = abi_param5;  //r8
    Xbyak_loongarch::XReg B_ = abi_param6;  //r9
    Xbyak_loongarch::XReg C_ = abi_param7;  //r10
    Xbyak_loongarch::XReg LDC_ = abi_param8;    //r11

    Xbyak_loongarch::XReg rax = t1;     // for calc in loongarch
    Xbyak_loongarch::XReg I_ = t2;  // r12
    Xbyak_loongarch::XReg J_ = t3;  //r13
    Xbyak_loongarch::XReg LoopCount_ = t4;  // rax
    //Xbyak::Reg64 AO_, BO_, CO1_, CO2_, AA_;
    Xbyak_loongarch::XReg AO_ = t5;     // r14
    Xbyak_loongarch::XReg BO_ = t6;     // r15
    Xbyak_loongarch::XReg CO1_ = t7;    // rbx
    Xbyak_loongarch::XReg CO2_ = t8;    // rbp
    Xbyak_loongarch::XReg AA_ = abi_param4; // rcx

    // Vector register assignments
    // Xbyak::Ymm dp_scratch_, ones_;
    // Xbyak::Ymm a_regs_[max_unroll_m_ >> 3], b_regs_[2];
    // Xbyak::Ymm c_regs_[max_unroll_m_ >> 3][max_unroll_n_];
    // Xbyak_loongarch::XVReg dp_scratch_ = xr6;   // Don't need.
    // Xbyak_loongarch::XVReg ones_ = xr7;          // Don't need.
    Xbyak_loongarch::XVReg a_regs_[3] = {xr0, xr1, xr2};
    Xbyak_loongarch::XVReg b_regs_ = xr3;
    Xbyak_loongarch::XVReg c_regs_[3][4] = {xr8, xr9, xr10, xr11,
                                            xr12, xr13, xr14, xr15,
                                            xr4, xr5, xr6, xr7}; 

    // Stack variable assignments
    int stack_alloc_size_ = 96;
    int args_offset = stack_alloc_size_ + get_size_of_abi_save_regs();
    // Xbyak::Address arg_a_, arg_b_, arg_c_, arg_ldc_, arg_coffset_c_,
    //         arg_coffset_r_;
    // Xbyak::Address coffset_cx_, coffset_cy_, coffset_rx_, coffset_ry_;
    // Xbyak::Address bcast_k2_, bcast_k1_;

    // Xbyak_loongarch::Address arg_a_ = ptr_a(sp, args_offset - 16);   // abi_param5 in loongarch
    // Xbyak_loongarch::Address arg_b_ = ptr_a(sp, args_offset - 8);    // abi_param6 in loongarch
    // Xbyak_loongarch::Address arg_c_ = ptr_a(sp, args_offset + 0);    // abi_param7 in loongarch
    // Xbyak_loongarch::Address arg_ldc_ = ptr_a(sp, args_offset + 8);  // abi_param8 in loongarch
    Xbyak_loongarch::Address arg_coffset_c_ = ptr_a(sp, args_offset + 0);
    Xbyak_loongarch::Address arg_coffset_r_ = ptr_a(sp, args_offset + 8);
    // Xbyak_loongarch::Address bcast_k2_ = ptr_a(sp, 0); 
    // Xbyak_loongarch::Address bcast_k1_ = ptr_a(sp, 32);
    Xbyak_loongarch::Address coffset_cx_ = ptr_a(sp, 64);
    Xbyak_loongarch::Address coffset_cy_ = ptr_a(sp, 72);
    Xbyak_loongarch::Address coffset_rx_ = ptr_a(sp, 80);
    Xbyak_loongarch::Address coffset_ry_ = ptr_a(sp, 88);
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_LOONGARCH64_GEMM_S8X8S32_JIT_LASX_GEMM_S8U8S32_KERN_HPP
