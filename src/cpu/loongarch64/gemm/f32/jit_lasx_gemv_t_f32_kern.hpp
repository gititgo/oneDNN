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

#ifndef CPU_LOONGARCH64_GEMM_F32_JIT_LASX_GEMV_T_F32_KERN_HPP
#define CPU_LOONGARCH64_GEMM_F32_JIT_LASX_GEMV_T_F32_KERN_HPP

#include "cpu/loongarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

class jit_lasx_gemv_t_f32_kern : public jit_generator {
public:
    jit_lasx_gemv_t_f32_kern(void);
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lasx_gemv_t_f32_kern);

protected:
    //bool is_avx2_;

    // this is load_bytes and store_bytes and vfmadd_s
    //void v_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);
    //void v_store(const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems);
    //void dot_product(const Xbyak::Xmm &dst, const Xbyak::Xmm &src1,
    //        const Xbyak::Xmm &src2);

    void innerloop(int unroll_m, int unroll_n);
    void outerloop(int unroll_x, int unroll_y, Xbyak_loongarch::Label *&outerloop_label);

    void generate() override ATTRIBUTE_OPTIMIZE;

private:
    static const int M_UNROLL_ = 16;
    static const int N_UNROLL_ = 4;

    static const int size_ = 4;

    static const int offset_a_ = 128, offset_x_ = 128;

    // Integer register assignments
    //Xbyak::Reg64 M_, N_, A_, LDA_, X_, INCY_, Y_, ALPHA_, I_, J_;
    Xbyak_loongarch::XReg M_ = abi_param1;
    Xbyak_loongarch::XReg N_ = abi_param2;
    Xbyak_loongarch::XReg ALPHA_ = abi_param3;
    Xbyak_loongarch::XReg A_ = abi_param4;
    Xbyak_loongarch::XReg LDA_ = abi_param5;
    Xbyak_loongarch::XReg X_ = abi_param6;
    Xbyak_loongarch::XReg INCY_ = abi_param7;
    Xbyak_loongarch::XReg Y_ = abi_param8; 
    Xbyak_loongarch::XReg I_ = t2;
    Xbyak_loongarch::XReg J_ = t3;
    //Xbyak::Reg64 AO_, XO_, YO_, YO2_;
    Xbyak_loongarch::XReg AO_ = t4;
    Xbyak_loongarch::XReg XO_ = t5;
    Xbyak_loongarch::XReg YO_ = t6;
    Xbyak_loongarch::XReg YO2_ = t7;

    // Vector register assignments
    //Xbyak::Ymm scratch_, alpha_, a_regs_[M_UNROLL_ >> 3][N_UNROLL_];
    Xbyak_loongarch::XVReg scratch_ = xr11;
    Xbyak_loongarch::XVReg alpha_ = xr10;
    Xbyak_loongarch::XVReg a_regs_[2][4] = {xr0, xr1, xr2, xr3, xr4, xr5, xr6, xr7};
    //Xbyak::Ymm x_regs_[M_UNROLL_ >> 3], y_regs_[N_UNROLL_], acc_[N_UNROLL_];
    Xbyak_loongarch::XVReg x_regs_[2] = {xr8, xr9};
    Xbyak_loongarch::XVReg y_regs_[N_UNROLL_] = {xr0, xr1, xr2, xr3};
    Xbyak_loongarch::XVReg acc_[N_UNROLL_] = {xr12, xr13, xr14, xr15};

    // Stack variable assignments
    //Xbyak::Address arg_lda_, arg_x_, arg_incx_, arg_y_, arg_incy_;
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_LOONGARCH64_GEMM_F32_JIT_LASX_GEMV_T_F32_KERN_HPP
