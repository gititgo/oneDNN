/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#ifndef CPU_LOONGARCH64_JIT_UNI_POOL_KERNEL_HPP
#define CPU_LOONGARCH64_JIT_UNI_POOL_KERNEL_HPP

#include <cfloat>
#include <functional>
#include <memory>

#include "common/memory_tracking.hpp"

#include "cpu/loongarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/loongarch64/jit_generator.hpp"
#include "cpu/loongarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

//struct bf16_emulation_t;
using namespace Xbyak_loongarch;

template <cpu_isa_t isa>
struct jit_uni_pool_kernel : public jit_generator {

    jit_uni_pool_kernel(
            const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md);
    jit_pool_conf_t jpp;
    ~jit_uni_pool_kernel();

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_kernel)

    static status_t init_conf(jit_pool_conf_t &jbp,
            memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
            int nthreads);

private:
    //using Xmm = Xbyak::Xmm;
    //using Ymm = Xbyak::Ymm;
    //using Zmm = Xbyak::Zmm;
    //using Opmask = Xbyak::Opmask;
    //using Reg32 = Xbyak::Reg32;
    //using Reg64 = Xbyak::Reg64;
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    int vmm_idx_upper_bound() const noexcept {
        //return utils::one_of(isa, avx512_common, avx512_core) ? 31 : 15;
        return 15;
    }

    int reg_idx(int idx) const noexcept { return vmm_idx_upper_bound() - idx; }

    //Xmm xreg(int idx) const noexcept { return Xmm(reg_idx(idx)); }
    VReg xreg(int idx) const noexcept { return VReg(reg_idx(idx)); }
    //Ymm yreg(int idx) const noexcept { return Ymm(reg_idx(idx)); }
    XVReg yreg(int idx) const noexcept { return XVReg(reg_idx(idx)); }
    //Zmm zreg(int idx) const noexcept { return Zmm(reg_idx(idx)); }
    Vmm vreg(int idx) const noexcept { return Vmm(reg_idx(idx)); }

    //const Xbyak::AddressFrame &vmmword = (isa == sse41)
    //        ? xword
    //        : (isa == avx || isa == avx2) ? yword : zword;

    /* x86 registers define
    Xmm vmm_mask = Xmm(0);
    Xmm xmm_tmp_1 = Xmm(0);
    Ymm ymm_tmp_1 = Ymm(0);
    Vmm vmm_tmp_1 = Vmm(0);

    // Used only for avx and if c tail is present
    Vmm vmm_c_tail_mask = Vmm(2);

    Xmm xmm_ker_area_h = Xmm(2);
    Xmm xmm_one = Xmm(2);
    Xmm xmm_tmp = Xmm(3);

    Vmm vmm_ker_area_h = Vmm(2);
    Vmm vmm_one = Vmm(2);
    Vmm vmm_tmp = Vmm(3);
    Ymm ymm_tmp = Ymm(3);

    Vmm vmm_k_offset = Vmm(1);

    // Used only for avx512 when bf16 is present
    inline Vmm vmm_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? Vmm(4) : Vmm(1);
        } else
            return Vmm(4);
    }

    Zmm bf16_emu_reserv_1 = Zmm(5);
    Zmm bf16_emu_reserv_2 = Zmm(6);
    Zmm bf16_emu_reserv_3 = Zmm(7);
    Reg64 bf16_emu_reserv_4 = r11;
    Zmm bf16_emu_reserv_5 = Zmm(8);

    Opmask k_c_tail_mask = Opmask(4);
    Opmask k_mask_cvt = Opmask(5);
    Opmask k_store_mask = Opmask(6);

    // Here be some (tame) dragons. This kernel does not follow the regular
    // OS-agnostic ABI pattern because when isa is sse41 it uses maskmovdqu
    // instruction which has its destination hardcoded in rdi. Therefore:
    // - all registers are hardcoded
    // - on Windows rdi and rcx are swapped to mimic the Unix x86_64 ABI
    //
    // While this is only required by the backward pass, the quirk above
    // is applied to the forward pass as well to keep things simpler.

    using reg64_t = const Reg64;
    reg64_t reg_param = rdi; // Always mimic the Unix ABI
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t reg_index = r10;
    reg64_t reg_output = r12;
    reg64_t reg_kd_pad_shift = r13;
    reg64_t dst_ptr = rdi; // Must be rdi due to maskmovdqu

    reg64_t kj = r14;
    reg64_t oi_iter = r15;
    reg64_t reg_kh = rax;
    reg64_t reg_k_shift = rbx;
    reg64_t tmp_gpr = rcx; // Must be rcx because rdi is used above
    reg64_t reg_ker_area_h = rdx;
    reg64_t reg_nbc = rsi;

    reg64_t reg_zero_ptr = r9;
    reg64_t reg_zero_id = r13;
    reg64_t reg_zero_ih = r14;
    reg64_t aux_reg_zero_ih = r15;
    reg64_t ki = r12;
    reg64_t aux_reg_input_d = r8;

    Reg32 reg_shuf_mask = esi; */

    VReg vmm_mask = VReg(0);
    VReg xmm_tmp_1 = VReg(0);
    XVReg ymm_tmp_1 = XVReg(0);
    Vmm vmm_tmp_1 = Vmm(0);

    // Used only for avx and if c tail is present
    Vmm vmm_c_tail_mask = Vmm(2);

    VReg xmm_ker_area_h = VReg(2);
    VReg xmm_one = VReg(2);
    VReg xmm_tmp = VReg(3);

    Vmm vmm_ker_area_h = Vmm(2);
    Vmm vmm_one = Vmm(2);
    Vmm vmm_tmp = Vmm(3);
    XVReg ymm_tmp = XVReg(3);

    Vmm vmm_k_offset = Vmm(1);

    // Used only for avx512 when bf16 is present
    inline Vmm vmm_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? Vmm(4) : Vmm(1);
        } else
            return Vmm(4);
    }

    // Here be some (tame) dragons. This kernel does not follow the regular
    // OS-agnostic ABI pattern because when isa is sse41 it uses maskmovdqu
    // instruction which has its destination hardcoded in rdi. Therefore:
    // - all registers are hardcoded
    // - on Windows rdi and rcx are swapped to mimic the Unix x86_64 ABI
    //
    // While this is only required by the backward pass, the quirk above
    // is applied to the forward pass as well to keep things simpler.

    XReg reg_param = a0; // Always mimic the Unix ABI
    XReg reg_input = t0;
    XReg aux_reg_input = t1;
    XReg reg_index = t2;
    XReg reg_output = t3;
    XReg reg_kd_pad_shift = t4;
    XReg dst_ptr = a0; // Must be rdi due to maskmovdqu

    XReg kj = t5;
    XReg oi_iter = t6;
    XReg reg_kh = t7;
    XReg reg_k_shift = t8;
    XReg tmp_gpr = s0; // Must be rcx because rdi is used above
    XReg reg_ker_area_h = a2;
    XReg reg_nbc = a3;

    XReg reg_zero_ptr = t1;
    XReg reg_zero_id = t4;
    XReg reg_zero_ih = t5;
    XReg aux_reg_zero_ih = t6;
    XReg ki = t3;
    XReg aux_reg_input_d = t0;

    XReg reg_shuf_mask = a4;

    bool sse_high_half = false;
    bool disable_postops_when_sse_high_half_processed_ = false;

    int prev_kw;

    void prepare_tail_mask();
    void put_one_in_vmm();
    void uni_broadcast_reg_val(const int reg_idx, const int vmm_idx);
    void push_vmm_val(const int idx);
    void pop_vmm_val(const int idx);
    void load(const int idx, const XReg &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);
    void store(const int idx, const XReg &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void avg_step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_fwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_bwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);

    void zero_diff_src(int ur_bc, bool with_c_tail_proccessing);

    void step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing) {
        if (jpp.alg == alg_kind::pooling_max) {
            if (jpp.is_backward)
                max_step_bwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
            else
                max_step_fwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
        } else
            avg_step(ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
    }

    void step_high_half(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_processing) {
        //add(reg_input, sizeof(float) * 4);
        addi_d(reg_input, reg_input, sizeof(float) * 4);
        //add(reg_output, sizeof(float) * 4);
        addi_d(reg_output, reg_output, sizeof(float) * 4);
        if (jpp.alg == alg_kind::pooling_max
                && (jpp.is_training || jpp.is_backward))
            //add(reg_index, types::data_type_size(jpp.ind_dt) * 4);
            add_imm(reg_index, reg_index, types::data_type_size(jpp.ind_dt) * 4, tmp_gpr);

        step(ur_w, ur_bc, pad_l, pad_r, with_c_tail_processing);
    }

    void generate() override;

    //void avx_vpadd1(const Ymm &y0, const Xmm &x1, const Xmm &xtmp) {
    void avx_vpadd1(const XVReg &y0, const VReg &x1, const VReg &xtmp) {
        assert(y0.getIdx() != x1.getIdx());
        //vextractf128(xtmp, y0, 0);
        xvpermi_q(XVReg(xtmp.getIdx()), y0, 0x30);
        //vpaddd(xtmp, xtmp, x1);
        vadd_w(xtmp, xtmp, x1);
        //vinsertf128(y0, y0, xtmp, 0);
        xvpermi_q(y0, XVReg(xtmp.getIdx()), 0x30);
        //vextractf128(xtmp, y0, 1);
        xvpermi_q(XVReg(xtmp.getIdx()), y0, 0x31);
        //vpaddd(xtmp, xtmp, x1);
        vadd_w(xtmp, xtmp, x1);
        //vinsertf128(y0, y0, xtmp, 1);
        xvpermi_q(y0, XVReg(xtmp.getIdx()), 0x02);
    }

    //void avx_vpadd1(const Xmm &x0, const Xmm &x1, const Xmm &) {
    void avx_vpadd1(const VReg &x0, const VReg &x1, const VReg &) {
        assert(false /*function should not be used*/);
        //paddd(x0, x1);
        vadd_w(x0, x0, x1);
    }

    //void avx_pmovzxbd(const Ymm &y0, const Xmm &x1, const Xmm &xtmp) {
    void avx_pmovzxbd(const XVReg &y0, const VReg &x1, const VReg &xtmp) {
        //Xmm x0(y0.getIdx());
        VReg x0(y0.getIdx());
        //pshufd(xmm_tmp, x1, 1);
        vshuf4i_w(xmm_tmp, x1, 1);
        //pmovzxbd(x0, x1);
        vext2xv_wu_bu(XVReg(x0.getIdx()), XVReg(x1.getIdx()));
        //pmovzxbd(xmm_tmp, xmm_tmp);
        vext2xv_wu_bu(XVReg(xmm_tmp.getIdx()), XVReg(xmm_tmp.getIdx()));
        //vinsertf128(y0, y0, xmm_tmp, 1);
        xvpermi_q(y0, XVReg(xmm_tmp.getIdx()), 0x02);
    }

    //void avx_pmovzxbd(const Xmm &x0, const Xmm &x1, const Xmm &) {
    void avx_pmovzxbd(const VReg &x0, const VReg &x1, const VReg &) {
        assert(false /*function should not be used*/);
        //pmovzxbd(x0, x1);
        vext2xv_wu_bu(XVReg(x0.getIdx()), XVReg(x1.getIdx()));
    }

    void avx_pcmpeqd(
            //const Ymm &y0, const Ymm &y1, const Ymm &y2, const Xmm &xtmp) {
            const XVReg &y0, const XVReg &y1, const XVReg &y2, const VReg &xtmp) {
        assert(y0.getIdx() != y1.getIdx());
        assert(y0.getIdx() != y2.getIdx());
        VReg x0(y0.getIdx());
        VReg x2(y2.getIdx());
        //vextractf128(x0, y1, 1);
        xvpermi_q(y0, y1, 0x31);
        //vextractf128(xtmp, y2, 1);
        xvpermi_q(XVReg(xtmp.getIdx()), y2, 0x31);
        //pcmpeqd(xtmp, x0);
        vseq_w(xtmp, xtmp, x0);
        //vextractf128(x0, y1, 0);
        xvpermi_q(y0, y1, 0x30);
        //pcmpeqd(x0, x2);
        vseq_w(x0, x0, x2);
        //vinsertf128(y0, y0, xtmp, 1);
        xvpermi_q(y0, XVReg(xtmp.getIdx()), 0x02);
    }

    void avx_pcmpeqd(const VReg &x0, const VReg &x1, const VReg &, const VReg &) {
        assert(false /*function should not be used*/);
        //pcmpeqd(x0, x1);
        vseq_w(x0, x0, x1);
    }

    void apply_postops(int ur_bc, int ur_w, int c_block,
            const std::function<bool(int, bool)> &is_tail_predicate);

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    //std::unique_ptr<bf16_emulation_t> bf16_emu_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
