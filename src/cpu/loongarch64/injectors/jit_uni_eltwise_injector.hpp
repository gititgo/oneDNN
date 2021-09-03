/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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

#ifndef CPU_LOONGARCH64_JIT_UNI_ELTWISE_INJECTOR_HPP
#define CPU_LOONGARCH64_JIT_UNI_ELTWISE_INJECTOR_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/injectors/injector_utils.hpp"
#include "cpu/loongarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

namespace eltwise_injector {
struct static_params_t {

    static_params_t(bool save_state = true,
            //Xbyak_loongarch::XReg x_table = Xbyak_loongarch::XReg(0),
            Xbyak_loongarch::XReg x_table = Xbyak_loongarch::XReg(4),
            //Xbyak_loongarch::PReg p_mask = Xbyak_loongarch::PReg(1),
            Xbyak_loongarch::XVReg p_mask = Xbyak_loongarch::XVReg(29),
            //Xbyak_loongarch::PReg p_tmp0 = Xbyak_loongarch::PReg(4),
            //Xbyak_loongarch::PReg p_all = Xbyak_loongarch::PReg(7),
            bool is_fwd = true, bool use_dst = false)
        : save_state(save_state)
        , x_table(x_table)
        , p_mask(p_mask)
        //, p_tmp0(p_tmp0)
        //, p_all(p_all)
        , is_fwd(is_fwd)
        , use_dst(use_dst) {}

    bool save_state;
    Xbyak_loongarch::XReg x_table;
    //Xbyak_loongarch::PReg p_mask;
    Xbyak_loongarch::XVReg p_mask;
    //Xbyak_loongarch::PReg p_tmp0;
    //Xbyak_loongarch::PReg p_all;
    bool is_fwd;
    bool use_dst;
};
} // namespace eltwise_injector

template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_f32 {
    //using TReg = typename cpu_isa_traits<isa>::TReg;
    //using Vmm = typename cpu_isa_traits<isa>::Vmm;
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    // Arguments description:
    // host - jit generator which is filled with instructions
    // alg, alpha, beta, scale - user eltwise arguments
    // save_state - when true, preserves on stack vmm_aux registers preventing
    //   results spoiling. Restores them when done in injector_postamble().
    // p_table - GPR where table label is stored to get access for pre-defined
    //   constants used in alg codes.
    // k_mask - k_register to operate with masks in alg codes.
    // is_fwd - when true, computes d = alg(s), otherwise, computes ds = alg'(s)
    //   - algorithm derivative.
    // use_dst - defines whether source or destination point is passed to alg
    //   code. Depends on algorithm. See `_use_dst_for_bwd` algs definition.
    jit_uni_eltwise_injector_f32(jit_generator *host, alg_kind_t alg,
            float alpha, float beta, float scale, bool save_state = true,
            Xbyak_loongarch::XReg x_table = Xbyak_loongarch::XReg(4),
            //Xbyak_loongarch::PReg p_mask = Xbyak_loongarch::PReg(1),
            Xbyak_loongarch::XVReg p_mask = Xbyak_loongarch::XVReg(29),
            //Xbyak_loongarch::PReg p_tmp0 = Xbyak_loongarch::PReg(4),
            //Xbyak_loongarch::PReg p_all = Xbyak_loongarch::PReg(7),
            bool is_fwd = true, bool use_dst = false)
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , h(host)
        , save_state_(save_state)
        , x_table(x_table)
        , p_mask(p_mask)
        //, p_tmp0(p_tmp0)
        //, p_all(p_all)
        , is_fwd_(is_fwd)
        , use_dst_(use_dst)

    {
        using namespace alg_kind;
        assert(utils::one_of(isa, sve_512));
        assert(utils::one_of(alg_, eltwise_relu, eltwise_tanh, eltwise_elu,
                eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
                eltwise_exp, eltwise_gelu_tanh, eltwise_swish, eltwise_log,
                eltwise_clip, eltwise_clip_v2, eltwise_gelu_erf, eltwise_round,
                eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
                eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
                eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd,
                eltwise_clip_v2_use_dst_for_bwd));
        register_table_entries();
    }

    jit_uni_eltwise_injector_f32(jit_generator *host,
            const post_ops_t::entry_t::eltwise_t &eltwise,
            bool save_state = true,
            Xbyak_loongarch::XReg x_table = Xbyak_loongarch::XReg(4),
            //Xbyak_loongarch::PReg p_mask = Xbyak_loongarch::PReg(1),
            Xbyak_loongarch::XVReg p_mask = Xbyak_loongarch::XVReg(29),
            //Xbyak_loongarch::PReg p_tmp0 = Xbyak_loongarch::PReg(4),
            //Xbyak_loongarch::PReg p_all = Xbyak_loongarch::PReg(7),
            bool is_fwd = true, bool use_dst = false)
        : jit_uni_eltwise_injector_f32(host, eltwise.alg, eltwise.alpha,
                eltwise.beta, eltwise.scale, save_state, x_table, p_mask,
                is_fwd, use_dst) {}

    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs);
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }
    void prepare_table(bool gen_table = true);
    //void load_table_addr() { h->adr(x_table, l_table); }
    void load_table_addr() { h->pcaddi(x_table, l_table); }

private:
    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;

    jit_generator *const h;

    const bool save_state_;
    const Xbyak_loongarch::XReg x_table;
    //const Xbyak_loongarch::PReg p_mask;
    const Xbyak_loongarch::XVReg p_mask;
    //const Xbyak_loongarch::PReg p_tmp0;
    //const Xbyak_loongarch::PReg p_all;
    const bool is_fwd_;
    const bool use_dst_;

    Xbyak_loongarch::Label l_table;

    // if only the injector was inherited from jit_generator...
    enum {
        _cmp_eq_oq = jit_generator::_cmp_eq_oq,
        _cmp_lt_os = jit_generator::_cmp_lt_os,
        _cmp_le_os = jit_generator::_cmp_le_os,
        _cmp_ge_os = jit_generator::_cmp_nlt_us,
        _cmp_gt_os = jit_generator::_cmp_nle_us,
        _op_floor = jit_generator::_op_floor,
        _op_mxcsr = jit_generator::_op_mxcsr
    };

    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;
    static constexpr size_t preserved_vecs_max = 9;
    static constexpr size_t preserved_gprs_max = 4;
    static constexpr size_t vecs_count = 32;
    static constexpr int n_mantissa_bits = 23;
    static constexpr int k_mask_size = 8;

    size_t vecs_to_preserve = 0;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t preserved_gpr_idxs[preserved_gprs_max] = {0};
    injector_utils::vmm_index_set_iterator_t start_idx_tail;

    /* These vector register must be assigned proper index. */
    Vmm vmm_mask {0}, vmm_aux0 {0}, vmm_aux1 {0}, vmm_aux2 {0}, vmm_aux3 {0},
            vmm_aux4 {0}, vmm_aux5 {0}, vmm_aux6 {0}, vmm_aux7 {0}, vmm_tmp {0};
    /* Default tempooral index. Chose a SVE register
     not to be same as jit_uni_eltwise.(cpp|hpp).
     This index is changed by assign_regs() in case of eltwise injection.
  */
    Vmm z_tmp {31};
    Vmm z_tmp2 {30};

    size_t aux_vecs_count();
    size_t aux_gprs_count();

    void compute_body(
            const injector_utils::vmm_index_set_iterator_t &start_idx_it,
            const injector_utils::vmm_index_set_iterator_t &end_idx_it);
    void injector_preamble(const injector_utils::vmm_index_set_t &vmm_idxs);
    void injector_preamble_tail(
            const injector_utils::vmm_index_set_iterator_t start_idx_it);
    void injector_postamble();
    void assign_regs();
    void set_coef_to_regs();
    void compute_cmp_mask(
            const Vmm &vmm_src, const Vmm &vmm_cmpare, int cmp_predicate);
    void blend_with_mask(const Vmm &vmm_dst, const Vmm &src);
    void test_mask();

    void exp_compute_vector_fwd(const Vmm &vmm_src);
    void relu_compute_vector_fwd(const Vmm &vmm_src);
    void relu_zero_ns_compute_vector_fwd(const Vmm &vmm_src);
    void elu_compute_vector_fwd(const Vmm &vmm_src);
    void tanh_compute_vector_fwd(const Vmm &vmm_src);
    void square_compute_vector_fwd(const Vmm &vmm_src);
    void abs_compute_vector_fwd(const Vmm &vmm_src);
    void sqrt_compute_vector_fwd(const Vmm &vmm_src);
    void linear_compute_vector_fwd(const Vmm &vmm_src);
    void bounded_relu_compute_vector_fwd(const Vmm &vmm_src);
    void soft_relu_compute_vector_fwd(const Vmm &vmm_src);
    void logistic_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_tanh_compute_vector_fwd(const Vmm &vmm_src);
    void swish_compute_vector_fwd(const Vmm &vmm_src);
    void log_compute_vector_fwd(const Vmm &vmm_src);
    void clip_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_erf_compute_vector_fwd(const Vmm &vmm_src);
    void round_compute_vector_fwd(const Vmm &vmm_src);

    void exp_compute_vector_bwd(const Vmm &vmm_src);
    void relu_compute_vector_bwd(const Vmm &vmm_src);
    void elu_compute_vector_bwd(const Vmm &vmm_src);
    void tanh_compute_vector_bwd(const Vmm &vmm_src);
    void square_compute_vector_bwd(const Vmm &vmm_src);
    void abs_compute_vector_bwd(const Vmm &vmm_src);
    void sqrt_compute_vector_bwd(const Vmm &vmm_src);
    void linear_compute_vector_bwd(const Vmm &vmm_src);
    void bounded_relu_compute_vector_bwd(const Vmm &vmm_src);
    void soft_relu_compute_vector_bwd(const Vmm &vmm_src);
    void logistic_compute_vector_bwd(const Vmm &vmm_src);
    void gelu_tanh_compute_vector_bwd(const Vmm &vmm_src);
    void swish_compute_vector_bwd(const Vmm &vmm_src);
    void log_compute_vector_bwd(const Vmm &vmm_src);
    void clip_compute_vector_bwd(const Vmm &vmm_src);
    void gelu_erf_compute_vector_bwd(const Vmm &vmm_src);

    enum key_t {
        scale = 0, // scale argument
        alpha, // alpha argument
        beta, // beta argument
        zero, // 0.f
        half, // 0.5f
        one, // 1.f  or  mask for exponent bits
        two, // 2.f
        minus_one, // -1.f  or  changes sign to opposite
        minus_two, // -2.f
        ln2f, // 0.69314718f
        positive_mask, // changes sign to positive
        sign_mask, // gets sign value
        exponent_bias, // (127 = 2^7 - 1), gets exponent bits
        exp_log2ef, // 1.44269502f - formula-based for approx
        exp_ln_flt_max_f, // logf(FLT_MAX) - max normal value
        exp_ln_flt_min_f, // logf(FLT_MIN) - min normal value
        exp_pol, // see correspondent table for float values
        exp_coeff1, // 0.6931473921 (0x3f31721c)
        exp_coeff2, // 0.2413862043 (0x3e772df2)
        exp_not_mask17, // ~((1u << 17) - 1)
        tanh_range, // tanh(x) = x - x^3/3 for |x| < tanh_range
        tanh_m1d3, // -1/3
        soft_relu_one_twenty_six, // 126.f
        soft_relu_mantissa_sign_mask, // mask for mantissa bits and sign
        soft_relu_pol, // see correspondent table for float values
        gelu_tanh_fitting_const, // 0.044715f
        gelu_tanh_fitting_const_times_three, // 0.134145f
        gelu_tanh_sqrt_two_over_pi, // sqrtf(2.f/pi) = 0.797884f
        gelu_erf_approx_const, // 0.3275911f - implementation based for approx
        gelu_erf_one_over_sqrt_two, // 1.f / sqrtf(2.f)
        gelu_erf_one_over_sqrt_pi, // 1.f / sqrtf(pi) = 0.564190f
        gelu_erf_pol, // see correspondent table for float values
        log_minus_inf, // -inf
        log_qnan, // qnan
        log_mantissa_mask, // gets mantissa bits
        log_full_k_reg_mask, // sets k_register with all bits of 1
        log_full_vector_reg_mask, // sets vector register will all bits of 1
        log_five_bit_offset, // 5 bits off (31 = 2^5 - 1)
        log_pol, // see correspondent table for float values
        log_predefined_vals, // see correspondent table for float values
        log_i127shl23,
        log_x7fffff,
        log_log2,
        log_log1p5,
        log_f2div3,
        log_coeffTbl,
        undef_key,
    };

    size_t table_off(key_t key, size_t key_off_val_shift = 0) {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        // TODO: enforce through data structure
        const auto it = entry_map_.find(key); // search an entry for a key
        assert(it != entry_map_.end());
        const auto &te = (*it).second;
        const auto scale = te.bcast ? vlen : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }

    Vmm table_val(key_t key, Vmm zreg, size_t key_off_val_shift = 0) {
        Xbyak_loongarch::XReg x_addr(h->X_DEFAULT_ADDR);
        auto off = table_off(key, key_off_val_shift);

        if (off) {
            h->add_imm(x_addr, x_table, off, h->X_TMP_0);
        } else {
            x_addr = x_table;
        }

        //h->ldr(TReg(zreg.getIdx()), ptr(x_addr));
        h->xvld(Vmm(zreg.getIdx()), x_addr, 0);
        return zreg;
    }

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t; // offsets are in bytes wrt p_table
    using table_entry_bcast_t = bool; // true => bcast value

    struct table_entry_t {
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };
    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };

    using table_t = std::multimap<key_t, table_entry_t>;
    using mapped_table_t = std::multimap<key_t, mapped_table_entry_t>;

    void register_table_entries();
    mapped_table_t entry_map_;
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif