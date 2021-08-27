/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#ifndef CPU_LOONGARCH64_JIT_GENERATOR_HPP
#define CPU_LOONGARCH64_JIT_GENERATOR_HPP

#include <limits.h>

#include "common/bit_cast.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/cpu_isa_traits.hpp"

#include "cpu/jit_utils/jit_utils.hpp"

#if defined(_WIN32) && !defined(__GNUC__)
#define STRUCT_ALIGN(al, ...) __declspec(align(al)) __VA_ARGS__
#else
#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))
#endif

#if GCC_WA_NO_TREE_DOMINATOR_OPTS
#define ATTRIBUTE_OPTIMIZE __attribute__((optimize("no-tree-dominator-opts")))
#else
#define ATTRIBUTE_OPTIMIZE
#endif

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { return STRINGIFY(jit_name); } \
    const char *source_file() const override { return __FILE__; }

#define IMM12_MIN_VALUE    -2048
#define IMM12_MAX_VALUE     2047
#define UIMM12_MAX_VALUE    4095
#define IMM14_MIN_VALUE    -8192
#define IMM14_MAX_VALUE     8191
#define UIMM14_MAX_VALUE    16383

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

// TODO: move this to jit_generator class?
namespace {

typedef enum {
    MAX_CODE_SIZE = 256 * 1024,
} max_code_size_t;

// TODO: move this somewhere else? Although this is only used by jit kernels
// (Roma)
static inline int float2int(float x) {
    return utils::bit_cast<int>(x);
}

// TODO: A GPR class that hides ABI details from the JIT kernels and allows
// numbering registers from 0 to 14 (x86_64) / 6 (x32) (gpr0, gpr1, ...) and
// stack register (sr).
//
// This will allow using syntax like this:
//
// param = gpr0;
// reg_input = gpr0;
// reg_output =  gpr1;
// ...
//
// #ifndef XBYAK64
// mov(param, ptr[sr])
// #endif
//
// (Roma)

constexpr Xbyak_loongarch::Operand::Code abi_save_gpr_regs[] = {
        Xbyak_loongarch::Operand::s0,
        Xbyak_loongarch::Operand::s1,
        Xbyak_loongarch::Operand::s2,
        Xbyak_loongarch::Operand::s3,
        Xbyak_loongarch::Operand::s4,
        Xbyak_loongarch::Operand::s5,
        Xbyak_loongarch::Operand::s6,
        Xbyak_loongarch::Operand::s7,
        Xbyak_loongarch::Operand::s8,
};

constexpr Xbyak_loongarch::Operand::Code abi_save_fpr_regs[] = {
        Xbyak_loongarch::Operand::f24,
        Xbyak_loongarch::Operand::f25,
        Xbyak_loongarch::Operand::f26,
        Xbyak_loongarch::Operand::f27,
        Xbyak_loongarch::Operand::f28,
        Xbyak_loongarch::Operand::f29,
        Xbyak_loongarch::Operand::f30,
        Xbyak_loongarch::Operand::f31,
};

static const Xbyak_loongarch::XReg abi_param1(Xbyak_loongarch::Operand::a0),
        abi_param2(Xbyak_loongarch::Operand::a1), abi_param3(Xbyak_loongarch::Operand::a2),
        abi_param4(Xbyak_loongarch::Operand::a3), abi_param5(Xbyak_loongarch::Operand::a4),
        abi_param6(Xbyak_loongarch::Operand::a5), abi_param7(Xbyak_loongarch::Operand::a6),
        abi_param8(Xbyak_loongarch::Operand::a7), abi_not_param1(Xbyak_loongarch::Operand::t8);
} // namespace

class jit_generator : public Xbyak_loongarch::CodeGenerator, public c_compatible {
public:
    using c_compatible::operator new;
    using c_compatible::operator new[];
    using c_compatible::operator delete;
    using c_compatible::operator delete[];

private:
    const size_t xreg_len = 8;          // 8 Byte
    const size_t vreg_len_preserve = 8; // Only bottom 8byte must be preserved.
    const size_t vreg_to_preserve = 8;  // VREG24 - VREG31

    const size_t num_abi_save_gpr_regs
            = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    const size_t num_abi_save_fpr_regs
            = sizeof(abi_save_fpr_regs) / sizeof(abi_save_fpr_regs[0]);

    const size_t preserved_stack_size = xreg_len * (1 + num_abi_save_gpr_regs)  //2->1
            + vreg_len_preserve * vreg_to_preserve;

    const size_t size_of_abi_save_regs = num_abi_save_gpr_regs * xreg_len
            + vreg_to_preserve * vreg_len_preserve;

public:
    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    const Xbyak_loongarch::XReg W_TMP_0 = s0;
    const Xbyak_loongarch::XReg W_TMP_1 = s1;
    const Xbyak_loongarch::XReg W_TMP_2 = s2;
    const Xbyak_loongarch::XReg W_TMP_3 = s3;
    const Xbyak_loongarch::XReg W_TMP_4 = s4;

    const Xbyak_loongarch::XReg X_TMP_0 = s0;
    const Xbyak_loongarch::XReg X_TMP_1 = s1;
    const Xbyak_loongarch::XReg X_TMP_2 = s2;
    const Xbyak_loongarch::XReg X_TMP_3 = s3;
    const Xbyak_loongarch::XReg X_TMP_4 = s4;
    const Xbyak_loongarch::XReg X_DEFAULT_ADDR = s5;
    const Xbyak_loongarch::XReg X_SP = s6;
    const Xbyak_loongarch::XReg X_TRANSLATOR_STACK = s7;

    const std::vector<Xbyak_loongarch::XReg> x_tmp_vec
            = {X_TMP_0, X_TMP_1, X_TMP_2, X_TMP_3, X_TMP_4};
    const int x_tmp_vec_size = x_tmp_vec.size();

    Xbyak_loongarch::XReg param1 = abi_param1;
    const int EVEX_max_8b_offt = 0x200;
    constexpr static size_t translator_stack_offset = 1024 * 128;
    const Xbyak_loongarch::XReg reg_EVEX_max_8b_offt = t0;

    inline size_t get_size_of_abi_save_regs() { return size_of_abi_save_regs; }

    void preamble() {
        int i = 1;
	addi_d(sp, sp, -1*preserved_stack_size);
	st_d(fp, sp, preserved_stack_size-8*(i++));
	add_d(fp, sp, zero);

        for (size_t j = 0; j < num_abi_save_fpr_regs; ++j) {
            fst_d(Xbyak_loongarch::XReg(abi_save_fpr_regs[j]), sp, preserved_stack_size-8*(i++));
        }
        for (size_t k = 0; k < num_abi_save_gpr_regs; ++k) {
            st_d(Xbyak_loongarch::XReg(abi_save_gpr_regs[k]), sp, preserved_stack_size-8*(i++));
	}

	add_d(X_SP, sp, zero);
        sub_imm(X_TRANSLATOR_STACK, X_SP, translator_stack_offset, X_TMP_0);
    }

    void postamble() {
        int i = 4;
        add_d(sp, fp, zero);

        for (size_t j = 0; j < num_abi_save_fpr_regs; ++j) {
            fld_d(Xbyak_loongarch::XReg(abi_save_fpr_regs[j]), sp, preserved_stack_size-8*(i++));
	}
	for (size_t k = 0; k < num_abi_save_gpr_regs; ++k) {
            ld_d(Xbyak_loongarch::XReg(abi_save_gpr_regs[k]), sp, preserved_stack_size-8*(i++));
        }

        ld_d(fp, sp, preserved_stack_size-8);
        addi_d(sp, sp, preserved_stack_size);

        jirl(zero, ra, 0);
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak_loongarch::Label &label) { Xbyak_loongarch::CodeGenerator::L(label); }

    void L_aligned(Xbyak_loongarch::Label &label, int alignment = 16) {
        align(alignment);
        L(label);
    }

    void uni_ld_b(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
	    ldx_b(rd, rj, X_TMP_2);
	    return;
	}
	ld_b(rd, rj, simm);
    }

    void uni_ld_bu(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            ldx_bu(rd, rj, X_TMP_2);
            return;
        }
        ld_bu(rd, rj, simm);
    }

    void uni_ld_h(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            ldx_h(rd, rj, X_TMP_2);
            return;
        }
        ld_h(rd, rj, simm);
    }

    void uni_ld_w(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            ldx_w(rd, rj, X_TMP_2);
            return;
        }
        ld_w(rd, rj, simm);
    }

    void uni_ld_d(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            ldx_d(rd, rj, X_TMP_2);
            return;
        }
        ld_d(rd, rj, simm);
    }

    void uni_fld_s(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            fldx_s(rd, rj, X_TMP_2);
            return;
        }
        fld_s(rd, rj, simm);
    }

    void uni_fld_d(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            fldx_d(rd, rj, X_TMP_2);
            return;
        }
        fld_d(rd, rj, simm);
    }

    void uni_ll_w(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            ll_w(rd, X_TMP_2, 0);
            return;
        }
        ll_w(rd, rj, simm);
    }

    void uni_ll_d(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            ll_d(rd, X_TMP_2, 0);
            return;
        }
        ll_d(rd, rj, simm);
    }

    void uni_ldptr_w(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            ldptr_w(rd, X_TMP_2, 0);
            return;
        }
        ldptr_w(rd, rj, simm);
    }

    void uni_ldptr_d(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            ldptr_d(rd, X_TMP_2, 0);
            return;
        }
        ldptr_d(rd, rj, simm);
    }

    void uni_st_b(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            stx_b(rd, rj, X_TMP_2);
            return;
        }
        st_b(rd, rj, simm);
    }

    void uni_st_h(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            stx_h(rd, rj, X_TMP_2);
            return;
        }
        st_h(rd, rj, simm);
    }

    void uni_st_w(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            stx_w(rd, rj, X_TMP_2);
            return;
        }
        st_w(rd, rj, simm);
    }

    void uni_st_d(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            stx_d(rd, rj, X_TMP_2);
            return;
        }
        st_d(rd, rj, simm);
    }

    void uni_fst_s(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            fstx_s(rd, rj, X_TMP_2);
            return;
        }
        fst_s(rd, rj, simm);
    }

    void uni_fst_d(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            fstx_d(rd, rj, X_TMP_2);
            return;
        }
        fst_d(rd, rj, simm);
    }

    void uni_sc_w(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            sc_w(rd, X_TMP_2, 0);
            return;
        }
        sc_w(rd, rj, simm);
    }

    void uni_sc_d(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            sc_d(rd, X_TMP_2, 0);
            return;
        }
        sc_d(rd, rj, simm);
    }

    void uni_stptr_w(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            stptr_w(rd, X_TMP_2, 0);
            return;
        }
        stptr_w(rd, rj, simm);
    }

    void uni_stptr_d(const Xbyak_loongarch::XReg &rd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            stptr_d(rd, X_TMP_2, 0);
            return;
        }
        stptr_d(rd, rj, simm);
    }

    void uni_xvld(const Xbyak_loongarch::XVReg &xd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            xvldx(xd, rj, X_TMP_2);
            return;
        }
        xvld(xd, rj, simm);
    }

    void uni_xvst(const Xbyak_loongarch::XVReg &xd, const Xbyak_loongarch::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            xvstx(xd, rj, X_TMP_2);
            return;
        }
        xvst(xd, rj, simm);
    }

    void uni_vpxor(const Xbyak_loongarch::XVReg &xd, const Xbyak_loongarch::XVReg &xj,
            const Xbyak_loongarch::XVReg &xk) {
        xvxor_v(xd, xj, xk);
    }

    /*
      Saturation facility functions. enable to prepare the register
      holding the saturation upperbound and apply the saturation on
      the floating point register
     */
    template <typename Vmm>
    void init_saturate_f32(Vmm vmm_lbound, Vmm vmm_ubound, Xbyak_loongarch::XReg &reg_tmp,
            data_type_t idt, data_type_t odt) {
        using namespace data_type;
        if (!((idt == f32) && utils::one_of(odt, u8, s8, s32))) return;

        assert(IMPLICATION(
                idt == u8, vmm_lbound.getIdx() != vmm_ubound.getIdx()));
        // No need to saturate on lower bound for signed integer types, as
        // the conversion to int would return INT_MIN, and then proper
        // saturation will happen in store_data
        if (odt == u8) uni_vpxor(vmm_lbound, vmm_lbound, vmm_lbound);

        float saturation_ubound = types::max_value<float>(odt);
        mov_imm(reg_tmp, float2int(saturation_ubound));
	xvreplgr2vr_w(vmm_ubound, reg_tmp);
    }

    template <typename Vmm>
    void saturate_f32(const Vmm &vmm, const Vmm &vmm_lbound,
            const Vmm &vmm_ubound, data_type_t odt) {
        // This function is used to saturate to odt in f32 before converting
        // to s32 in order to avoid bad saturation due to cvtps2dq
        // behavior (it returns INT_MIN if the f32 is out of the
        // s32 range)
        using namespace data_type;
        if (!utils::one_of(odt, u8, s8, s32)) return;

        // no need to apply lower saturation bound when odt is
        // signed, as cvtps2dq will return MIN_INT if the value
        // does not fit
        if (odt == u8) {
            if (is_valid_isa(lasx))
                //vmaxps(vmm, vmm, vmm_lbound);
		xvfmax_s(vmm, vmm, vmm_lbound);
            else
                //maxps(vmm, vmm_lbound);
		xvfmax_s(vmm, vmm, vmm_lbound);
        }
        if (is_valid_isa(lasx))
            //vminps(vmm, vmm, vmm_ubound);
	    xvfmin_s(vmm, vmm, vmm_ubound);
        else
            //minps(vmm, vmm_ubound);
	    xvfmin_s(vmm, vmm, vmm_ubound);
    }

    /**
    * load_bytes is the utility function to facilitate loading of
    * load_size (0 <= load_size <= 32) many contiguous bytes into the Xmm/Ymm
    * register from the memory referenced by ptr[reg + offset] address.
    *
    * Functionally, invocation of load_bytes is equivalent to
    * the following loop:
    *
    * for (int idx = 0; idx < load_size; ++idx)
    *     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
    *
    * TODO: Add an option to zero-out unloaded bytes in the Xmm register.
    * TODO: Add an option for unsafe_load wherein one could read outside the
    * provided memory buffer so as to minimize the total number of read
    * memory instructions.
    */
    template <typename Vmm>
    void load_bytes(const Vmm &vmm, const Xbyak_loongarch::XReg &reg, int64_t offset,
            int load_size) {

        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        // Ensure data fits completely inside the Xmm/Ymm register
        assert(load_size >= 0 && load_size <= 32);

        assert(is_valid_isa(lasx)
                && "routine is not supported for the current isa");

        auto xvreg = Xbyak_loongarch::XVReg(vmm.getIdx());

        switch (load_size) {
            case 0: break;
            case 1:
	    case 2:
	    case 3:
	    case 4:
            case 5:
            case 6:
            case 7:
            case 8: 
                uni_ld_d(X_TMP_3, reg, offset);
                xvinsgr2vr_d(xvreg, X_TMP_3, 0); 
		break;
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
                uni_ld_d(X_TMP_3, reg, offset);
                xvinsgr2vr_d(xvreg, X_TMP_3, 0);
		uni_ld_d(X_TMP_3, reg, offset + 8);
                xvinsgr2vr_d(xvreg, X_TMP_3, 1);
	        break;
	    case 17:
            case 18:
            case 19:
            case 20:
            case 21:
            case 22:
            case 23:
            case 24:
		uni_ld_d(X_TMP_3, reg, offset);
                xvinsgr2vr_d(xvreg, X_TMP_3, 0);
                uni_ld_d(X_TMP_3, reg, offset + 8);
                xvinsgr2vr_d(xvreg, X_TMP_3, 1);
		uni_ld_d(X_TMP_3, reg, offset + 16);
                xvinsgr2vr_d(xvreg, X_TMP_3, 2);
            case 32:
            default:
                uni_xvld(xvreg, reg, offset);
		break;
        }
    }

    /**
    * store_bytes is the utility function to facilitate storing of
    * store_size (0 <= store_size <= 32) many contiguous bytes from the Xmm/Ymm
    * register into the memory referenced by ptr[reg + offset] address.
    *
    * Additionally, when store_size > 16, the input Ymm register will not be
    * preserved due to the usage of vextracti128 instruction.
    *
    * Functionally, invocation of store_bytes is equivalent
    * to the following loop:
    *
    * for (int idx = 0; idx < store_size; ++idx)
    *     vpextrb(ptr[reg + offset + idx], xmm, idx);
    *
    * TODO: Add an option for unsafe_store wherein one could store extra dwords
    * past the provided memory buffer so as to minimize the total number of
    * write memory instructions.
    */
public:
    template <typename Vmm>
    void store_bytes(const Vmm &vmm, const Xbyak_loongarch::XReg &reg, int64_t offset,
            int store_size) {

        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        // Ensure data fits completely inside the Xmm/Ymm register
        assert(store_size >= 0 && store_size <= 32);

        assert(is_valid_isa(lasx)
                && "routine is not supported for the current isa");

	auto xvreg = Xbyak_loongarch::XVReg(vmm.getIdx());

        switch (store_size) {
            case 0: break;
            case 1:
	    case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                xvpickve2gr_d(X_TMP_3, xvreg, 0);
		uni_st_d(X_TMP_3, reg, offset);
		break;
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
		xvpickve2gr_d(X_TMP_3, xvreg, 0);
                uni_st_d(X_TMP_3, reg, offset);
		xvpickve2gr_d(X_TMP_3, xvreg, 1);
                uni_st_d(X_TMP_3, reg, offset + 8);
	      	break;
            case 17:
            case 18:
	    case 19:
	    case 20:
	    case 21:
            case 22:
	    case 23:
	    case 24:
		xvpickve2gr_d(X_TMP_3, xvreg, 0);
                uni_st_d(X_TMP_3, reg, offset);
                xvpickve2gr_d(X_TMP_3, xvreg, 1);
                uni_st_d(X_TMP_3, reg, offset + 8);
		xvpickve2gr_d(X_TMP_3, xvreg, 2);
                uni_st_d(X_TMP_3, reg, offset + 16);
                break;
	    case 32:
            default:
		uni_xvst(xvreg, reg, offset);
        }
    }

public:
    /**
    * load_bytes_to_dword_extension is the utility function to facilitate
    * loading of load_size (0 <= load_size <= 16) many contiguous bytes in
    * the Xmm register from the memory referenced by ptr[reg + offset]
    * address and then do signed/zero extension of those to double words.
    *
    * Functionally, invocation of load_bytes_to_dword_extension is equivalent
    * to the following:
    *
    * for (int idx = 0; idx < load_size; ++idx)
    *     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
    * if (is_signed) vpmovsxbd(vmm, vmm); else vpmovzxbd(vmm, vmm);
    *
    * Valid values for the load_size variable are:
    * [0..4] for XMM version of the function
    * [0..8] for YMM version of the function.
    * TODO: Implement this routine for every ISA.
    */
    template <typename Vmm>
    void load_bytes_to_dword_extension(const Vmm &vmm, const Xbyak_loongarch::XReg &reg,
            int64_t offset, bool is_signed, int load_size) {
        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        // Ensure extended double words fit inside Ymm (32 * load_size <= 256)
        assert(load_size >= 0 && load_size <= 8);

        assert(is_valid_isa(lasx)
                && "routine is not supported for the current isa");
	for (int32_t i = 0; i < load_size; ++i) {
            if (is_signed)
                uni_ld_b(X_TMP_1, reg, i);
            else
                uni_ld_bu(X_TMP_1, reg, i);
            xvinsgr2vr_w(vmm, reg, i);
        }
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator);

public:
    /* All uni_ instructions -- apart from uni_vzeroupper() -- will comply with
     * the max_cpu_isa argument */
    jit_generator(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true, cpu_isa_t max_cpu_isa = isa_all)
        : Xbyak_loongarch::CodeGenerator(code_size,
                (code_ptr == nullptr && use_autogrow) ? Xbyak_loongarch::AutoGrow
                                                      : code_ptr)
        , max_cpu_isa_(max_cpu_isa) {}

    virtual ~jit_generator() {}

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    void register_jit_code(const uint8_t *code, size_t code_size) const {
        jit_utils::register_jit_code(code, code_size, name(), source_file());
    }

    const uint8_t *jit_ker() const { return jit_ker_; }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual status_t create_kernel() {
        generate();
        jit_ker_ = getCode();
        return (jit_ker_) ? status::success : status::runtime_error;
    }

private:
    const cpu_isa_t max_cpu_isa_;
    const uint8_t *getCode() {
        this->ready();
        if (!is_initialized()) return nullptr;
        const uint8_t *code = CodeGenerator::getCode();
        register_jit_code(code, getSize());
        return code;
    }

    inline bool is_valid_isa(cpu_isa_t isa) {
        return mayiuse(isa);
    }

    static inline bool is_initialized() {
        //return Xbyak_loongarch::GetError() == Xbyak_loongarch::ERR_NONE;
	/* At the moment, Xbyak_loongarch does not have GetError()\
         so that return dummy result. */
	return true;
    }

protected:
    virtual void generate() = 0;
    const uint8_t *jit_ker_ = nullptr;
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
