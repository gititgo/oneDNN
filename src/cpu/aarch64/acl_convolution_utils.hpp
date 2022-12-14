/*******************************************************************************
* Copyright 2020-2021 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP
#define CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP

#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <typename NEConv>
struct acl_obj_t {
    NEConv conv;
    arm_compute::NEArithmeticAddition add;
    arm_compute::NEActivationLayer act;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
    arm_compute::Tensor dst_acc_tensor;
};

struct acl_conv_conf_t {
    bool with_bias;
    bool is_int8;
    bool sum_with_eltwise;
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo wei_info;
    arm_compute::TensorInfo bia_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::PadStrideInfo padstride_info;
    arm_compute::Size2D dilation_info;
    arm_compute::WeightsInfo weights_info;
    arm_compute::ActivationLayerInfo act_info;
};

namespace acl_convolution_utils {

status_t init_conf_gemm(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

status_t init_conf_indirect_gemm(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

status_t init_conf_wino(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

} // namespace acl_convolution_utils

template <typename conv_obj_t, typename conv_pd_t, typename src_data_t,
        typename wei_data_t = src_data_t, typename dst_data_t = src_data_t,
        typename bia_data_t = src_data_t>
status_t execute_forward_conv_acl(
        const exec_ctx_t &ctx, conv_obj_t &acl_conv_obj, const conv_pd_t *pd) {
    bool with_bias = pd->acp_.with_bias;
    bool sum_with_eltwise = pd->acp_.sum_with_eltwise;

    auto src_base = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto dst_base = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    // import_memory() and free() methods do not allocate/free any additional
    // memory, only acquire/release pointers.
    acl_conv_obj.src_tensor.allocator()->import_memory(
            const_cast<src_data_t *>(src_base));
    acl_conv_obj.wei_tensor.allocator()->import_memory(
            const_cast<wei_data_t *>(wei_base));
    acl_conv_obj.dst_tensor.allocator()->import_memory(dst_base);

    if (with_bias) {
        auto bia_base = CTX_IN_MEM(const bia_data_t *, DNNL_ARG_BIAS);
        acl_conv_obj.bia_tensor.allocator()->import_memory(
                const_cast<bia_data_t *>(bia_base));
    }

    acl_conv_obj.conv.run();

    if (sum_with_eltwise) {
        acl_conv_obj.add.run();
        acl_conv_obj.act.run();
    }

    acl_conv_obj.src_tensor.allocator()->free();
    acl_conv_obj.wei_tensor.allocator()->free();
    acl_conv_obj.dst_tensor.allocator()->free();
    if (with_bias) { acl_conv_obj.bia_tensor.allocator()->free(); }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP
