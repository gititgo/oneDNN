#include "mkl_dnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkl_dnn.hpp"

namespace mkl_dnn {

struct conv_any_fmt_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    convolution::algorithm aalgorithm;
    memory::format src_fmt_in;
    memory::format src_fmt_exp;
    memory::format weights_fmt_in;
    memory::format weights_fmt_exp;
    memory::format bias_fmt_in;
    memory::format bias_fmt_exp;
    memory::format dst_fmt_in;
    memory::format dst_fmt_exp;
    test_convolution_descr_t test_cd;
};

template <typename data_t>
class convolution_any_fmt_test
        : public ::testing::TestWithParam<conv_any_fmt_test_params> {
protected:
    virtual void SetUp()
    {
        conv_any_fmt_test_params p = ::testing::
                TestWithParam<conv_any_fmt_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu
                || p.engine_kind == engine::kind::cpu_lazy);
        ASSERT_EQ(p.aprop_kind, prop_kind::forward);
        ASSERT_EQ(p.aalgorithm, convolution::direct);
        auto eng = engine(p.engine_kind, 0);
        memory::precision prec = data_traits<data_t>::prec;
        ASSERT_EQ(prec, mkl_dnn::memory::precision::f32);
        const memory::format any_fmt = memory::format::any;

        // Some format chekers
        ASSERT_NE(p.src_fmt_exp, any_fmt);
        ASSERT_NE(p.weights_fmt_exp, any_fmt);
        ASSERT_NE(p.bias_fmt_exp, any_fmt);
        ASSERT_NE(p.dst_fmt_exp, any_fmt);
        ASSERT_TRUE(p.src_fmt_in == any_fmt || p.src_fmt_in == p.src_fmt_exp);
        ASSERT_TRUE(
                p.weights_fmt_in == any_fmt || p.src_fmt_in == p.src_fmt_exp);
        ASSERT_TRUE(p.bias_fmt_in == any_fmt || p.src_fmt_in == p.src_fmt_exp);
        ASSERT_TRUE(p.dst_fmt_in == any_fmt || p.src_fmt_in == p.src_fmt_exp);

        test_convolution_descr_t cd = p.test_cd;

        auto c_src_desc
                = create_md({ cd.mb, cd.ic, cd.ih, cd.iw }, prec, p.src_fmt_in);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        prec, p.weights_fmt_in) :
                create_md(
                        { cd.oc, cd.ic, cd.kh, cd.kw }, prec, p.weights_fmt_in);
        auto c_dst_desc
                = create_md({ cd.mb, cd.oc, cd.oh, cd.ow }, prec, p.dst_fmt_in);

        bool with_bias = p.bias_fmt_in != memory::format::format_undef;
        auto c_bias_desc = with_bias ?
                create_md({ cd.oc }, prec, p.bias_fmt_in) :
                create_md({}, prec, p.bias_fmt_in);

        auto conv_desc = with_bias ?
                convolution::desc(p.aprop_kind, p.aalgorithm, c_src_desc,
                        c_weights_desc, c_bias_desc, c_dst_desc,
                        { cd.strh, cd.strw }, { cd.padh, cd.padw },
                        padding_kind::zero) :
                convolution::desc(p.aprop_kind, p.aalgorithm, c_src_desc,
                        c_weights_desc, c_dst_desc, { cd.strh, cd.strw },
                        { cd.padh, cd.padw }, padding_kind::zero);

        auto conv_prim_desc = convolution::primitive_desc(conv_desc, eng);
        ASSERT_EQ(conv_prim_desc.data.src_primitive_desc.memory_desc.format,
                memory::convert_to_c(p.src_fmt_exp));
        ASSERT_EQ(conv_prim_desc.data.weights_primitive_desc.memory_desc.format,
                memory::convert_to_c(p.weights_fmt_exp));
        if (with_bias)
            ASSERT_EQ(
                    conv_prim_desc.data.bias_primitive_desc.memory_desc.format,
                    memory::convert_to_c(p.bias_fmt_exp));
        ASSERT_EQ(conv_prim_desc.data.dst_primitive_desc.memory_desc.format,
                memory::convert_to_c(p.dst_fmt_exp));
    }
};

using conv_any_fmt_test_float = convolution_any_fmt_test<float>;
using conv_any_fmt_test_params_float = conv_any_fmt_test_params;

TEST_P(conv_any_fmt_test_float, TestsConvolutionAnyFmt)
{
}
INSTANTIATE_TEST_CASE_P(TestConvolutionAnyFmtForward, conv_any_fmt_test_float,
        ::testing::Values(conv_any_fmt_test_params_float{ prop_kind::forward,
                engine::kind::cpu, convolution::direct, memory::format::any,
                memory::format::nchw, memory::format::any, memory::format::oihw,
                memory::format::any, memory::format::x, memory::format::any,
                memory::format::nchw,
                { 2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionAlexnetAnyFmtForwardBlocked, conv_any_fmt_test_float,
        ::testing::Values(
                conv_any_fmt_test_params_float{ prop_kind::forward,
                        engine::kind::cpu, convolution::direct,
                        memory::format::any, memory::format::nChw8c,
                        memory::format::any, memory::format::gOIhw8i8o,
                        memory::format::any, memory::format::x,
                        memory::format::any, memory::format::nChw8c,
                        { 2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1 } },
                conv_any_fmt_test_params_float{ prop_kind::forward,
                        engine::kind::cpu, convolution::direct,
                        memory::format::any, memory::format::nChw8c,
                        memory::format::any, memory::format::OIhw8i8o,
                        memory::format::any, memory::format::x,
                        memory::format::any, memory::format::nChw8c,
                        { 2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_any_fmt_test_params_float{ prop_kind::forward,
                        engine::kind::cpu, convolution::direct,
                        memory::format::any, memory::format::nChw8c,
                        memory::format::any, memory::format::gOIhw8i8o,
                        memory::format::any, memory::format::x,
                        memory::format::any, memory::format::nChw8c,
                        { 2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_any_fmt_test_params_float{ prop_kind::forward,
                        engine::kind::cpu, convolution::direct,
                        memory::format::any, memory::format::nChw8c,
                        memory::format::any, memory::format::gOIhw8i8o,
                        memory::format::any, memory::format::x,
                        memory::format::any, memory::format::nChw8c,
                        { 2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1,
                                1 } }));
}
