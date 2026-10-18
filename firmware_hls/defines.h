#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<8,5,AP_RND,AP_WRAP,0> input_t;
typedef ap_fixed<16,8,AP_RND,AP_WRAP,0> dense_18_accum_t;
typedef ap_fixed<8,5,AP_RND,AP_WRAP,0> layer2_t;
typedef ap_fixed<8,1,AP_RND,AP_WRAP,0> dense_18_weight_t;
typedef ap_fixed<8,1,AP_RND,AP_WRAP,0> dense_18_bias_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<8,5,AP_RND,AP_WRAP,0> layer3_t;
typedef ap_fixed<18,8> dense_18_relu_table_t;
typedef ap_fixed<16,8,AP_RND,AP_WRAP,0> dense_19_accum_t;
typedef ap_fixed<8,5,AP_RND,AP_WRAP,0> layer5_t;
typedef ap_fixed<8,1,AP_RND,AP_WRAP,0> dense_19_weight_t;
typedef ap_fixed<8,1,AP_RND,AP_WRAP,0> dense_19_bias_t;
typedef ap_uint<1> layer5_index;
typedef ap_fixed<8,5,AP_RND,AP_WRAP,0> layer6_t;
typedef ap_fixed<18,8> dense_19_relu_table_t;
typedef ap_fixed<16,8,AP_RND,AP_WRAP,0> dense_20_accum_t;
typedef ap_fixed<12,5> layer7_t;
typedef ap_fixed<8,2,AP_RND,AP_WRAP,0> dense_20_weight_t;
typedef ap_fixed<8,2,AP_RND,AP_WRAP,0> dense_20_bias_t;
typedef ap_uint<1> layer7_index;
typedef ap_fixed<12,5> result_t;
typedef ap_fixed<18,8> dense_20_sigmoid_table_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
