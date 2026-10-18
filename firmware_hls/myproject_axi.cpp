#include "myproject_axi.h"

void myproject_axi(
    float in_r[11],
    float out_r[1]
) {
    // Memory-mapped AXI-Lite registers (Table IV in the paper)
    #pragma HLS INTERFACE s_axilite port=return bundle=BUS_A
    #pragma HLS INTERFACE s_axilite port=in_r   bundle=BUS_A
    #pragma HLS INTERFACE s_axilite port=out_r  bundle=BUS_A

    input_t  in_local[11];
    result_t out_local[1];

    // Hardware float32 to fixed-point conversion
    for (int i = 0; i < 11; i++) {
        #pragma HLS PIPELINE
        in_local[i] = (input_t)in_r[i];
    }

    // Call the 8-bit neural network core
    myproject(in_local, out_local);

    // Hardware fixed-point to float32 conversion
    out_r[0] = (float)out_local[0];
}
