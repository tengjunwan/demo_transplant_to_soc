/*
  Copyright (c), 2001-2022, Shenshu Tech. Co., Ltd.
 */

#ifndef SAMPLE_NPU_PROCESS_H
#define SAMPLE_NPU_PROCESS_H

#include "ot_type.h"
#include <stddef.h>

#include "detectobjs.h"

td_s32 sample_svp_npu_acl_prepare_init();
td_s32 sample_svp_npu_load_model(const char* om_model_path, td_u32 model_index, td_bool is_cached);
td_void sample_svp_npu_acl_prepare_exit(td_u32 thread_num);
td_s32 sample_svp_npu_dataset_prepare_init(td_u32 model_index);
td_s32 sample_svp_npu_create_input_databuf(td_void *data_buf, size_t data_len, td_u32 model_index);
td_s32 sample_svp_npu_create_input_databuf_v2(td_void *data1_buf, size_t data1_len, 
                                              td_void *data2_buf, size_t data2_len, 
                                              td_u32 model_index);
td_void sample_svp_npu_model_link_buffer(td_u32 model1_index, 
                                         td_u32 model2_index, 
                                         td_u32 model3_index);
// td_void sample_svp_npu_output_model_result_head(td_u32 model_index, 
//                                             stmTrackerState *state);
td_s32 sample_svp_npu_model_execute(td_u32 model_index);
td_void sample_svp_npu_destroy_output(td_u32 model_index);
td_void sample_svp_npu_destroy_input_dataset(td_u32 model_index);

#endif
