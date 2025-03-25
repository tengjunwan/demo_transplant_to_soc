/*
  Copyright (c), 2001-2022, Shenshu Tech. Co., Ltd.
 */

#ifndef SAMPLE_NPU_MODEL_H
#define SAMPLE_NPU_MODEL_H
#include <stddef.h>
#include "ot_type.h"
#include "acl.h"
#include "ss_mpi_sys.h"
#include "detectobjs.h"

#define MAX_THREAD_NUM 20
#define MAX_INPUT_NUM 5
#define MAX_OUTPUT_NUM 5

typedef struct npu_acl_model {
    td_u32 model_id;
    td_ulong model_mem_size;
    td_ulong model_weight_size;
    td_void *model_mem_ptr;
    td_void *model_weight_ptr;
    td_phys_addr_t model_mem_phy_addr;
    td_phys_addr_t model_weight_phy_addr;
    td_bool is_load_flag;
    aclmdlDesc *model_desc;
    aclmdlDataset *input_dataset;
    aclmdlDataset *output_dataset;
    td_phys_addr_t output_phy_addr[MAX_OUTPUT_NUM];
    td_phys_addr_t input_phy_addr[MAX_INPUT_NUM];
} npu_acl_model_t;

td_s32 sample_npu_load_model_with_mem(const char *model_path, td_u32 model_index);

td_s32 sample_npu_create_desc(td_u32 model_index);
td_void sample_npu_destroy_desc(td_u32 model_index);

td_s32 sample_npu_create_input_dataset(td_u32 model_index);
td_void sample_npu_destroy_input_dataset(td_u32 model_index);
td_s32 sample_npu_create_input_databuf(td_void *data_buf, size_t data_len, td_u32 model_index);
td_void sample_npu_destroy_input_databuf(td_u32 model_index);

td_s32 sample_npu_create_input_databuf_v2(td_void *data1_buf, size_t data1_len, 
                                          td_void *data2_buf, size_t data2_len, 
                                          td_u32 model_index);
td_void sample_npu_model_link_buffer(td_u32 model1_index, 
                                     td_u32 model2_index, 
                                     td_u32 model3_index);
td_void sample_npu_output_model_result_head(td_u32 model_index, 
                                            const stmTrackerState *state, stmTrackerState *result_state);

void printModelIODescription(td_u32 model_index);

td_s32 sample_npu_create_output(td_u32 model_index);
td_void sample_npu_output_model_result(td_u32 model_index);
td_void sample_npu_destroy_output(td_u32 model_index);

td_s32 sample_npu_model_execute(td_u32 model_index);

td_void sample_npu_unload_model(td_u32 model_index);

td_s32 prepare_for_siamfcpp_execution(void);
td_s32 prepare_for_template_execute(td_s32 model_index, td_void* template_im_buf, size_t template_im_len);
td_void cleanup_for_template_execute(td_s32 model_index);
td_s32 prepare_for_search_execute(td_s32 model_index, void* search_im_buf, size_t search_im_len);
td_void siamfcpp_postprocess(const stmTrackerState* state, stmTrackerState* result_state);

#endif
