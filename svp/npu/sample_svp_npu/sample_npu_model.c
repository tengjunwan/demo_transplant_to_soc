/*
  Copyright (c), 2001-2022, Shenshu Tech. Co., Ltd.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <semaphore.h>
#include <pthread.h>
#include <math.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ot_common_svp.h"
#include "sample_common_svp.h"
#include "sample_npu_model.h"

static npu_acl_model_t g_npu_acl_model[MAX_THREAD_NUM] = {0};
// ========================siamfcpp=========================
// template output buffer
static void* c_z_k = TD_NULL;
static void* r_z_k = TD_NULL;
static aclDataBuffer* data_c_z_k = TD_NULL;
static aclDataBuffer* data_r_z_k = TD_NULL;
// search output buffer
static void* score = TD_NULL;
static void* bbox = TD_NULL;
static aclDataBuffer* data_score = TD_NULL;
static aclDataBuffer* data_bbox = TD_NULL;

td_s32 sample_npu_load_model_with_mem(const char *model_path, td_u32 model_index) {
    if (g_npu_acl_model[model_index].is_load_flag) {
        sample_svp_trace_err("has already loaded a model\n");
        return TD_FAILURE;
    }

    td_s32 ret = aclmdlQuerySize(model_path, &g_npu_acl_model[model_index].model_mem_size,
        &g_npu_acl_model[model_index].model_weight_size);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("query model failed, model file is %s\n", model_path);
        return TD_FAILURE;
    }

    ret = aclrtMalloc(&g_npu_acl_model[model_index].model_mem_ptr, g_npu_acl_model[model_index].model_mem_size,
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("malloc buffer for mem failed, require size is %lu\n",
            g_npu_acl_model[model_index].model_mem_size);
        return TD_FAILURE;
    }

    ret = aclrtMalloc(&g_npu_acl_model[model_index].model_weight_ptr, g_npu_acl_model[model_index].model_weight_size,
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("malloc buffer for weight fail, require size is %lu\n",
            g_npu_acl_model[model_index].model_weight_size);
        return TD_FAILURE;
    }

    ret = aclmdlLoadFromFileWithMem(model_path, &g_npu_acl_model[model_index].model_id,
        g_npu_acl_model[model_index].model_mem_ptr, g_npu_acl_model[model_index].model_mem_size,
        g_npu_acl_model[model_index].model_weight_ptr, g_npu_acl_model[model_index].model_weight_size);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("load model from file failed, model file is %s\n", model_path);
        return TD_FAILURE;
    }

    sample_svp_trace_info("load mem_size:%lu weight_size:%lu id:%d\n", g_npu_acl_model[model_index].model_mem_size,
        g_npu_acl_model[model_index].model_weight_size, g_npu_acl_model[model_index].model_id);

    g_npu_acl_model[model_index].is_load_flag = TD_TRUE;
    sample_svp_trace_info("load model %s success\n", model_path);

    return TD_SUCCESS;
}

td_s32 sample_npu_load_model_with_mem_cached(const char *model_path, td_u32 model_index) {
    if (g_npu_acl_model[model_index].is_load_flag) {
        sample_svp_trace_err("has already loaded a model\n");
        return TD_FAILURE;
    }

    td_s32 ret = aclmdlQuerySize(model_path, &g_npu_acl_model[model_index].model_mem_size,
        &g_npu_acl_model[model_index].model_weight_size);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("query model failed, model file is %s\n", model_path);
        return TD_FAILURE;
    }

    ret = ss_mpi_sys_mmz_alloc_cached(&g_npu_acl_model[model_index].model_mem_phy_addr,
        &g_npu_acl_model[model_index].model_mem_ptr, "model_mem", NULL, g_npu_acl_model[model_index].model_mem_size);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("malloc buffer for mem failed\n");
        return TD_FAILURE;
    }
    memset_s(g_npu_acl_model[model_index].model_mem_ptr, g_npu_acl_model[model_index].model_mem_size, 0,
        g_npu_acl_model[model_index].model_mem_size);
    ss_mpi_sys_flush_cache(g_npu_acl_model[model_index].model_mem_phy_addr, g_npu_acl_model[model_index].model_mem_ptr,
        g_npu_acl_model[model_index].model_mem_size);

    ret = ss_mpi_sys_mmz_alloc_cached(&g_npu_acl_model[model_index].model_weight_phy_addr,
        &g_npu_acl_model[model_index].model_weight_ptr, "model_weight",
        NULL, g_npu_acl_model[model_index].model_weight_size);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("malloc buffer for weight fail\n");
        return TD_FAILURE;
    }
    memset_s(g_npu_acl_model[model_index].model_weight_ptr, g_npu_acl_model[model_index].model_weight_size, 0,
        g_npu_acl_model[model_index].model_weight_size);
    ss_mpi_sys_flush_cache(g_npu_acl_model[model_index].model_weight_phy_addr,
        g_npu_acl_model[model_index].model_weight_ptr, g_npu_acl_model[model_index].model_weight_size);

    ret = aclmdlLoadFromFileWithMem(model_path, &g_npu_acl_model[model_index].model_id,
        g_npu_acl_model[model_index].model_mem_ptr, g_npu_acl_model[model_index].model_mem_size,
        g_npu_acl_model[model_index].model_weight_ptr, g_npu_acl_model[model_index].model_weight_size);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("load model from file failed, model file is %s\n", model_path);
        return TD_FAILURE;
    }

    sample_svp_trace_info("load mem_size:%lu weight_size:%lu id:%d\n", g_npu_acl_model[model_index].model_mem_size,
        g_npu_acl_model[model_index].model_weight_size, g_npu_acl_model[model_index].model_id);

    g_npu_acl_model[model_index].is_load_flag = TD_TRUE;
    sample_svp_trace_info("load model %s success\n", model_path);

    return TD_SUCCESS;
}

td_s32 sample_npu_create_desc(td_u32 model_index) {
    td_s32 ret;

    g_npu_acl_model[model_index].model_desc = aclmdlCreateDesc();
    if (g_npu_acl_model[model_index].model_desc == TD_NULL) {
        sample_svp_trace_err("create model description failed\n");
        return TD_FAILURE;
    }

    ret = aclmdlGetDesc(g_npu_acl_model[model_index].model_desc, g_npu_acl_model[model_index].model_id);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("get model description failed\n");
        return TD_FAILURE;
    }

    sample_svp_trace_info("create model description success\n");

    return TD_SUCCESS;
}

td_void sample_npu_destroy_desc(td_u32 model_index) {
    if (g_npu_acl_model[model_index].model_desc != TD_NULL) {
        (td_void)aclmdlDestroyDesc(g_npu_acl_model[model_index].model_desc);
        g_npu_acl_model[model_index].model_desc = TD_NULL;
    }

    sample_svp_trace_info("destroy model description success\n");
}

td_s32 sample_npu_create_input_dataset(td_u32 model_index) {
    /* om used in this sample has only one input */
    if (g_npu_acl_model[model_index].model_desc == TD_NULL) {
        sample_svp_trace_err("no model description, create input failed\n");
        return TD_FAILURE;
    }

    g_npu_acl_model[model_index].input_dataset = aclmdlCreateDataset();
    if (g_npu_acl_model[model_index].input_dataset == TD_NULL) {
        sample_svp_trace_err("can't create dataset, create input failed\n");
        return TD_FAILURE;
    }

    // sample_svp_trace_info("create model input dataset success\n");
    return TD_SUCCESS;
}

td_void sample_npu_destroy_input_dataset(td_u32 model_index) {
    if (g_npu_acl_model[model_index].input_dataset == TD_NULL) {
        return;
    }

    aclmdlDestroyDataset(g_npu_acl_model[model_index].input_dataset);
    g_npu_acl_model[model_index].input_dataset = TD_NULL;

    // sample_svp_trace_info("destroy model input dataset success\n");
}

td_s32 sample_npu_create_input_databuf(td_void *data_buf, size_t data_len, td_u32 model_index) {
    /* om used in this sample has only one input */
    if (g_npu_acl_model[model_index].model_desc == TD_NULL) {
        sample_svp_trace_err("no model description, create input failed\n");
        return TD_FAILURE;
    }

    size_t input_size = aclmdlGetInputSizeByIndex(g_npu_acl_model[model_index].model_desc, 0);
    if (data_len != input_size) {
        sample_svp_trace_err("input image size[%zu] != model input size[%zu]\n", data_len, input_size);
        return TD_FAILURE;
    }

    aclDataBuffer *input_data = aclCreateDataBuffer(data_buf, data_len);
    if (input_data == TD_NULL) {
        sample_svp_trace_err("can't create data buffer, create input failed\n");
        return TD_FAILURE;
    }

    aclError ret = aclmdlAddDatasetBuffer(g_npu_acl_model[model_index].input_dataset, input_data);
    if (ret != ACL_SUCCESS) {
        sample_svp_trace_err("add input dataset buffer failed, ret is %d\n", ret);
        (void)aclDestroyDataBuffer(input_data);
        input_data = TD_NULL;
        return TD_FAILURE;
    }
    // sample_svp_trace_info("create model input success\n");

    return TD_SUCCESS;
}

td_s32 sample_npu_create_input_databuf_v2(td_void *data1_buf, size_t data1_len, 
                                          td_void *data2_buf, size_t data2_len, 
                                          td_u32 model_index) {
    aclError ret;
    if (g_npu_acl_model[model_index].model_desc == TD_NULL) {
        sample_svp_trace_err("no model description, create input failed\n");
        return TD_FAILURE;
    }

    // set 1st data buffer
    size_t input1_size = aclmdlGetInputSizeByIndex(g_npu_acl_model[model_index].model_desc, 0);
    if (data1_len != input1_size) {
        sample_svp_trace_err("1st input image size[%zu] != 1st model input size[%zu]\n", data1_len, input1_size);
        return TD_FAILURE;
    }

    aclDataBuffer *input1_data = aclCreateDataBuffer(data1_buf, data1_len);
    if (input1_data == TD_NULL) {
        sample_svp_trace_err("can't create 1st data buffer, create 1st input failed\n");
        return TD_FAILURE;
    }

    ret = aclmdlAddDatasetBuffer(g_npu_acl_model[model_index].input_dataset, input1_data);
    if (ret != ACL_SUCCESS) {
        sample_svp_trace_err("add 1st input dataset buffer failed, ret is %d\n", ret);
        (void)aclDestroyDataBuffer(input1_data);
        input1_data = TD_NULL;
        return TD_FAILURE;
    }

    // set 2nd data buffer
    size_t input2_size = aclmdlGetInputSizeByIndex(g_npu_acl_model[model_index].model_desc, 1);
    if (data2_len != input2_size) {
        sample_svp_trace_err("2nd input image size[%zu] != 2dn model input size[%zu]\n", data2_len, input2_size);
        return TD_FAILURE;
    }

    aclDataBuffer *input2_data = aclCreateDataBuffer(data2_buf, data2_len);
    if (input2_data == TD_NULL) {
        sample_svp_trace_err("can't create 2nd data buffer, create input failed\n");
        return TD_FAILURE;
    }

    ret = aclmdlAddDatasetBuffer(g_npu_acl_model[model_index].input_dataset, input2_data);
    if (ret != ACL_SUCCESS) {
        sample_svp_trace_err("add 2nd input dataset buffer failed, ret is %d\n", ret);
        (void)aclDestroyDataBuffer(input2_data);
        input2_data = TD_NULL;
        return TD_FAILURE;
    }

    // sample_svp_trace_info("create model input success\n");

    return TD_SUCCESS;
}

td_void sample_npu_destroy_input_databuf(td_u32 model_index) {
    td_u32 i;

    if (g_npu_acl_model[model_index].input_dataset == TD_NULL) {
        return;
    }

    for (i = 0; i < aclmdlGetDatasetNumBuffers(g_npu_acl_model[model_index].input_dataset); ++i) {
        aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model_index].input_dataset, i);
        aclDestroyDataBuffer(data_buffer);
    }

    // sample_svp_trace_info("destroy model input data buf success\n");
}

td_s32 sample_npu_create_output(td_u32 model_index) {
    td_u32 output_size;

    if (g_npu_acl_model[model_index].model_desc == TD_NULL) {
        sample_svp_trace_err("no model description, create output failed\n");
        return TD_FAILURE;
    }

    g_npu_acl_model[model_index].output_dataset = aclmdlCreateDataset();
    if (g_npu_acl_model[model_index].output_dataset == TD_NULL) {
        sample_svp_trace_err("can't create dataset, create output failed\n");
        return TD_FAILURE;
    }

    output_size = aclmdlGetNumOutputs(g_npu_acl_model[model_index].model_desc);
    for (td_u32 i = 0; i < output_size; ++i) {
        td_u32 buffer_size = aclmdlGetOutputSizeByIndex(g_npu_acl_model[model_index].model_desc, i);

        td_void *output_buffer = TD_NULL;
        td_s32 ret = aclrtMalloc(&output_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            sample_svp_trace_err("can't malloc buffer, size is %u, create output failed\n", buffer_size);
            return TD_FAILURE;
        }

        aclDataBuffer *output_data = aclCreateDataBuffer(output_buffer, buffer_size);
        if (output_data == TD_NULL) {
            sample_svp_trace_err("can't create data buffer, create output failed\n");
            aclrtFree(output_buffer);
            return TD_FAILURE;
        }

        ret = aclmdlAddDatasetBuffer(g_npu_acl_model[model_index].output_dataset, output_data);
        if (ret != ACL_ERROR_NONE) {
            sample_svp_trace_err("can't add data buffer, create output failed\n");
            aclrtFree(output_buffer);
            aclDestroyDataBuffer(output_data);
            return TD_FAILURE;
        }
    }

    // sample_svp_trace_info("create model output TD_SUCCESS\n");
    return TD_SUCCESS;
}

td_void sample_npu_output_model_result(td_u32 model_index) {
    aclDataBuffer *data_buffer = TD_NULL;
    td_void *data = TD_NULL;
    td_u32 len;
    td_u32 i, j;

    for (i = 0; i < aclmdlGetDatasetNumBuffers(g_npu_acl_model[model_index].output_dataset); ++i) {
        data_buffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model_index].output_dataset, i);
        if (data_buffer == TD_NULL) {
            sample_svp_trace_err("get data buffer null.\n");
            continue;
        }

        data = aclGetDataBufferAddr(data_buffer);
        len = aclGetDataBufferSizeV2(data_buffer);
        if (data == TD_NULL || len == 0) {
            sample_svp_trace_err("get data null.\n");
            continue;
        }
    }

    // sample_svp_trace_info("output data success\n");
    return;
}

td_void sample_npu_destroy_output(td_u32 model_index) {
    if (g_npu_acl_model[model_index].output_dataset == TD_NULL) {
        return;
    }

    for (td_u32 i = 0; i < aclmdlGetDatasetNumBuffers(g_npu_acl_model[model_index].output_dataset); ++i) {
        aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model_index].output_dataset, i);
        td_void *data = aclGetDataBufferAddr(data_buffer);
        (td_void)aclrtFree(data);
        (td_void)aclDestroyDataBuffer(data_buffer);
    }

    (td_void)aclmdlDestroyDataset(g_npu_acl_model[model_index].output_dataset);
    g_npu_acl_model[model_index].output_dataset = TD_NULL;
}

td_s32 sample_npu_model_execute(td_u32 model_index) {
    td_s32 ret;
    ret = aclmdlExecute(g_npu_acl_model[model_index].model_id, g_npu_acl_model[model_index].input_dataset,
        g_npu_acl_model[model_index].output_dataset);
    // sample_svp_trace_info("end aclmdlExecute, modelId is %u\n", g_npu_acl_model[model_index].model_id);
    return ret;
}

td_void sample_npu_unload_model(td_u32 model_index) {
    if (!g_npu_acl_model[model_index].is_load_flag) {
        sample_svp_trace_info("no model had been loaded.\n");
        return;
    }

    td_s32 ret = aclmdlUnload(g_npu_acl_model[model_index].model_id);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("unload model failed, modelId is %u\n", g_npu_acl_model[model_index].model_id);
    }

    if (g_npu_acl_model[model_index].model_desc != TD_NULL) {
        (td_void)aclmdlDestroyDesc(g_npu_acl_model[model_index].model_desc);
        g_npu_acl_model[model_index].model_desc = TD_NULL;
    }

    if (g_npu_acl_model[model_index].model_mem_ptr != TD_NULL) {
        aclrtFree(g_npu_acl_model[model_index].model_mem_ptr);
        g_npu_acl_model[model_index].model_mem_ptr = TD_NULL;
        g_npu_acl_model[model_index].model_mem_size = 0;
    }

    if (g_npu_acl_model[model_index].model_weight_ptr != TD_NULL) {
        aclrtFree(g_npu_acl_model[model_index].model_weight_ptr);
        g_npu_acl_model[model_index].model_weight_ptr = TD_NULL;
        g_npu_acl_model[model_index].model_weight_size = 0;
    }

    g_npu_acl_model[model_index].is_load_flag = TD_FALSE;
    sample_svp_trace_info("unload model SUCCESS, modelId is %u\n", g_npu_acl_model[model_index].model_id);
}

td_void sample_npu_model_link_buffer(td_u32 model1_index, td_u32 model2_index, td_u32 model3_index) {
    aclmdlDataset *output1_dataset, *output2_dataset, *input3_dataset;
    if (g_npu_acl_model[model1_index].output_dataset == TD_NULL) {
        sample_svp_trace_err("link error, model1 has no output dataset");
        return;
    }
    if (g_npu_acl_model[model2_index].output_dataset == TD_NULL) {
        sample_svp_trace_err("link error, model2 has no output dataset");
        return;
    }
    if (g_npu_acl_model[model3_index].input_dataset == TD_NULL) {
        sample_svp_trace_err("link error, model3 has no input dataset");
        return;
    }
    // fetch data buffer
    aclDataBuffer *output1_databuffer, *output2_databuffer, *input3_databuffer;

    // sample_svp_trace_info("link check done0\n");
    output1_databuffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model1_index].output_dataset, 0);
    if (output1_databuffer == TD_NULL) {
        sample_svp_trace_err("link error, model1 already has not output databuffer");
        return;
    }
    // sample_svp_trace_info("link check done1\n");
    output2_databuffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model2_index].output_dataset, 0);
    if (output2_databuffer == TD_NULL) {
        sample_svp_trace_err("link error, model2 already has not output databuffer");
        return;
    }
    // sample_svp_trace_info("link check done2\n");
    // add output databuffers of model1 and model2 to the input dataset of model3
    aclError ret;
    ret = aclmdlAddDatasetBuffer(g_npu_acl_model[model3_index].input_dataset, output2_databuffer); // fm
    if (ret != ACL_SUCCESS) {
        sample_svp_trace_err("add input dataset buffer failed, ret is %d\n", ret);
        (void)aclDestroyDataBuffer(output2_databuffer);
        output2_databuffer = TD_NULL;
        return ;
    }
    ret = aclmdlAddDatasetBuffer(g_npu_acl_model[model3_index].input_dataset, output1_databuffer); // fq
    if (ret != ACL_SUCCESS) {
        sample_svp_trace_err("add input dataset buffer failed, ret is %d\n", ret);
        (void)aclDestroyDataBuffer(output1_databuffer);
        output1_databuffer = TD_NULL;
        return ;
    }
}

td_void sample_npu_output_model_result_head(td_u32 model_index, const stmTrackerState* state, stmTrackerState* result_state) {
    aclDataBuffer* score_buffer = TD_NULL;
    aclDataBuffer* bbox_buffer = TD_NULL;
    td_void* score = TD_NULL;
    td_void* bbox = TD_NULL;
    td_u32 len_score, len_bbox;

    // score 
    score_buffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model_index].output_dataset, 0);
    if (score_buffer == TD_NULL) {
        sample_svp_trace_err("get score buffer null.\n");
    }
    score = aclGetDataBufferAddr(score_buffer);
    len_score = aclGetDataBufferSizeV2(score_buffer);  // size of bytes

    // bbox
    bbox_buffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model_index].output_dataset, 1);
    if (bbox_buffer == TD_NULL) {
        sample_svp_trace_err("get bbox buffer null.\n");
    }
    bbox = aclGetDataBufferAddr(bbox_buffer);
    len_bbox = aclGetDataBufferSizeV2(bbox_buffer);  // size of bytes

    stmTrack_result(score, len_score/sizeof(float), bbox, len_bbox/sizeof(float), state, result_state);

    // sample_svp_trace_info("stmtrack output data success\n");
    return;
}

void printModelIODescription(td_u32 model_index) {
    aclmdlDesc *model_desc = g_npu_acl_model[model_index].model_desc;
    if (model_desc == NULL) {
        sample_svp_trace_err("model[%d] description is NULL.\n", model_index);
        return;
    }
    td_u32 num_inputs = aclmdlGetNumInputs(model_desc);
    td_u32 num_outputs = aclmdlGetNumOutputs(model_desc);

    sample_svp_trace_info("model[%d] has %d inputs and %d outputs\n", model_index, num_inputs, num_outputs);

    // print input details
    for (td_u32 i = 0; i < num_inputs; i++) {
        size_t input_size = aclmdlGetInputSizeByIndex(model_desc, i);
        aclmdlIODims input_dims;
        aclFormat input_format = aclmdlGetInputFormat(model_desc, i);
        aclDataType input_data_type = aclmdlGetInputDataType(model_desc, i);

        aclmdlGetInputDims(model_desc, i, &input_dims);
        
        printf("input %d: size=%zu bytes, format=%d, dtype=%d, shape=[", i, input_size, input_format, input_data_type);
        for (size_t j=0; j < input_dims.dimCount; j++) {
            printf("%ld", input_dims.dims[j]);
            if (j < input_dims.dimCount - 1) printf(", ");
        }
        printf("]\n");
    }

    // print output details
    for (td_u32 i = 0; i < num_outputs; i++) {
        size_t output_size = aclmdlGetOutputSizeByIndex(model_desc, i);
        aclmdlIODims output_dims;
        aclFormat output_format = aclmdlGetOutputFormat(model_desc, i);
        aclDataType output_data_type = aclmdlGetOutputDataType(model_desc, i);

        aclmdlGetOutputDims(model_desc, i, &output_dims);
        
        printf("output %d: size=%zu bytes, format=%d, dtype=%d, shape=[", i, output_size, output_format, output_data_type);
        for (size_t j=0; j < output_dims.dimCount; j++) {
            printf("%ld", output_dims.dims[j]);
            if (j < output_dims.dimCount - 1) printf(", ");
        }
        printf("]\n");
    }
}


td_s32 prepare_for_siamfcpp_execution(void) {
    // ---------------------------------------part 1: pre-allocate reused buffers---------------------------------------
    td_s32 ret;
    td_u32 buffer_size;
    td_u32 dataset_size;
    // sanity check
    dataset_size = aclmdlGetNumOutputs(g_npu_acl_model[0].model_desc);  // template model output: c_z_k, r_z_k
    if (dataset_size !=2) {
        printf("num of model[%d]'s inputs is not equal to 2, prepare_for_siamfcpp_execution failed\n", dataset_size);
        return TD_FAILURE;
    }

    // allocate c_z_k buffer
    buffer_size = aclmdlGetOutputSizeByIndex(g_npu_acl_model[0].model_desc, 0);
    ret = aclrtMalloc(&c_z_k, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("can't malloc buffer(c_z_k), size is %u, prepare_for_siamfcpp_execution failed\n", buffer_size);
        return TD_FAILURE;
    }
    // create c_z_k data buffer
    data_c_z_k = aclCreateDataBuffer(c_z_k, buffer_size);
    if (data_c_z_k == TD_NULL) {
        sample_svp_trace_err("can't create data buffer(c_z_k), prepare_for_siamfcpp_execution failed\n");
        aclrtFree(c_z_k);
        return TD_FAILURE;
    }

    // allocate r_z_k buffer
    buffer_size = aclmdlGetOutputSizeByIndex(g_npu_acl_model[0].model_desc, 1);
    ret = aclrtMalloc(&r_z_k, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("can't malloc buffer(r_z_k), size is %u, prepare_for_siamfcpp_execution failed\n", buffer_size);
        return TD_FAILURE;
    }
    // create query data buffer
    data_r_z_k = aclCreateDataBuffer(r_z_k, buffer_size);
    if (data_r_z_k == TD_NULL) {
        sample_svp_trace_err("can't create data buffer(r_z_k), prepare_for_siamfcpp_execution failed\n");
        aclrtFree(r_z_k);
        return TD_FAILURE;
    }

    // sanity check
    dataset_size = aclmdlGetNumOutputs(g_npu_acl_model[1].model_desc);  // search model output: score, bbox
    if (dataset_size !=2) {
        printf("num of model[%d]'s outputs is not equal to 2, prepare_for_siamfcpp_execution failed\n", dataset_size);
        return TD_FAILURE;
    }
    // allocate score buffer
    buffer_size= aclmdlGetOutputSizeByIndex(g_npu_acl_model[1].model_desc, 0);
    ret = aclrtMalloc(&score, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("can't malloc buffer(score), size is %u, prepare_for_siamfcpp_execution failed\n", buffer_size);
        return TD_FAILURE;
    }
    // create score data buffer
    data_score = aclCreateDataBuffer(score, buffer_size);
    if (data_score == TD_NULL) {
        sample_svp_trace_err("can't create data buffer(score), prepare_for_siamfcpp_execution failed\n");
        aclrtFree(score);
        return TD_FAILURE;
    }
    // allocate bbox buffer
    buffer_size = aclmdlGetOutputSizeByIndex(g_npu_acl_model[1].model_desc, 1);
    ret = aclrtMalloc(&bbox, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("can't malloc buffer(bbox), size is %u, prepare_for_siamfcpp_execution failed\n", buffer_size);
        return TD_FAILURE;
    }
    // create bbox data buffer
    data_bbox = aclCreateDataBuffer(bbox, buffer_size);
    if (data_bbox == TD_NULL) {
        sample_svp_trace_err("can't create data buffer(bbox), prepare_for_siamfcpp_execution failed\n");
        aclrtFree(bbox);
        return TD_FAILURE;
    }
    
    // ---------------------------------------part 2: set dataset---------------------------------------
    // unchanged dataset: 1)output dataset of template; 2)output dataset of search model;
    // 1)output dataset of template model                                     
    g_npu_acl_model[0].output_dataset = aclmdlCreateDataset();
    aclmdlAddDatasetBuffer(g_npu_acl_model[0].output_dataset, data_c_z_k);
    aclmdlAddDatasetBuffer(g_npu_acl_model[0].output_dataset, data_r_z_k);

    // 2)output dataset of memory
    g_npu_acl_model[1].output_dataset = aclmdlCreateDataset();
    aclmdlAddDatasetBuffer(g_npu_acl_model[1].output_dataset, data_score);
    aclmdlAddDatasetBuffer(g_npu_acl_model[1].output_dataset, data_bbox);

    return TD_SUCCESS;
}


td_s32 prepare_for_template_execute(td_s32 model_index, td_void* template_im_buf, size_t template_im_len) {
    // sanity check
    if (g_npu_acl_model[model_index].model_desc == TD_NULL) {
        sample_svp_trace_err("no model description, prepare_for_template_execute failed\n");
        return TD_FAILURE;
    }
    // create dataset
    g_npu_acl_model[model_index].input_dataset = aclmdlCreateDataset();
    if (g_npu_acl_model[model_index].input_dataset == TD_NULL) {
        sample_svp_trace_err("can't create input dataset, prepare_for_template_execute failed\n");
        return TD_FAILURE;
    }
    // sanity check
    size_t buffer_size = aclmdlGetInputSizeByIndex(g_npu_acl_model[model_index].model_desc, 0);
    if (buffer_size != template_im_len) {
        sample_svp_trace_err("wrong buffer size, prepare_for_template_execute failed\n");
        return TD_FAILURE;
    }
    // create template input data buffer
    aclDataBuffer *input_data = aclCreateDataBuffer(template_im_buf, template_im_len);
    if (input_data == TD_NULL) {
        sample_svp_trace_err("can't create data buffer, prepare_for_template_execute failed\n");
        return TD_FAILURE;
    }

    // add data buffer to dataset
    td_s32 ret = aclmdlAddDatasetBuffer(g_npu_acl_model[model_index].input_dataset, input_data);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("can't add data buffer, prepare_for_template_execute failed\n");
        aclDestroyDataBuffer(input_data);
        return TD_FAILURE;
    }

    sample_svp_trace_info("prepare_for_template_execute SUCCESS\n");

    return TD_SUCCESS;
}


td_void cleanup_for_template_execute(td_s32 model_index) {
    // release input databuffer and dataset 
    if (g_npu_acl_model[model_index].input_dataset == TD_NULL) {
        return;
    }
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(g_npu_acl_model[model_index].input_dataset); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model_index].input_dataset, i);
        aclDestroyDataBuffer(dataBuffer); // destroy data buffer
    }
    aclmdlDestroyDataset(g_npu_acl_model[model_index].input_dataset);  // destroy input dataset

    // // release output dataset only
    // aclmdlDestroyDataset(g_npu_acl_model[model_index].output_dataset); // release output dataset

    // // unload template model, destroy model desc, set flags and release resources related to 'g_npu_acl_model'
    // sample_npu_unload_model(model_index);

    sample_svp_trace_info("template execution cleanup success\n");
}


td_s32 prepare_for_search_execute(td_s32 model_index, void* search_im_buf, size_t search_im_len) {
    // sanity check
    if (g_npu_acl_model[model_index].model_desc == TD_NULL) {
        sample_svp_trace_err("no model description, prepare_for_search_execute failed\n");
        return TD_FAILURE;
    }
    // create dataset
    g_npu_acl_model[model_index].input_dataset = aclmdlCreateDataset();
    if (g_npu_acl_model[model_index].input_dataset == TD_NULL) {
        sample_svp_trace_err("can't create input dataset, prepare_for_search_execute failed\n");
        return TD_FAILURE;
    }
    // sanity check
    size_t buffer_size = aclmdlGetInputSizeByIndex(g_npu_acl_model[model_index].model_desc, 0);
    if (buffer_size != search_im_len) {
        sample_svp_trace_err("wrong buffer size, prepare_for_search_execute failed\n");
        return TD_FAILURE;
    }
    // create search input data buffer
    aclDataBuffer *input_data = aclCreateDataBuffer(search_im_buf, search_im_len);
    if (input_data == TD_NULL) {
        sample_svp_trace_err("can't create data buffer, prepare_for_search_execute failed\n");
        return TD_FAILURE;
    }
    // add search input data buffer to dataset
    td_s32 ret = aclmdlAddDatasetBuffer(g_npu_acl_model[model_index].input_dataset, input_data);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("can't add data buffer, prepare_for_search_execute failed\n");
        aclDestroyDataBuffer(input_data);
        return TD_FAILURE;
    }
    aclmdlAddDatasetBuffer(g_npu_acl_model[model_index].input_dataset, data_c_z_k);
    aclmdlAddDatasetBuffer(g_npu_acl_model[model_index].input_dataset, data_r_z_k);

    sample_svp_trace_info("prepare_for_search_execute SUCCESS\n");

    return TD_SUCCESS;
}


td_void siamfcpp_postprocess(const stmTrackerState* state, stmTrackerState* result_state) {
    td_u32 len_score, len_bbox;

    // score 
    len_score = aclGetDataBufferSizeV2(data_score);  // size of bytes

    // bbox
    len_bbox = aclGetDataBufferSizeV2(data_bbox);  // size of bytes

    siamfcpp_result(score, len_score/sizeof(float), bbox, len_bbox/sizeof(float), state, result_state);

    // sample_svp_trace_info("stmtrack output data success\n");
    return;
}




td_void cleanup_for_search_execute(td_s32 model_index) {
    // release input databuffer and dataset 
    if (g_npu_acl_model[model_index].input_dataset == TD_NULL) {
        return;
    }
    
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_npu_acl_model[model_index].input_dataset, 0);
    aclDestroyDataBuffer(dataBuffer); // only destroy 1st data buffer

    aclmdlDestroyDataset(g_npu_acl_model[model_index].input_dataset);  // destroy input dataset

    sample_svp_trace_info("search execution cleanup success\n");
}


td_void cleanup_for_siamfcpp() {
    // release and destroy resued data buffer
    aclrtFree(c_z_k);
    aclDestroyDataBuffer(data_c_z_k);
    aclrtFree(r_z_k);
    aclDestroyDataBuffer(data_r_z_k);
    aclrtFree(score);
    aclDestroyDataBuffer(data_score);
    aclrtFree(bbox);
    aclDestroyDataBuffer(data_bbox);

    // destroy output dataset of search model 
    aclmdlDestroyDataset(g_npu_acl_model[0].output_dataset); 
    aclmdlDestroyDataset(g_npu_acl_model[1].output_dataset);

    // unload model
    sample_npu_unload_model(0);
    sample_npu_unload_model(1);
}