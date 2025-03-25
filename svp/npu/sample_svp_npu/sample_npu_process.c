/*
  Copyright (c), 2001-2022, Shenshu Tech. Co., Ltd.
 */

#include "sample_npu_process.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <semaphore.h>
#include <pthread.h>
#include <limits.h>

#include "ot_common_svp.h"
#include "sample_common_svp.h"
#include "sample_npu_model.h"
#include "detectobjs.h"

static td_u32 g_npu_dev_id = 0;

static td_void sample_svp_npu_destroy_resource(td_void) {
    aclError ret;

    ret = aclrtResetDevice(g_npu_dev_id);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("reset device fail\n");
    }
    sample_svp_trace_info("end to reset device is %d\n", g_npu_dev_id);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("finalize acl fail\n");
    }
    sample_svp_trace_info("end to finalize acl\n");
}

static td_s32 sample_svp_npu_init_resource(td_void) {
    /* ACL init */
    const char *acl_config_path = "";
    aclrtRunMode run_mode;
    td_s32 ret;

    ret = aclInit(acl_config_path);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("acl init fail.\n");
        return TD_FAILURE;
    }
    sample_svp_trace_info("acl init success.\n");

    /* open device */
    ret = aclrtSetDevice(g_npu_dev_id);
    if (ret != ACL_ERROR_NONE) {
        sample_svp_trace_err("acl open device %d fail.\n", g_npu_dev_id);
        return TD_FAILURE;
    }
    sample_svp_trace_info("open device %d success.\n", g_npu_dev_id);

    /* get run mode */
    ret = aclrtGetRunMode(&run_mode);
    if ((ret != ACL_ERROR_NONE) || (run_mode != ACL_DEVICE)) {
        sample_svp_trace_err("acl get run mode fail.\n");
        return TD_FAILURE;
    }
    sample_svp_trace_info("get run mode success\n");

    return TD_SUCCESS;
}

td_s32 sample_svp_npu_acl_prepare_init() {
    td_s32 ret;

    ret = sample_svp_npu_init_resource();
    if (ret != TD_SUCCESS) {
        sample_svp_npu_destroy_resource();
    }

    return ret;
}

td_void sample_svp_npu_acl_prepare_exit(td_u32 thread_num) {
    for (td_u32 model_index = 0; model_index < thread_num; model_index++) {
        sample_npu_destroy_desc(model_index);
        sample_npu_unload_model(model_index);
    }
    sample_svp_npu_destroy_resource();
}

td_s32 sample_svp_npu_load_model(const char* om_model_path, td_u32 model_index, td_bool is_cached) {
    td_char path[PATH_MAX] = { 0 };
    td_s32 ret;

    if (sizeof(om_model_path) > PATH_MAX) {
        sample_svp_trace_err("pathname too long!.\n");
        return TD_NULL;
    }
    if (realpath(om_model_path, path) == TD_NULL) {
        sample_svp_trace_err("invalid file!.\n");
        return TD_NULL;
    }

    if (is_cached == TD_TRUE) {
        ret = sample_npu_load_model_with_mem_cached(path, model_index);
    } else {
        ret = sample_npu_load_model_with_mem(path, model_index);
    }

    if (ret != TD_SUCCESS) {
        sample_svp_trace_err("execute load model fail, model_index is:%d.\n", model_index);
        goto acl_prepare_end1;
    }
    ret = sample_npu_create_desc(model_index);
    if (ret != TD_SUCCESS) {
        sample_svp_trace_err("execute create desc fail.\n");
        goto acl_prepare_end2;
    }

    printModelIODescription(model_index);

    return TD_SUCCESS;

acl_prepare_end2:
    sample_npu_destroy_desc(model_index);

acl_prepare_end1:
    sample_npu_unload_model(model_index);
    return ret;
}

td_s32 sample_svp_npu_dataset_prepare_init(td_u32 model_index) {
    td_s32 ret;

    ret = sample_npu_create_input_dataset(model_index);
    if (ret != TD_SUCCESS) {
        sample_svp_trace_err("execute create input fail.\n");
        return TD_FAILURE;
    }
    ret = sample_npu_create_output(model_index);
    if (ret != TD_SUCCESS) {
        sample_npu_destroy_input_dataset(model_index);
        sample_svp_trace_err("execute create output fail.\n");
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

td_s32 sample_svp_npu_model_execute(td_u32 model_index) {
    return sample_npu_model_execute(model_index);
}

td_void sample_svp_npu_destroy_output(td_u32 model_index) {
    sample_npu_destroy_output(model_index);
}

td_void sample_svp_npu_destroy_input_dataset(td_u32 model_index) {
    sample_npu_destroy_input_dataset(model_index);
}

td_s32 sample_svp_npu_create_input_databuf(td_void *data_buf, size_t data_len, td_u32 model_index) {
    return sample_npu_create_input_databuf(data_buf, data_len, model_index);
}

td_s32 sample_svp_npu_create_input_databuf_v2(td_void *data1_buf, size_t data1_len, 
                                              td_void *data2_buf, size_t data2_len, 
                                              td_u32 model_index) {
    return sample_npu_create_input_databuf_v2(data1_buf, data1_len, 
                                              data2_buf, data2_len,
                                              model_index);
}

td_void sample_svp_npu_destroy_input_databuf(td_u32 thread_num) {
    for (td_u32 model_index = 0; model_index < thread_num; model_index++) {
        sample_npu_destroy_input_databuf(model_index);
    }
}

td_void sample_svp_npu_model_link_buffer(td_u32 model1_index, 
                                         td_u32 model2_index, 
                                         td_u32 model3_index) {
    sample_npu_model_link_buffer(model1_index, 
                                 model2_index, 
                                 model3_index);
}

// td_void sample_svp_npu_output_model_result_head(td_u32 model_index, 
//                                             stmTrackerState *state) {
//     sample_npu_output_model_result_head(model_index, state);
// }