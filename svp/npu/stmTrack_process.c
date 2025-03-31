#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>

#include "sample_comm.h"
#include "sample_npu_process.h"
#include "stmTrack_process.h"

// 参数常量
static const float penalty_k = 0.04f;
// static const float window_influence = 0.21f;
static const float window_influence = 0.0f;
static const float test_lr = 0.95f;
static float consine_window[625];
static bool consine_window_initialized = false;

// 定义 PI
#define PI 3.14159265358979323846f

void init_cosine_window() {
    for (int i = 0; i < 25; i++) {
        for (int j = 0; j < 25; j++) {
            float h = 0.5f * (1 - cosf(2 * PI * i / 24));
            float w = 0.5f * (1 - cosf(2 * PI * j / 24));
            consine_window[i * 25 + j] = h * w;
        }
    }
}

// 计算带填充的大小
float size_with_pad(float w, float h) {
    float pad = (w + h) * 0.5f;
    return sqrtf((w + pad) * (h + pad));
}

// 后处理分数并应用惩罚
int postprocess_score(const float *score, const float *bbox,
                      stmTrackerState *state,
                      float pscore[625], float penalty[625]) {
    float prev_w = state->w / state->scale; // 恢复到 289 缩放比例
    float prev_h = state->h / state->scale; // 恢复到 289 缩放比例
    float prev_size = size_with_pad(prev_w, prev_h);
    float prev_ratio = prev_w / prev_h;

    int best_pscore_id = 0;
    float best_pscore = -1.0f;

    for (int i = 0; i < 625; i++) {
        float w = bbox[i * 4 + 2] - bbox[i * 4];  // w = x1 - x0
        float h = bbox[i * 4 + 3] - bbox[i * 4 + 1];  // h = y1 - y0

        // 尺寸变化
        float current_size = size_with_pad(w, h);
        float size_change = fmaxf(current_size / prev_size, prev_size / current_size);

        // 比例变化
        float current_ratio = w / h;
        float ratio_change = fmaxf(current_ratio / prev_ratio, prev_ratio / current_ratio);

        // 惩罚分数（因变形）
        penalty[i] = expf((1.0f - size_change * ratio_change) * penalty_k);
        pscore[i] = penalty[i] * score[i];

        // 减小由于快速位置变化引起的分数
        pscore[i] = pscore[i] * (1 - window_influence) + consine_window[i] * window_influence;

        // 更新最佳分数
        if (pscore[i] > best_pscore) {
            best_pscore = pscore[i];
            best_pscore_id = i;
        }
    }

    return best_pscore_id;
}

// 后处理 bbox（EMA 更新）
void postprocess_bbox(const float *score, const float *bbox,
                      stmTrackerState *state,
                      const float penalty[625], int best_id,
                      float result[4]) {
    // 获取最佳 bbox
    float x0 = bbox[best_id * 4] * state->scale;  // 恢复到原始比例
    float y0 = bbox[best_id * 4 + 1] * state->scale;
    float x1 = bbox[best_id * 4 + 2] * state->scale;
    float y1 = bbox[best_id * 4 + 3] * state->scale;

    // xyxy 转换为 cxcywh，并返回到全局坐标
    float cx = (x0 + x1) * 0.5f + state->cx - (289.0f / 2.0f) * state->scale;
    float cy = (y0 + y1) * 0.5f + state->cy - (289.0f / 2.0f) * state->scale;
    float w = x1 - x0;
    float h = y1 - y0;

    // 使用 EMA 更新宽高
    float lr = penalty[best_id] * score[best_id] * test_lr;
    w = state->w * (1 - lr) + w * lr;
    h = state->h * (1 - lr) + h * lr;

    // 将结果存储在 result 数组中
    result[0] = cx;
    result[1] = cy;
    result[2] = w;
    result[3] = h;
}

// 主函数：结果计算
void stmTrack_result(const float *srcScore, unsigned int lenScore, 
                     const float *srcBbox, unsigned int lenBbox, 
                     stmTrackerState *state) {
    // 提示：score.shape=(1, 625=25*25, 1), bbox.shape=(1, 625=25*25, 4)
    if (lenScore != 625 || lenBbox != 625 * 4) {
        return ;
    }

    if (!consine_window_initialized) {
        init_cosine_window();
        consine_window_initialized = true;
        sample_print("cosine window initialized\n");
    }

    // 分数后处理
    float pscore[625];
    float penalty[625];
    int best_id = postprocess_score(srcScore, srcBbox, state, pscore, penalty);

    // bbox 后处理
    float bbox[4];
    postprocess_bbox(srcScore, srcBbox, state, penalty, best_id, bbox);

    // sample_print("cx: %.2f, cy: %.2f, w: %.2f, h: %.2f, score: %.2f\n", bbox[0], bbox[1], bbox[2], bbox[3], pscore[best_id]);

    if (bbox[0] < 0) {
        bbox[0] = 0;
    } 
    if (bbox[1] < 0) {
        bbox[1] = 0;
    } 
    if (bbox[2] <= 0) {
        bbox[2] = 1;
    } 
    if (bbox[3] <= 0) {
        bbox[3] = 1;
    } 

    // 更新状态
    state->cx = bbox[0];
    state->cy = bbox[1];
    state->w = bbox[2];
    state->h = bbox[3];
    state->score = pscore[best_id];

    return ;
}

static long getms() {
    struct timeval start;
    gettimeofday(&start, NULL);
    long ms = (start.tv_sec) * 1000 + (start.tv_usec) / 1000;
    return ms;
}

td_void stmTrack_modleInit(td_void) {
    td_s32 ret;
    const char *om_model_path_query = "STMTrack_FeatureExtractionQuery.om";
    const char *om_model_path_memory = "STMTrack_FeatureExtractionMemory.om";
    const char *om_model_path_readMemoryAndHead = "STMTrack_ReadMemoryAndHead.om";

    ret = sample_svp_npu_acl_prepare_init();
    if (ret != TD_SUCCESS) {
        sample_print("sample_svp_npu_acl_prepare_init failed\n");
        return;
    }

    ret = sample_svp_npu_load_model(om_model_path_query, 0, TD_FALSE);
    if (ret != TD_SUCCESS) {
        sample_print("sample_svp_npu_load_model failed\n");
        goto acl_process_end0;
    }
    ret = sample_svp_npu_load_model(om_model_path_memory, 1, TD_FALSE);
    if (ret != TD_SUCCESS) {
        sample_print("sample_svp_npu_load_model failed\n");
        goto acl_process_end0;
    }
    ret = sample_svp_npu_load_model(om_model_path_readMemoryAndHead, 2, TD_FALSE);
    if (ret != TD_SUCCESS) {
        sample_print("sample_svp_npu_load_model failed\n");
        goto acl_process_end0;
    }

    sample_print("stmTrack_init success\n");
    return ;

acl_process_end0:
    sample_svp_npu_acl_prepare_exit(3);
    return ;
}

td_s32 stmTrack_execute(td_void* query_buf, size_t query_len, 
                   td_void* memory_buf, size_t memory_len, 
                   td_void* mask_buf, size_t mask_len, 
                   stmTrackerState *state) {
    td_s32 ret;
    ret = sample_svp_npu_dataset_prepare_init(0);
    if (ret != TD_SUCCESS) {
        sample_print("dataset prepare init fail.\n");
    }
    ret = sample_svp_npu_dataset_prepare_init(1);
    if (ret != TD_SUCCESS) {
        sample_print("dataset prepare init fail.\n");
    }
    ret = sample_svp_npu_dataset_prepare_init(2);
    if (ret != TD_SUCCESS) {
        sample_print("dataset prepare init fail.\n");
    }

    ret = sample_svp_npu_create_input_databuf(query_buf, query_len, 0);
    if (ret != TD_SUCCESS) {
        sample_print("memcpy_s query device buffer fail.\n");
        return TD_FAILURE;
    }
    ret = sample_svp_npu_create_input_databuf_v2(memory_buf, memory_len, mask_buf, mask_len, 1);  // use memory buffer & mask buffer to create data buffer for memory model
    if (ret != TD_SUCCESS) {
        sample_print("memcpy_s memory device buffer fail.\n");
        return TD_FAILURE;
    }

    long time_start0 = getms();
    ret = sample_svp_npu_model_execute(0);
    if (ret != TD_SUCCESS) {
        sample_print("0-sample_npu_model_execute failed.\n");
        return TD_FAILURE;
    }
    long time_done0 = getms();
    sample_print("0-done, time: %ld ms\n", time_done0 - time_start0);

    long time_start1 = getms();
    ret = sample_svp_npu_model_execute(1);
    if (ret != TD_SUCCESS) {
        sample_print("1-sample_npu_model_execute failed.\n");
        return TD_FAILURE;
    }
    long time_done1 = getms();
    sample_print("1-done, time: %ld ms\n", time_done1 - time_start1);

    long time_start2 = getms();
    sample_svp_npu_model_link_buffer(0, 1, 2);
    ret = sample_svp_npu_model_execute(2);
    if (ret != TD_SUCCESS) {
        sample_print("2-sample_npu_model_execute failed.\n");
        return -1;
    }
    long time_done2 = getms();
    sample_print("2-done, time: %ld ms\n", time_done2 - time_start2);

    // sample_svp_npu_output_model_result_head(2, state);

    sample_svp_npu_destroy_output(0); 
    sample_svp_npu_destroy_input_dataset(0);
    sample_svp_npu_destroy_output(1); 
    sample_svp_npu_destroy_input_dataset(1);
    sample_svp_npu_destroy_output(2); 
    sample_svp_npu_destroy_input_dataset(2);
    return TD_SUCCESS;
}