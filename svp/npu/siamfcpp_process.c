#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>

#include "sample_comm.h"
#include "sample_npu_process.h"
#include "sample_npu_model.h"
#include "siamfcpp_process.h"

// 参数常量
static const float penalty_k = 0.08f;
static const float window_influence = 0.2f;
static const float test_lr = 0.58f;
static float consine_window[289];
static bool consine_window_initialized = false;
static float search_input_size = 303.0f;

// 定义 PI
#define PI 3.14159265358979323846f

static void init_cosine_window() {
    for (int i = 0; i < 17; i++) {
        for (int j = 0; j < 17; j++) {
            float h = 0.5f * (1 - cosf(2 * PI * i / 16));
            float w = 0.5f * (1 - cosf(2 * PI * j / 16));
            consine_window[i * 17 + j] = h * w;
        }
    }
}

// 计算带填充的大小
static float size_with_pad(float w, float h) {
    float pad = (w + h) * 0.5f;
    return sqrtf((w + pad) * (h + pad));
}

// 后处理分数并应用惩罚
static int postprocess_score(const float *score, const float *bbox,
                      const stmTrackerState *state,
                      float pscore[289], float penalty[289]) {
    float prev_w = state->w / state->scale; // 恢复到 289 缩放比例
    float prev_h = state->h / state->scale; // 恢复到 289 缩放比例
    float prev_size = size_with_pad(prev_w, prev_h);
    float prev_ratio = prev_w / prev_h;

    int best_pscore_id = 0;
    float best_pscore = -1.0f;

    for (int i = 0; i < 289; i++) {
        float w = bbox[i * 4 + 2] - bbox[i * 4];  // w = x1 - x0
        float h = bbox[i * 4 + 3] - bbox[i * 4 + 1];  // h = y1 - y0

        // 尺寸变化
        float current_size = size_with_pad(w, h);
        float size_change = fmaxf(current_size / prev_size, prev_size / current_size);
        // printf("size_change[%d]: %.2f\n", i, size_change);
        // 比例变化
        float current_ratio = w / h;
        float ratio_change = fmaxf(current_ratio / prev_ratio, prev_ratio / current_ratio);
        // printf("ratio_change[%d]: %.2f\n", i, ratio_change);

        // 惩罚分数（因变形）
        penalty[i] = expf((1.0f - size_change * ratio_change) * penalty_k);
        // printf("penalty[%d]: %.2f\n", i, penalty[i]);
        pscore[i] = penalty[i] * score[i];
        // printf("score[%d]: %.2f\n", i, score[i]);
        // printf("pscore[%d]: %.2f\n", i, pscore[i]);

        // 减小由于快速位置变化引起的分数
        pscore[i] = pscore[i] * (1 - window_influence) + consine_window[i] * window_influence;

        // printf("pscore[%d] after: %.2f\n", i, pscore[i]);

        // 更新最佳分数
        if (pscore[i] > best_pscore) {
            best_pscore = pscore[i];
            best_pscore_id = i;
        }
    }

    return best_pscore_id;
}

// 后处理 bbox（EMA 更新）
static void postprocess_bbox(const float *score, const float *bbox,
                      const stmTrackerState *state,
                      const float penalty[289], int best_id,
                      float result[4]) {
    // 获取最佳 bbox
    float x0 = bbox[best_id * 4] * state->scale;  // 恢复到原始比例
    float y0 = bbox[best_id * 4 + 1] * state->scale;
    float x1 = bbox[best_id * 4 + 2] * state->scale;
    float y1 = bbox[best_id * 4 + 3] * state->scale;

    // xyxy 转换为 cxcywh，并返回到全局坐标
    float cx = (x0 + x1) * 0.5f + state->cx - (search_input_size / 2.0f) * state->scale;
    float cy = (y0 + y1) * 0.5f + state->cy - (search_input_size / 2.0f) * state->scale;
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

static void clipf_inplace(float* value, float min, float max) {
    if (*value < min) {
        *value = min;
    } else if (*value > max) {
        *value = max;
    }
}



void siamfcpp_result(const float *srcScore, unsigned int lenScore, const float *srcBbox, 
                     unsigned int lenBbox, const stmTrackerState *state, stmTrackerState *result_state) {
    sample_print("       state: cx: %.2f, cy: %.2f, w: %.2f, h: %.2f, score: %.2f\n", state->cx, 
        state->cy, state->w, state->h, state->score);
    sample_print("result_state: cx: %.2f, cy: %.2f, w: %.2f, h: %.2f, score: %.2f\n", result_state->cx,
            result_state->cy, result_state->w, result_state->h, result_state->score);
    // 提示：score.shape=(1, 289=17*17, 1), bbox.shape=(1, 289=17*17, 4)
    if (lenScore != 289 || lenBbox != 289 * 4) {
        return -1;
    }

    if (!consine_window_initialized) {
        init_cosine_window();
        consine_window_initialized = true;
        printf("cosine window initialized\n");
    }

    // 分数后处理
    float pscore[289], penalty[289];
    int best_id = postprocess_score(srcScore, srcBbox, state, pscore, penalty);

    // bbox 后处理
    float bbox[4];
    postprocess_bbox(srcScore, srcBbox, state, penalty, best_id, bbox);
    printf("bbox[0]:%.2f, bbox[1]:%.2f, bbox[2]:%.2f, bbox[3]:%.2f\n", bbox[0], bbox[1], bbox[2], bbox[3]);

    // 更新状态
    result_state->cx = bbox[0];
    result_state->cy = bbox[1];
    result_state->w = bbox[2];
    result_state->h = bbox[3];
    result_state->score = pscore[best_id];

    sample_print("       state: cx: %.2f, cy: %.2f, w: %.2f, h: %.2f, score: %.2f\n", state->cx, 
        state->cy, state->w, state->h, state->score);
    sample_print("result_state: cx: %.2f, cy: %.2f, w: %.2f, h: %.2f, score: %.2f\n", result_state->cx,
         result_state->cy, result_state->w, result_state->h, result_state->score);


    // clipf_inplace(&result_state->cx, 0.0f, 303.0f);
    // clipf_inplace(&result_state->cy, 0.0f, 303.0f);
    // clipf_inplace(&result_state->w, 1.0f, 300.0f);
    // clipf_inplace(&result_state->h, 1.0f, 300.0f);


    return 0;
}

td_void siamfcpp_init(td_void) {
   const char *om_model_path_template = "siamfcpp_template_direct_half.om";  
    const char *om_model_path_search = "siamfcpp_search_direct_half.om";  
    // const char *om_model_path_template = "siamfcpp_template_opt.om";  
    // const char *om_model_path_search = "siamfcpp_search_opt.om";  
    td_s32 ret;

    // ===================acl init===================
    ret = sample_svp_npu_acl_prepare_init();
    if (ret != TD_SUCCESS) {
        sample_print("sample_svp_npu_acl_prepare_init failed\n");
        return;
    }
    // ===================load models==================
    // load template model
    ret = sample_svp_npu_load_model(om_model_path_template, 0, TD_FALSE); // load model, create model desc; model index = '0'
    if (ret != TD_SUCCESS) {
        printf("failed to load model: : %s\n", om_model_path_template);
        goto acl_process_end0;
    }

    // load search model 
    ret = sample_svp_npu_load_model(om_model_path_search, 1, TD_FALSE);  // model index = '1'
    if (ret != TD_SUCCESS) {
        printf("failed to load model: : %s\n", om_model_path_search);
        goto acl_process_end0;
    }

    // allocate reused buffer
    ret = prepare_for_siamfcpp_execution();
    if (ret != TD_SUCCESS) {
        printf("failed to allocate reused buffer\n");
        goto acl_process_end0;
    }
    return;

acl_process_end0:
    sample_svp_npu_acl_prepare_exit(2);  // unload model, destroy model desc, (TODO stream, context), reset device, acl finalize
    return;
}


td_s32 template_execute(td_void* template_im_buf, size_t template_im_len) {
    td_s32 ret;
    ret = prepare_for_template_execute(0, template_im_buf, template_im_len);
    ret = sample_svp_npu_model_execute(0);
    cleanup_for_template_execute(0);
}


td_s32 search_execute(td_void* search_im_buf, size_t search_im_len, const stmTrackerState* state, stmTrackerState* result_state) {
    td_s32 ret;
    ret = prepare_for_search_execute(1, search_im_buf, search_im_len);
    ret = sample_svp_npu_model_execute(1);
    siamfcpp_postprocess(state, result_state);
    cleanup_for_search_execute(1);
}


void siamfcpp_cleanup(void) {
    cleanup_for_siamfcpp();
}


