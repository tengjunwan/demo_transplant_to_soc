#ifndef _SIAMFCPPNNNPROCESS_H__
#define _SIAMFCPPNNNPROCESS_H__

#include "detectobjs.h"

void siamfcpp_init(void);
int template_execute(void* template_im_buf, size_t template_im_len);
int search_execute(void* search_im_buf, size_t search_im_len, const stmTrackerState* state, stmTrackerState* result_state);
void siamfcpp_cleanup(void);
void siamfcpp_result(const float *srcScore, unsigned int lenScore, const float *srcBbox, unsigned int lenBbox, 
                     const stmTrackerState *state, stmTrackerState *result_state);
#endif