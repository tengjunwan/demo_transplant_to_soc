#ifndef _NNNPROCESS_H__
#define _NNNPROCESS_H__

#include "detectobjs.h"

void stmTrack_modleInit(void);
int stmTrack_execute(void* query_buf, size_t query_len, 
                void* memory_buf, size_t memory_len, 
                void* mask_buf, size_t mask_len, 
                stmTrackerState *state);
void stmTrack_result(const float *srcScore, unsigned int lenScore, 
                     const float *srcBbox, unsigned int lenBbox, 
                     stmTrackerState *state);
stmTrackerState getTrackState(void);


#endif