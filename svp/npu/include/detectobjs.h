#ifndef __DETECTOBJS_H__
#define __DETECTOBJS_H__

#ifdef __cplusplus
extern "C" {
#endif

  // object tracing stuff
  typedef struct {
    float cx; // center x of bbox
    float cy; // center y of bbox
    float w;
    float h;
    float score; // score after post-processing
    float scale; // scale from source to 289*289
  } stmTrackerState;


#ifdef __cplusplus
}
#endif
#endif
