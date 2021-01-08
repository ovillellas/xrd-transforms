#ifndef XRD_TRANSFORMS_TASK_SYSTEM_H
#define XRD_TRANSFORMS_TASK_SYSTEM_H


typedef void (*task_fn)(void *data, size_t iteration);

void
xrd_transforms_task_apply(size_t count, void* data, task_fn function);

#endif /* XRD_TRANSFORMS_TASK_SYSTEM_H */
