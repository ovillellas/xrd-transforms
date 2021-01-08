
#include <dispatch/dispatch.h>


static dispatch_queue_t task_queue=NULL;

void
xrd_transforms_task_apply(size_t count, void* data, task_fn function)
{
    dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    dispatch_apply_f(count, q, data, function);
}


