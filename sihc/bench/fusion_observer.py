"""Observe compile-time DDP partition counts without changing the backend."""
def observe_ddp(report):
    from torch._dynamo.backends.distributed import DDPOptimizer
    original = DDPOptimizer.compile_fn
    report['ddp_backend_partitions'] = []
    def observed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        report['ddp_backend_partitions'].append(dict(
            buckets=len(getattr(self, 'buckets', [])),
            bucket_sizes=[b.size for b in getattr(self, 'buckets', [])],
            returned_submodules=len(list(result.named_children())) if hasattr(result, 'named_children') else None))
        return result
    DDPOptimizer.compile_fn = observed
