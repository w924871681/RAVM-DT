"""Common observable lifecycle boundaries; never changes an action or optimizer."""
from common import *
import time,threading,psutil
def install(b):
    maintenance=b.Timeline.maintenance;service=b.Timeline.service;candidate=b.make_candidate;score=b.score;put=b.CAS.put;native_adapt=b.native.adapt;persist=b.persist
    def maintain(self,t):
        if t==1:b.TLS.cost_blobs={}
        b.TLS.cost=dict(stage=t,start=time.perf_counter(),training_seconds=0.,verification_seconds=0.,model_persistence_seconds=0.,rss_start=psutil.Process().memory_info().rss)
        if b.DEVICE=='cuda':b.torch.cuda.reset_peak_memory_stats()
        return maintenance(self,t)
    def open_service(self,t):
        timer=getattr(b.TLS,'cost',None)
        if timer and timer['stage']==t and 'service_open' not in timer:timer['service_open']=time.perf_counter()
        return service(self,t)
    def timed(fn,field):
        def call(*a,**k):
            begin=time.perf_counter()
            try:
                value=fn(*a,**k)
                if field=='model_persistence_seconds' and getattr(b.TLS,'cost',None):b.TLS.cost_blobs[value['file_sha256']]=value['bytes']
                if field=='verification_seconds' and getattr(b.TLS,'cost',None):
                    history=list(a[3])+list(a[2]);b.TLS.cost['persistent_buffer_bytes']=sum(s.x.nbytes+s.y.nbytes+s.ts.nbytes for s in history)
                return value
            finally:
                timer=getattr(b.TLS,'cost',None)
                if timer:timer[field]+=time.perf_counter()-begin
        return call
    def persist_timing(ctx,row):
        result=persist(ctx,row)
        finish(b,ctx['td'],row['stage'],native_row=row)
        return result
    b.Timeline.maintenance=maintain;b.Timeline.service=open_service;b.make_candidate=timed(candidate,'training_seconds');b.score=timed(score,'verification_seconds');b.CAS.put=timed(put,'model_persistence_seconds');b.native.adapt=timed(native_adapt,'training_seconds');b.persist=persist_timing

def finish(b,td,t,native_row=None):
    timer=getattr(b.TLS,'cost',None)
    if not timer or timer['stage']!=t:return
    path=Path(td)/f'COST_UNIFIED_{t:03}.json'
    if path.exists():return
    now=time.perf_counter();opened=timer['service_open'];maintenance=opened-timer['start']
    result=read(Path(td)/f'result_{t:03}.json')
    verification=native_row['verification_time_s'] if native_row is not None else timer['verification_seconds']
    record=dict(stage=t,maintenance_common_seconds=maintenance,training_seconds=timer['training_seconds'],verification_and_gate_seconds=verification,
        model_persistence_seconds=timer['model_persistence_seconds'],remaining_preparation_seal_and_bookkeeping_seconds=max(0.,maintenance-timer['training_seconds']-verification-timer['model_persistence_seconds']),
        postseal_service_evaluation_and_result_persistence_seconds=now-opened,cpu_rss_start_bytes=timer['rss_start'],cpu_rss_end_bytes=psutil.Process().memory_info().rss,
        postseal_timer_scope='includes evaluation, late feedback if any and result persistence; separately measured scientific evaluator and late feedback below',
        scientific_service_evaluation_seconds=result['post_seal_scientific_evaluation_seconds'],late_feedback_generation_seconds=result['late_score_seconds'],maintenance_total_with_late_feedback_seconds=maintenance+result['late_score_seconds'],
        gpu_process_peak_allocated_bytes=b.torch.cuda.max_memory_allocated() if b.DEVICE=='cuda' else None,
        persistent_buffer_bytes=native_row['persistent_buffer_bytes'] if native_row is not None else timer.get('persistent_buffer_bytes'),
        persistent_model_logical_unique_bytes=sum(b.TLS.cost_blobs.values()),persistent_model_unique_hashes=len(b.TLS.cost_blobs),
        persistent_buffer_measurement='native exact buffer bytes' if native_row is not None else 'actual prior history plus recent raw block arrays at score call; excludes transient window tensors',
        timing_boundary='entry into Timeline.maintenance through first Timeline.service access; includes candidate, gate, CAS, seal, and EWC postseal Fisher; excludes shared M0 and Service evaluation',
        measurement_scope='single ready lifecycle per repetition on this shared host; other applications may still compete; not complexity or energy guarantee')
    atomic(path,record)
