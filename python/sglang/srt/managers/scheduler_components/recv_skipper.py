from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class SchedulerRecvSkipper:
    """
    Executing socket polling and CPU Gloo synchronization too frequently wastes
    host CPU cycles and introduces CPU scheduling jitter. If one CPU core
    experiences a micro-stall (e.g., OS thread descheduling), all other GPUs
    stall waiting on the Gloo broadcast, producing a GPU bubble.
    """
    @staticmethod
    def maybe_create():
        if get_schedule().scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper()

    @staticmethod
    def derive_forward_mode(gathered_modes: List[int]) -> Optional[ForwardMode]:
        """Collapse the gathered per-DP-rank forward modes into one weight-table
        bucket; the input is rank-identical, so the recv decision is too."""
        active = set(gathered_modes) - {
            ForwardMode.IDLE.value,
            ForwardMode.PREBUILT.value,
        }
        # In Attention Data Parallelism (Attn-DP), DP Rank 0 might be actively
        # decoding, while DP Rank 1 is idle or extending. If each rank used its
        # local forward mode to decide whether to skip, Rank 0 might decide to
        # skip, while Rank 1 decides to poll and calls. Rank 1 enters the Gloo
        # broadcast waiting for Rank 0; Rank 0 never enters. The cluster deadlocks.
        if not active:
            return None  # globally idle: same bucket as "no last batch"
        if active - {ForwardMode.DECODE.value, ForwardMode.TARGET_VERIFY.value}:
            return ForwardMode.EXTEND  # any extend-like rank: prompt recv
        if active == {ForwardMode.TARGET_VERIFY.value}:
            return ForwardMode.TARGET_VERIFY
        return ForwardMode.DECODE

    def __init__(self):
        self._use_synced_mode = get_parallel().attn_dp_enabled
        self._counter = 0
        self._threshold = get_schedule().scheduler_recv_interval
        # All can be tuned if needed
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def _pick_mode(self, last_batch: Optional[ScheduleBatch]) -> Optional[ForwardMode]:
        # The recv decision must be identical on every rank in the request
        # broadcast. Local modes differ across DP ranks (IDLE vs DECODE), so
        # use the rank-consistent mode derived from the MLP sync all-gather.
        if last_batch is None:
            return None
        if self._use_synced_mode:
            return last_batch.recv_skipper_forward_mode
        return last_batch.forward_mode

    def handle(self, last_batch: Optional[ScheduleBatch]) -> bool:
        should_recv = False

        # Instead of polling on every step, SchedulerRecvSkipper throttles
        # ingress using a weighted counter:
        # - DECODE (Fast iterations): Has a smaller weight (or higher interval
        # threshold). SGLang might check for new incoming requests only once
        # every 4 or 8 decode steps. This lets the engine run a string of
        # back-to-back decode steps with zero IPC or Gloo overhead.
        # - EXTEND (Prefill / Prompts): Prefill forward passes take much longer
        # (e.g., 20–100 ms). You don't need to skip because the CPU has plenty
        # of headroom while the GPU computes dense GEMMs.
        # - None (Idle): When the engine is idle, last_weight triggers immediate
        # polling (since default weight is much larger than the threshold) so
        # incoming requests are picked up without added latency.
        last_weight = self._weight_of_forward_mode.get(
            self._pick_mode(last_batch), self._default_weight
        )
        self._counter += last_weight

        if self._counter >= self._threshold:
            self._counter = 0
            should_recv = True

        return should_recv
