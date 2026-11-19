# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
import time
import uuid
from array import array
from collections.abc import Callable
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils.log_utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


class Session:
    def __init__(
        self,
        capacity_of_str_len: int,
        session_id: Optional[str] = None,
        streaming: bool = False,
        timeout: Optional[float] = None,
    ):
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.capacity_of_str_len = capacity_of_str_len
        # Streaming session is an optimization specifically designed for strictly
        # linear, append-only workloads (like feeding chunks of a live audio/text stream
        # or sequential agent steps). Therefore, we don't need to keep tree of history
        # or allow modifcation in streaming mode.
        self.streaming = streaming
        self.timeout = timeout
        self.last_active_time: float = time.monotonic()
        # rid -> that turn's request; a new turn may continue from any of them.
        self.turns: Dict[str, Req] = {}
        self.close_on_finish: bool = False
        self._inflight: bool = False
        # Token-array lengths of last_req as of its finish_req. The share path
        # appends speculatively beyond these; only finish_req confirms them, so
        # _share_token_arrays trims back first (heals aborted turns).
        self.committed_origin_len: Optional[int] = None
        self.committed_unpadded_len: Optional[int] = None
        self.committed_fill_len: Optional[int] = None

    def is_timed_out(self) -> bool:
        if self.timeout is None:
            return False
        return time.monotonic() - self.last_active_time > self.timeout

    @staticmethod
    def _strip_bos_token(req: TokenizedGenerateReqInput, tokenizer) -> None:
        """Trim a leading BOS on an appended turn; shift mm offsets to match."""
        # BOS stands for Beginning of Sentence: <s>, <|begin_of_text|>, <bos>.
        if not (
            tokenizer is not None
            and req.input_ids
            and req.input_ids[0] == tokenizer.bos_token_id
        ):
            return
        # Since the first turn already contains BOS, we trim the BOS of the second turn:
        # Turn 1 ("Hello"):           [BOS, "Hello"]
        # Turn 2 ("What about C++?"): [BOS, "What", "about", "C++", "?"]
        # The combine sequence would become:
        # [BOS, "Hello", "Hi there!", BOS, "What", "about", "C++", "?"]
        req.input_ids = req.input_ids[1:]
        if req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if item.offsets:
                    if any(s == 0 for s, _ in item.offsets):
                        logging.warning(
                            "mm_item offset starts at 0 (BOS position), "
                            "clamping to 0 after BOS strip"
                        )
                    # image placeholder in a prompt spans (start, end) offsets,
                    # so we need to shift it left by one.
                    item.offsets = [
                        (max(0, s - 1), max(0, e - 1)) for s, e in item.offsets
                    ]

    def _share_token_arrays(self, last_req: Req, new_input_ids):
        """Plain streaming append: reuse last_req's token arrays in place.

        Trims each array back to its committed length first — an earlier turn
        may have appended its tokens and then aborted before finish_req, and
        turns still points at last_req, so anything beyond the committed
        lengths is unconfirmed. Then extends with last turn's output and the
        new input. Returns (input_ids, input_ids_unpadded, carry_fill);
        carry_fill (== the new origin) spares the first fill_ids rebuild.
        """
        # Due to overlap scheduling and speculative decoding, the engine may generate
        # some extra trailing tokens past 'max_new_tokens', so we need to trim the
        # output.
        out_tail = last_req.output_ids[: last_req.sampling_params.max_new_tokens]

        # Trim both the padded input and unpadded input to the last committed len.
        # REFACTOR: This version is clearer.
        input_ids = last_req.origin_input_ids
        input_ids_unpadded = last_req.origin_input_ids_unpadded
        del input_ids[self.committed_origin_len :]
        if input_ids_unpadded is not input_ids:
            del input_ids_unpadded[self.committed_unpadded_len :]

        # del input_ids[self.committed_origin_len :]
        # if last_req.origin_input_ids_unpadded is input_ids:
        #     input_ids_unpadded = input_ids
        # else:
        #     input_ids_unpadded = last_req.origin_input_ids_unpadded
        #     del input_ids_unpadded[self.committed_unpadded_len :]

        carry_fill = last_req.full_untruncated_fill_ids
        if (
            not isinstance(carry_fill, array)
            or carry_fill is input_ids
            or carry_fill is input_ids_unpadded
        ):
            # Unexpected type or aliased with an origin array (extending it
            # below would double-append): let refresh_fill_ids rebuild.
            # Not sure how this case can ever happen, but for safety, set 'carry_fill'
            # to None to avoid double append.
            carry_fill = None
        else:
            # Truncate the array to the last committed len first.
            del carry_fill[self.committed_fill_len :]
            # 'baked' is the number of output tokens already baked into 'carry_fill'.
            # For example:
            # - input_ids = [10, 20, 30]
            # - carry_fill = [10, 20, 30, 901, 902]
            # - out_tail = [901, 902, 903]
            # In this case, 'baked = 2' since two output tokens are already appended to
            # the 'carry_fill' array, so we just need to extend
            # 'out_tail[2:] = 903'.
            baked = len(carry_fill) - len(input_ids)
            if 0 <= baked <= len(out_tail):
                carry_fill.extend(out_tail[baked:])
                carry_fill.extend(new_input_ids)
            else:
                carry_fill = None

        # Notice that out_tail (last_req's output) is only appended into input_ids
        # when the NEXT turn starts!
        # So if that next turn aborts, input_ids currently contains
        # [last_req's input] + [last_req's output (out_tail)] + [aborted turn's new_input_ids]
        input_ids.extend(out_tail)
        input_ids.extend(new_input_ids)
        if input_ids_unpadded is not input_ids:
            input_ids_unpadded.extend(out_tail)
            input_ids_unpadded.extend(new_input_ids)

        # At this point, 'input_ids' and 'carry_fill' have the same content. However,
        # 'input_ids' is strictly immutable while 'carry_fill' is an append-only working
        # buffer.
        return input_ids, input_ids_unpadded, carry_fill

    @staticmethod
    def _concat_token_arrays(
        last_req: Req, req: TokenizedGenerateReqInput, session_params: SessionParams
    ):
        """Copy-based assembly for non-streaming turns."""
        out_tail = last_req.output_ids[: last_req.sampling_params.max_new_tokens]

        # OPT: We should do 'input_ids = last_req.origin_input_ids + out_tail'
        # only when 'drop_previous_output' is false.
        input_ids = last_req.origin_input_ids + out_tail
        if session_params.drop_previous_output:
            input_ids = last_req.origin_input_ids[:]
        if session_params.offset and session_params.offset != 0:
            input_ids = input_ids[: session_params.offset] + req.input_ids
        else:
            input_ids += req.input_ids

        # BUG: 'input_ids_unpadded' and 'input_ids' are two different Python list
        # objects. For should check for alias before performing this object allocation.
        input_ids_unpadded = last_req.origin_input_ids_unpadded + out_tail
        if session_params.drop_previous_output:
            input_ids_unpadded = last_req.origin_input_ids_unpadded[:]
        if session_params.offset and session_params.offset != 0:
            # In a text-only model, 'input_ids' and 'input_ids_unpadded' have identical
            # lengths, so slicing at offset cuts at the exact same token. However, in a
            # multi-modal request, applying the exact same integer offset to both arrays
            # would slice them at completely different semantic positions.
            # In practice, offset is designed for text-based editing and is not intended
            # to slice across multi-modal image blocks.
            input_ids_unpadded = (
                input_ids_unpadded[: session_params.offset] + req.input_ids
            )
        else:
            input_ids_unpadded += req.input_ids
        return input_ids, input_ids_unpadded

    def create_req(
        self,
        req: TokenizedGenerateReqInput,
        tokenizer,
        vocab_size: int,
        eos_token_ids=None,
        disagg_mode: Optional[DisaggregationMode] = None,
        req_factory: Callable[..., Req] = Req,
    ):
        assert req.session_params is not None
        session_params = req.session_params

        last_req = None
        abort = False
        abort_message = ""
        if session_params.replace:
            abort = True
            abort_message = (
                "Session replace is not supported. To branch from a turn, append "
                "with its rid; to stop a running turn first, abort it."
            )
        elif self.streaming:
            # Streaming sessions: only simple appends allowed; reject otherwise.
            if self._inflight:
                # Only one request from the current session can run at a time.
                abort = True
                abort_message = "Streaming session already has an active request."
            elif session_params.drop_previous_output:
                # Need to keep everything in streaming mode.
                abort = True
                abort_message = (
                    "Streaming sessions do not support drop_previous_output."
                )
            elif session_params.offset and session_params.offset != 0:
                # No tree to branch out.
                abort = True
                abort_message = "Streaming sessions do not support offset."
            elif self.turns:
                assert len(self.turns) == 1
                # Peek (don't pop) the single turn. turns is updated
                # only in finish_req after the request completes successfully.
                # Streaming sessions keep only one turn (`Req`), which contains all the
                # chat history. We don't need multiple turns since we don't support
                # branching, so we fold each completed turn into the single entry in
                # `self.turns`.
                [last_req] = self.turns.values()
        else:
            if session_params.rid is not None:
                if session_params.rid not in self.turns:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req = self.turns[session_params.rid]
                    if not last_req.finished():
                        abort = True
                        abort_message = "Session request is appending to a request that hasn't finished."
                        logging.warning(abort_message)

        carry_fill = None
        if last_req is not None:
            self._strip_bos_token(req, tokenizer)
            # In-place sharing is only safe for the plain streaming append:
            # streaming sessions allow a single inflight request, last_req has
            # finished, and the committed_* lengths recorded by finish_req let
            # _share_token_arrays trim away tokens appended by an aborted turn.
            # offset / drop_previous_output rewrite history and must copy.
            # REFACTOR: We don't need to check for 'drop_previous_output' and 'offset'.
            # The above checks already cover this. 'committed_origin_len' is updated
            # once a streaming request finishes.
            can_share_token_arrays = (
                self.streaming
                and self.committed_origin_len is not None
                and not session_params.drop_previous_output
                and not (session_params.offset and session_params.offset != 0)
            )
            if can_share_token_arrays:
                # Reuse the two input lists and output buffer of the previous request.
                input_ids, input_ids_unpadded, carry_fill = self._share_token_arrays(
                    last_req, req.input_ids
                )
            else:
                input_ids, input_ids_unpadded = self._concat_token_arrays(
                    last_req, req, session_params
                )
        else:
            # No existing history for this session, use the current prompt as is.
            input_ids = req.input_ids
            input_ids_unpadded = req.input_ids

        if not abort and len(input_ids) == 0:
            abort = True
            abort_message = (
                "A session request must contain input tokens after restoring history."
            )

        new_req = req_factory(
            rid=req.rid,
            origin_input_text=None,
            origin_input_ids=input_ids,
            origin_input_ids_unpadded=input_ids_unpadded,
            sampling_params=req.sampling_params,
            lora_id=req.lora_id,
            session=self,
            custom_logit_processor=req.custom_logit_processor,
            stream=req.stream,
            return_logprob=req.return_logprob,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            return_sampling_mask=req.return_sampling_mask,
            sampling_logprobs_mode=req.sampling_logprobs_mode,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
            require_reasoning=req.require_reasoning,
            return_hidden_states=req.return_hidden_states,
            return_routed_experts=req.return_routed_experts,
            routed_experts_start_len=req.routed_experts_start_len,
            bootstrap_host=req.bootstrap_host,
            bootstrap_port=req.bootstrap_port,
            bootstrap_room=req.bootstrap_room,
            disagg_mode=disagg_mode,
            routed_dp_rank=req.routed_dp_rank,
            disagg_prefill_dp_rank=req.disagg_prefill_dp_rank,
            priority=req.priority,
            routing_key=req.routing_key,
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
            http_worker_ipc=req.http_worker_ipc,
            time_stats=req.time_stats,
        )
        if last_req is not None:
            new_req.multimodal_inputs = last_req.multimodal_inputs
            new_req.bind_multimodal_cache_namespace()
            # An offset can truncate inherited media.
            if not abort and not session_params.offset:
                new_req.inherit_mm_token_counts(last_req)
        new_req.tokenizer = tokenizer
        if carry_fill is not None:
            new_req.full_untruncated_fill_ids = carry_fill

        if abort:
            # Set the request's status to abort to let the scheduler handle it.
            new_req.set_finish_with_abort(abort_message)
        elif self.streaming:
            self.last_active_time = time.monotonic()
            # turns is NOT updated here; finish_req() handles it.
            self._inflight = True
        else:
            # Non-streaming mode registers nodes eagerly at creation time, whereas
            # streaming mode registers nodes lazily upon successful completion (see
            # 'finish_req')
            self.last_active_time = time.monotonic()
            # Note that 'session_params.rid' is used to identify which node to continue
            # from or replace while 'req.rid' is the current (new) request ID.
            self.turns[req.rid] = new_req

        return new_req

    def finish_req(self, req):
        """Update turns after a streaming request finishes successfully."""
        # NOTE: This method is strictly and exclusively designed for streaming mode.
        self._inflight = False
        if self.turns:
            # This unpack operation requires 'req_nodes.values()' to contain exactly one
            # item.
            [prev_req] = self.turns.values()
            prev_req.session = None
            self.turns.clear()
        # In streaming mode, we defer deleting the last request node and creating new
        # request node to guarantee abort safety. If we delete the last node and abort
        # mid-way, 'req_nodes' only contains the aborted node.
        self.turns[req.rid] = req
        # Confirm this req's token arrays as the session's rollback point.
        self.committed_origin_len = len(req.origin_input_ids)
        self.committed_unpadded_len = len(req.origin_input_ids_unpadded)
        self.committed_fill_len = len(req.full_untruncated_fill_ids)

    def abort_req(self):
        """Clear inflight flag on abort (turns stays unchanged)."""
        self._inflight = False


class SessionController:
    def __init__(self, tree_cache: BasePrefixCache):
        self.sessions: Dict[str, Session] = {}
        self._last_reap_time: float = 0.0
        self.tree_cache = tree_cache

    def __contains__(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def open(self, recv_req: OpenSessionReqInput) -> OpenSessionReqOutput:
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id=session_id, success=False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id=session_id, success=False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len,
                session_id,
                streaming=bool(recv_req.streaming),
                timeout=recv_req.timeout,
            )
            log_info_on_rank0(
                logger, f"Session opened: {session_id} (active={len(self.sessions)})"
            )
            return OpenSessionReqOutput(session_id=session_id, success=True)

    def close(self, recv_req: CloseSessionReqInput):
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            self._close(session_id)

    def _close(self, session_id: str):
        session = self.sessions[session_id]
        if self._has_live_readers(session):
            # A request still reads what the close frees: a streaming turn
            # decodes on the session's KV memory, or an unfinished turn holds
            # the session's media. Mark the session for deferred cleanup: the
            # request keeps its session reference so release_kv_cache takes
            # the streaming path, and maybe_reap closes the session once no
            # request reads it.
            session.close_on_finish = True
            logger.info(
                "Deferring session close for %s (unfinished request)",
                session_id,
            )
            return

        # No reader left -- safe to release immediately.
        if session.streaming and session.turns:
            req = next(iter(session.turns.values()))
            req.session = None

        # Release multimodal features held by session requests.
        # Session reqs skip the normal mm cleanup path (scheduler and
        # output_processor) so features stay alive until the session closes.
        seen_mm = set()
        for turn in session.turns.values():
            mm = turn.multimodal_inputs
            if mm is not None and id(mm) not in seen_mm:
                seen_mm.add(id(mm))
                mm.release_features()
            turn.multimodal_inputs = None

        self.tree_cache.release_radix_session(session_id)
        self.tree_cache.release_session(session_id)
        del self.sessions[session_id]
        log_info_on_rank0(
            logger, f"Session closed: {session_id} (active={len(self.sessions)})"
        )

    def maybe_reap(self, now: float, interval: float = 1.0):
        # reap sessions every second
        if now - self._last_reap_time > interval:
            self._last_reap_time = now

            # Finish deferred closes for sessions no request reads anymore.
            pending = [
                sid
                for sid, session in self.sessions.items()
                if session.close_on_finish and not self._has_live_readers(session)
            ]
            for sid in pending:
                log_info_on_rank0(
                    logger, f"Deferred close ready for session {sid}, releasing."
                )
                self._close(sid)

            # A deferred close is already pending; closing it again each
            # second would only repeat its log lines.
            timed_out = [
                sid
                for sid, session in self.sessions.items()
                if session.is_timed_out() and not session.close_on_finish
            ]
            for sid in timed_out:
                log_info_on_rank0(logger, f"Session {sid} timed out, closing.")
                self._close(sid)

    @staticmethod
    def _has_live_readers(session: Session) -> bool:
        """Whether a request still reads what closing `session` frees: the
        streaming turn in flight, or an unfinished turn that holds media."""
        return session._inflight or any(
            not turn.finished() and turn.multimodal_inputs is not None
            for turn in session.turns.values()
        )

    @staticmethod
    def adjust_mm_offsets(recv_req: TokenizedGenerateReqInput, req: Req, image_inputs):
        # For session requests, adjust mm_inputs offsets by the prefix length.
        # Session.create_req prepends previous context to origin_input_ids,
        # so offsets from the new prompt need to be shifted.
        if len(recv_req.input_ids) >= len(req.origin_input_ids):
            return
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
        for mm_item in image_inputs.mm_items:
            if mm_item.offsets:
                mm_item.offsets = [
                    (start + prefix_len, end + prefix_len)
                    for start, end in mm_item.offsets
                ]
