import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Union

from fastapi import Request
from fastapi import Request as RawRequest
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import (
    update_stop_words_from_env,
    update_tokenizer_special_tokens,
)
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint
from rtp_llm.frontend.frontend_worker import FrontendWorker, TokenizerEncodeResponse
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import SpecialTokens, TaskType
from rtp_llm.server.misc import format_exception
from rtp_llm.structure.request_extractor import request_id_field_name
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import check_with_info

USAGE_HEADER = "USAGE"


class FrontendServer(object):
    def __init__(
        self,
        rank_id: int,
        server_id: int,
        py_env_configs=None,
    ):
        self.py_env_configs = py_env_configs
        self._access_logger = AccessLogger(
            get_log_path(),
            py_env_configs.profiling_debug_logging_config.log_file_backup_count,
            rank_id,
            server_id,
        )
        self._frontend_worker = None
        self._openai_endpoint = None
        self._embedding_endpoint = None
        self.is_embedding = False
        self.thread_lock_ = threading.Lock()
        self._request_perf_index = 0
        self._global_controller = get_global_controller()
        self.rank_id = str(rank_id)
        self.server_id = str(server_id)
        kmonitor.init()

    def _next_request_perf_index(self) -> int:
        with self.thread_lock_:
            self._request_perf_index += 1
            return self._request_perf_index

    @staticmethod
    def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _safe_to_dict(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, BaseModel):
            return obj.model_dump(exclude_none=True)
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return {}

    @classmethod
    def _response_aux_info(cls, response: Any) -> Dict[str, Any]:
        return cls._safe_to_dict(cls._safe_get(response, "aux_info"))

    @classmethod
    def _response_usage_info(cls, response: Any) -> Dict[str, Any]:
        return cls._safe_to_dict(cls._safe_get(response, "usage"))

    @staticmethod
    def _request_shape(req: Dict[Any, Any]) -> str:
        generate_config = req.get("generate_config", {})
        if not isinstance(generate_config, dict):
            generate_config = {}
        prompt = req.get("prompt", "")
        messages = req.get("messages", [])
        return (
            f"kind={'openai' if ChatCompletionRequest.is_openai_request(req) else 'raw'},"
            f"stream={req.get('stream', False)},"
            f"source={req.get('source', 'unknown')},"
            f"prompt_chars={len(prompt) if isinstance(prompt, str) else 0},"
            f"messages_count={len(messages) if isinstance(messages, list) else 0},"
            f"max_tokens={req.get('max_tokens', '')},"
            f"max_completion_tokens={req.get('max_completion_tokens', '')},"
            f"max_new_tokens={generate_config.get('max_new_tokens', '')},"
            f"private_request={req.get('private_request', False)}"
        )

    def start(self):
        if (
            self.py_env_configs.profiling_debug_logging_config.debug_start_fake_process
            == 1
        ):
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake server")
            self._frontend_worker = None
            return

        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=self.py_env_configs.kv_cache_config,
            profiling_debug_logging_config=self.py_env_configs.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
            vit_config=self.py_env_configs.vit_config,
        )

        # Create a temporary tokenizer to initialize special_tokens
        # We'll update it with the actual tokenizer after FrontendWorker is created
        special_tokens = SpecialTokens()
        if self.py_env_configs.generate_env_config:
            update_stop_words_from_env(
                special_tokens, self.py_env_configs.generate_env_config
            )

        # Create FrontendWorker with special_tokens and config
        self._frontend_worker = FrontendWorker(
            self.py_env_configs,
            model_config,
            special_tokens,
        )

        # Update special_tokens with actual tokenizer
        update_tokenizer_special_tokens(special_tokens, self._frontend_worker.tokenizer)

        # Only initialize OpenaiEndpoint for LANGUAGE_MODEL task type
        if model_config.task_type == TaskType.LANGUAGE_MODEL:
            # Update model_config with the latest values
            model_config.special_tokens = special_tokens
            model_config.generate_env_config = self.py_env_configs.generate_env_config
            model_config.render_config = self.py_env_configs.render_config
            model_config.model_name = self.py_env_configs.model_args.model_type
            model_config.template_type = None

            self._openai_endpoint = OpenaiEndpoint(
                model_config=model_config,
                misc_config=self.py_env_configs.misc_config,
                vit_config=self.py_env_configs.vit_config,
                tokenizer=self._frontend_worker.tokenizer,
                backend_rpc_server_visitor=self._frontend_worker.backend_rpc_server_visitor,
            )
        else:
            from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint

            self._embedding_endpoint = EmbeddingEndpoint(
                model_config=model_config,
                grpc_config=self.py_env_configs.grpc_config,
                server_config=self.py_env_configs.server_config,
                tokenizer=self._frontend_worker.tokenizer,
            )
            self.is_embedding = True

    def stop(self):
        if self._frontend_worker is not None:
            self._frontend_worker.stop()

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        start_time = time.time()
        try:
            if isinstance(request, str):
                request = json.loads(request)
            kmonitor.report(
                AccMetrics.QPS_METRIC, 1, {"source": request.get("source", "unknown")}
            )
            sequence = self._global_controller.increment() % 4096  # 12 bits
            request[request_id_field_name] = generate_request_id(
                self.py_env_configs.server_config.ip,
                self.py_env_configs.server_config.server_port,
                self.server_id,
                sequence,
            )
        except Exception as e:
            return self._handle_exception(request, e)

        try:
            assert (
                self._embedding_endpoint is not None
            ), "embedding pipeline should not be None"
            result, logable_result = await self._embedding_endpoint.embedding(request)
            # do not log result since too big
            if logable_result is not None:
                self._access_logger.log_success_access(request, logable_result)
            end_time = time.time()
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, (end_time - start_time) * 1000
            )
            kmonitor.report(
                AccMetrics.SUCCESS_QPS_METRIC,
                1,
                {"source": request.get("source", "unknown")},
            )
            usage = result.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            return ORJSONResponse(result, headers={USAGE_HEADER: json.dumps(usage)})
        except BaseException as e:
            return self._handle_exception(request, e)
        finally:
            self._global_controller.decrement()

    # use asyncio.sleep(0) to correctly exit when client closed https://github.com/tiangolo/fastapi/issues/4146
    async def stream_response(
        self,
        request: Dict[str, Any],
        response: CompleteResponseAsyncGenerator,
    ):
        is_openai_response = request.get("stream", False)
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            async for res in response:
                data_str = res.model_dump_json(exclude_none=True)
                yield response_data_prefix + data_str + "\r\n\r\n"
                await asyncio.sleep(0)
            if not is_openai_response:
                yield f"data:[done]\r\n\r\n"
            await self._collect_complete_response_and_record_access_log(
                request, response
            )
        except asyncio.CancelledError as e:
            self._access_logger.log_exception_access(request, e)
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unkown"),
                },
            )
        except BaseException as e:
            # 捕获非Cancel以外所有的异常,所以使用BaseException
            self._access_logger.log_exception_access(request, e)
            format_e = format_exception(e)
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unkown"),
                    "error_code": str(format_e.get("error_code_str", -1)),
                },
            )
            yield response_data_prefix + json.dumps(
                format_e, ensure_ascii=False
            ) + "\r\n\r\n"
        finally:
            self._global_controller.decrement()

    async def inference(self, req: Union[str, Dict[Any, Any]], raw_request: RawRequest):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            sequence = self._global_controller.increment() % 4096  # 12 bits
            req[request_id_field_name] = generate_request_id(
                self.py_env_configs.server_config.ip,
                self.py_env_configs.server_config.server_port,
                self.server_id,
                sequence,
            )
        except Exception as e:
            return self._handle_exception(req, e)

        def generate_call():
            assert self._frontend_worker is not None
            return self._frontend_worker.inference(**req)

        try:
            rep = await self._infer_wrap(req, raw_request, generate_call)
        except Exception as e:
            self._global_controller.decrement()
            raise e

        if not isinstance(rep, StreamingResponse):
            self._global_controller.decrement()

        return rep

    async def _infer_wrap(
        self,
        req: Dict[str, Any],
        raw_request: RawRequest,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
    ):
        try:
            rep = await self._infer_impl(req, raw_request, generate_call)
        except BaseException as e:
            rep = self._handle_exception(req, e)
        return rep

    async def chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        sequence = self._global_controller.increment() % 4096  # 12 bits
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )

        def generate_call():
            assert self._openai_endpoint != None
            response = self._openai_endpoint.chat_completion(
                request_id, request, raw_request
            )
            assert isinstance(
                response, CompleteResponseAsyncGenerator
            ), f"error type: {type(response)}"
            return response

        try:
            request_dict = request.model_dump(exclude_none=True)
            request_dict[request_id_field_name] = request_id
            rep = await self._infer_wrap(request_dict, raw_request, generate_call)
        except Exception as e:
            self._global_controller.decrement()
            raise e

        if not isinstance(rep, StreamingResponse):
            self._global_controller.decrement()

        return rep

    async def batch_chat_completion(self, request, raw_request: Request):
        from rtp_llm.openai.api_datatype import BatchChatCompletionResponse

        sequence = self._global_controller.increment() % 4096
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )
        try:
            assert self._openai_endpoint is not None
            responses = await self._openai_endpoint.batch_chat_completion(
                request_id, request
            )
            return ORJSONResponse(
                content=BatchChatCompletionResponse(
                    responses=[r.model_dump(exclude_none=True) for r in responses]
                ).model_dump()
            )
        finally:
            self._global_controller.decrement()

    async def batch_infer(self, req: dict, raw_request: Request):
        from rtp_llm.frontend.frontend_worker import BatchPipelineResponse

        # Concurrency accounting: a batch counts as ONE scheduling unit because the engine
        # atomically enqueues all prompts via BatchGenerateCall. Per-item counting would over-
        # reject under the same concurrency_limit; the trade-off is that a large batch occupies
        # only one slot regardless of N.
        sequence = self._global_controller.increment() % 4096
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )
        try:
            assert self._frontend_worker is not None
            prompts = req.get("prompt_batch", [])
            generate_config = req.get("generate_config", {})
            result = await self._frontend_worker.batch_infer(
                prompts=prompts,
                request_id=request_id,
                generate_config=generate_config,
            )
            return ORJSONResponse(content=result.model_dump(exclude_none=True))
        finally:
            self._global_controller.decrement()

    async def chat_render(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            assert self._openai_endpoint != None
            return self._openai_endpoint.chat_render(request)
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def _handle_exception(self, request: Dict[str, Any], e: BaseException):
        exception_json = format_exception(e)
        error_code_str = exception_json.get("error_code_str", "")
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
        elif isinstance(e, asyncio.CancelledError):
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                },
            )
            self._access_logger.log_exception_access(request, e)
        else:
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                    "error_code": error_code_str,
                },
            )
            self._access_logger.log_exception_access(request, e)

        rep = ORJSONResponse(exception_json, status_code=500)
        return rep

    async def _call_generate_with_report(
        self,
        req: Dict[Any, Any],
        request_perf_index: int,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
    ):
        async def __gen_response_with_report(start_time: float, response_generator):
            last_iterate_time = current_time_ms()
            first_token = True
            iter_count = 0
            first_token_rt_ms = None
            last_aux_info: Dict[str, Any] = {}
            last_usage_info: Dict[str, Any] = {}
            completed = False
            try:
                async for response in response_generator:
                    end_time = current_time_ms()
                    aux_info = self._response_aux_info(response)
                    usage_info = self._response_usage_info(response)
                    if aux_info:
                        last_aux_info = aux_info
                    if usage_info:
                        last_usage_info = usage_info
                    if first_token:
                        first_token = False
                        first_token_rt_ms = end_time - start_time
                        kmonitor.report(
                            GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC,
                            end_time - last_iterate_time,
                        )
                        logging.info(
                            "Request first token timing: request_index=%s request_id=%s rank_id=%s server_id=%s first_token_rt_ms=%.2f shape=[%s]",
                            request_perf_index,
                            req.get(request_id_field_name, ""),
                            self.rank_id,
                            self.server_id,
                            first_token_rt_ms,
                            self._request_shape(req),
                        )
                    else:
                        step_output_len = 1
                        if hasattr(response, "aux_info"):
                            if isinstance(response.aux_info, list):
                                step_output_len = 0
                                for info in response.aux_info:
                                    step_output_len += info.get("step_output_len", 1)
                            elif isinstance(response.aux_info, dict):
                                step_output_len = max(
                                    response.aux_info.get("step_output_len", 1),
                                    step_output_len,
                                )

                        kmonitor.report(
                            GaugeMetrics.RESPONSE_ITER_RT_METRIC,
                            (end_time - last_iterate_time) / step_output_len,
                        )
                    kmonitor.report(
                        AccMetrics.ITER_QPS_METRIC,
                        1,
                        {
                            "rank_id": self.rank_id,
                            "server_id": self.server_id,
                        },
                    )
                    last_iterate_time = end_time
                    iter_count += 1
                    yield response
                completed = True
            finally:
                e2e_ms = current_time_ms() - start_time
                kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
                kmonitor.report(GaugeMetrics.LANTENCY_METRIC, e2e_ms)
                if completed:
                    kmonitor.report(
                        AccMetrics.SUCCESS_QPS_METRIC,
                        1,
                        {
                            "rank_id": self.rank_id,
                            "server_id": self.server_id,
                        },
                    )
                logging.info(
                    "Request performance summary: request_index=%s request_id=%s rank_id=%s server_id=%s completed=%s iter_count=%s first_token_rt_ms=%s e2e_ms=%.2f input_len=%s output_len=%s step_output_len=%s reuse_len=%s local_reuse_len=%s remote_reuse_len=%s memory_reuse_len=%s wait_time=%s aux_cost_time=%s aux_first_token_cost_time=%s usage_prompt_tokens=%s usage_completion_tokens=%s usage_total_tokens=%s shape=[%s]",
                    request_perf_index,
                    req.get(request_id_field_name, ""),
                    self.rank_id,
                    self.server_id,
                    completed,
                    iter_count,
                    f"{first_token_rt_ms:.2f}" if first_token_rt_ms is not None else "",
                    e2e_ms,
                    last_aux_info.get("input_len", ""),
                    last_aux_info.get("output_len", ""),
                    last_aux_info.get("step_output_len", ""),
                    last_aux_info.get("reuse_len", ""),
                    last_aux_info.get("local_reuse_len", ""),
                    last_aux_info.get("remote_reuse_len", ""),
                    last_aux_info.get("memory_reuse_len", ""),
                    last_aux_info.get("wait_time", ""),
                    last_aux_info.get("cost_time", ""),
                    last_aux_info.get("first_token_cost_time", ""),
                    last_usage_info.get("prompt_tokens", ""),
                    last_usage_info.get("completion_tokens", ""),
                    last_usage_info.get("total_tokens", ""),
                    self._request_shape(req),
                )

        assert self._frontend_worker is not None
        start_time = current_time_ms()
        response_generator = generate_call()
        return CompleteResponseAsyncGenerator(
            __gen_response_with_report(start_time, response_generator),
            response_generator._collect_complete_response_func,
        )

    async def _collect_complete_response_and_record_access_log(
        self, req: Dict[Any, Any], res: Any
    ):
        complete_response = await res.gen_complete_response_once()
        complete_response = (
            complete_response.model_dump(exclude_none=True)
            if isinstance(complete_response, BaseModel)
            else complete_response
        )
        self._access_logger.log_success_access(req, complete_response)

        return complete_response

    async def _infer_impl(
        self,
        req: Dict[Any, Any],
        raw_request: RawRequest,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
    ):
        assert self._frontend_worker is not None
        kmonitor.report(
            AccMetrics.QPS_METRIC,
            1,
            {
                "rank_id": self.rank_id,
                "server_id": self.server_id,
                "source": req.get("source", "unkown"),
            },
        )
        self._access_logger.log_query_access(req)
        is_streaming = self._frontend_worker.is_streaming(req)
        request_perf_index = self._next_request_perf_index()
        logging.info(
            "Request performance start: request_index=%s request_id=%s rank_id=%s server_id=%s shape=[%s]",
            request_perf_index,
            req.get(request_id_field_name, ""),
            self.rank_id,
            self.server_id,
            self._request_shape(req),
        )
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        res = await self._call_generate_with_report(
            req, request_perf_index, generate_call
        )

        if is_streaming:
            return StreamingResponse(
                self.stream_response(req, res), media_type="text/event-stream"
            )
        async for x in res:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await res.aclose()
                raise asyncio.CancelledError("client disconnects")

        complete_response = await self._collect_complete_response_and_record_access_log(
            req, res
        )
        return ORJSONResponse(content=complete_response)

    def tokenize(self, req: str | Dict[str, Any]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            if ChatCompletionRequest.is_openai_request(req):
                chat_request = ChatCompletionRequest(**req)
                token_ids = self._openai_endpoint.render_chat(chat_request).input_ids
            else:
                prompt = req.pop("prompt")
                token_ids = self._frontend_worker.pipeline.encode(prompt)
            return ORJSONResponse({"token_ids": token_ids})
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def tokenizer_encode(self, req: Union[str, Dict[Any, Any]]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            prompt = req.pop("prompt")
            assert self._frontend_worker is not None
            if req.get("return_offsets_mapping", None) == True:
                mapping = self._frontend_worker.tokenizer_offset_mapping(prompt)
                response = TokenizerEncodeResponse(
                    offset_mapping=mapping["offset_mapping"],
                    token_ids=mapping["input_ids"],
                )
            else:
                token_ids, tokens = self._frontend_worker.tokenizer_encode(prompt)
                response = TokenizerEncodeResponse(token_ids=token_ids, tokens=tokens)
            return ORJSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def check_health(self):
        assert self._frontend_worker is not None
        return (
            self._frontend_worker.backend_rpc_server_visitor.is_backend_service_ready(
                refresh=False
            )
        )
