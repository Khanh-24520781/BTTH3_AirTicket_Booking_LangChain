"""Kiểm thử nhánh API bằng SDK thật + HTTP mock; không dùng key thật/không gọi mạng.

Chạy trong .venv có requirements-api.txt: python -m unittest -v test_agents test_api
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from agents import HybridFlightAgent, PlanThenExecuteFlightAgent, ReActFlightAgent
from evaluate_agents import aggregate_metrics, export_results, run_benchmark
from flight_harness import FlightConstraints, FlightDatabase, HarnessGuard
from model_config import ModelConfig, ModelConfigurationError, create_model, load_model_config


def api_plan(flight="VN122", passenger="Nguyen Van A"):
    # NOTE TEST API: payload model đã đóng schema, không phải policy model thật.
    empty = {k: "" for k in ["origin", "destination", "date", "flight", "passenger_name", "code", "payment_method"]}
    return {"steps": [
        {"tool": "book_seat", "args": {**empty, "flight": flight, "passenger_name": passenger}},
        {"tool": "pay", "args": {**empty, "code": "$booking_code", "payment_method": "wallet"}},
        {"tool": "get_booking", "args": {**empty, "code": "$booking_code"}},
    ]}


def openai_response(*, plan=None, call=None, text="", usage=True):
    message = {"role": "assistant", "content": text}
    if plan is not None or call is not None:
        name, arguments = ("ApiPlan", plan) if plan is not None else call
        message["tool_calls"] = [{"id": "call_contract", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(arguments)}}]
    payload = {"id": "chatcmpl-contract", "object": "chat.completion", "created": 1, "model": "contract-model",
               "choices": [{"index": 0, "message": message,
                            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]}
    if usage:
        payload["usage"] = {"prompt_tokens": 23, "completion_tokens": 11, "total_tokens": 34}
    return payload


def google_response(*, plan=None, call=None, text="", signature=None):
    if plan is not None or call is not None:
        name, arguments = ("ApiPlan", plan) if plan is not None else call
        part = {"functionCall": {"name": name, "args": arguments}}
    else:
        part = {"text": text}
    if signature:
        part["thoughtSignature"] = signature
    return {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 15, "totalTokenCount": 45},
            "modelVersion": "gemini-contract-test"}


class ConfigTests(unittest.TestCase):
    def test_offline_does_not_load_env_or_construct_provider(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "secret", "LLM_PROVIDER": "unknown"}), \
                patch("model_config.Path.is_file", side_effect=AssertionError("offline không đọc .env")):
            config = load_model_config(mode="offline")
            self.assertEqual(config.mode, "offline")
            self.assertIsNone(create_model(config))

    def test_env_cli_priority_and_public_config_hide_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("LLM_PROVIDER=openai\nLLM_MODEL=file-model\nLLM_API_KEY=file-secret\n", encoding="utf-8-sig")
            args = argparse.Namespace(mode="api", env_file=path, model="cli-model")
            with patch.dict(os.environ, {"LLM_MODEL": "env-model", "LLM_API_KEY": "env-secret"}, clear=True):
                config = load_model_config(args)
            self.assertEqual(config.model, "cli-model")
            self.assertEqual(config.api_key, "env-secret")
            self.assertNotIn("env-secret", repr(config))
            self.assertNotIn("env-secret", json.dumps(config.public_settings()))

    def test_bad_config_does_not_fallback(self):
        for values in [{"api_key": ""}, {"model": ""}, {"provider": "unknown"},
                       {"base_url": "https://host/v1?key=secret"}, {"timeout_seconds": float("nan")},
                       {"max_retries": 3}, {"provider": "openai-compatible", "base_url": None}]:
            with self.subTest(values=values), self.assertRaises(ModelConfigurationError):
                ModelConfig(**{**{"mode": "api", "provider": "openai", "model": "test", "api_key": "test-secret"}, **values})


SDK_AVAILABLE = all(importlib.util.find_spec(name) for name in ("langchain_openai", "langchain_google_genai"))


@unittest.skipUnless(SDK_AVAILABLE, "Cài requirements-api.txt để chạy SDK contract tests")
class ApiContractTests(unittest.TestCase):
    def make(self, scenario="success"):
        constraints = FlightConstraints()
        db = FlightDatabase(scenario)
        return constraints, db, HarnessGuard(constraints, db)

    def make_model(self, responder, *, provider="openai-compatible", method="function_calling"):
        import httpx
        requests = []
        def transport(request):
            body = json.loads(request.content)
            requests.append(body)  # không lưu header/key vào bằng chứng
            response = responder(body, len(requests))
            return response if isinstance(response, httpx.Response) else httpx.Response(200, json=response)
        http = httpx.Client(transport=httpx.MockTransport(transport))
        self.addCleanup(http.close)
        config = ModelConfig(mode="api", provider=provider, model="gemini-contract-test" if provider == "google-genai" else "contract-model",
                             api_key="contract-test-secret", base_url="https://mock-provider.invalid/v1" if provider == "openai-compatible" else None,
                             plan_method=method)
        if provider == "google-genai":
            from google.genai import Client
            client = Client(api_key=config.api_key, http_options={"httpx_client": http})
            self.addCleanup(client.close)
            with patch("langchain_google_genai.chat_models.Client", return_value=client):
                model = create_model(config)
        else:
            from langchain_openai import ChatOpenAI
            with patch("langchain_openai.ChatOpenAI", side_effect=lambda **kwargs: ChatOpenAI(http_client=http, **kwargs)):
                model = create_model(config)
        self.assertFalse(requests, "Khởi tạo SDK/check-config không được gọi API")
        return config, model, requests

    def test_provider_planner_schema_usage_and_booking_through_harness(self):
        for provider in ["openai-compatible", "google-genai"]:
            for method in ["function_calling", "json_schema"]:
                with self.subTest(provider=provider, method=method):
                    payload = api_plan()
                    signature = "Y29udHJhY3Qtc2lnbmF0dXJl"
                    def response(body, index):
                        if provider == "google-genai":
                            return (google_response(plan=payload, signature=signature) if method == "function_calling"
                                    else google_response(text=json.dumps(payload), signature=signature))
                        return openai_response(plan=payload) if method == "function_calling" else openai_response(text=json.dumps(payload))
                    config, model, requests = self.make_model(response, provider=provider, method=method)
                    c, db, guard = self.make()
                    with contextlib.ExitStack() as stack:
                        if provider == "google-genai":
                            generate = stack.enter_context(patch.object(
                                model.client.models, "generate_content", wraps=model.client.models.generate_content))
                        result = PlanThenExecuteFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
                    if provider == "google-genai":
                        self.assertTrue(generate.call_args.kwargs["config"].automatic_function_calling.disable)
                    self.assertEqual(result.status, "SUCCESS", result.error_message)
                    self.assertEqual(result.model_mode, "api")
                    self.assertEqual(result.llm_calls, 1)
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(result.total_tokens, 45 if provider == "google-genai" else 34)
                    self.assertTrue(result.usage_complete)
                    self.assertEqual(len(db.bookings), 1)
                    self.assertTrue(guard.is_done())
                    self.assertNotIn(signature, json.dumps(asdict(result)))
                    for entry in result.trace:
                        if entry.get("step") == "model_response":
                            self.assertIsInstance(entry["content"], str)
                    # Assert request thật có schema args, không gửi object rỗng.
                    body = requests[0]
                    if provider == "google-genai":
                        schema = (body["tools"][0]["functionDeclarations"][0]["parameters"] if method == "function_calling"
                                  else body["generationConfig"]["responseJsonSchema"])
                    else:
                        schema = (body["tools"][0]["function"]["parameters"] if method == "function_calling"
                                  else body["response_format"]["json_schema"]["schema"])
                    wire = json.dumps(schema)
                    self.assertIn("passenger_name", wire)
                    if method == "json_schema":
                        self.assertIn("maxItems", wire)
                    self.assertNotIn(config.api_key, json.dumps(asdict(result)))

    def test_react_uses_real_sdk_tool_roundtrip_and_total_usage(self):
        c, db, guard = self.make()
        def response(body, index):
            if index == 1:
                return openai_response(call=("search_flights", {"origin": c.origin, "destination": c.destination, "date": c.date}))
            if index == 2:
                return openai_response(call=("book_seat", {"flight": "VN122", "passenger_name": c.passenger_name}))
            if index == 3:
                booked = json.loads(body["messages"][-1]["content"])
                return openai_response(call=("pay", {"code": booked["code"], "payment_method": "wallet"}))
            return openai_response(text="Đã thanh toán chuyến VN122.")
        config, model, requests = self.make_model(response)
        result = ReActFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
        self.assertEqual(result.status, "SUCCESS", result.error_message)
        self.assertEqual((result.llm_calls, result.tool_calls), (4, 3))
        self.assertEqual(result.total_tokens, 136)
        self.assertEqual(len(requests), 4)

    def test_gemini_react_keeps_signatures_and_harness_executes_tools_without_afc(self):
        # NOTE TEST API: response HTTP giả có chữ ký như Gemini 3. Xác nhận
        # chữ ký còn trong request kế tiếp, không lọt vào trace để chụp báo cáo.
        c, db, guard = self.make()
        signature = "Y29udHJhY3Qtc2lnbmF0dXJl"
        def response(body, index):
            if index > 1:
                prior_parts = [part for content in body["contents"] for part in content["parts"]
                               if "functionCall" in part]
                self.assertTrue(prior_parts)
                self.assertTrue(all(part.get("thoughtSignature") == signature for part in prior_parts))
            if index == 1:
                call = ("search_flights", {"origin": c.origin, "destination": c.destination, "date": c.date})
            elif index == 2:
                call = ("book_seat", {"flight": "VN122", "passenger_name": c.passenger_name})
            elif index == 3:
                call = ("pay", {"code": next(iter(db.bookings)), "payment_method": "wallet"})
            else:
                return google_response(text="Đã thanh toán chuyến VN122.", signature=signature)
            return google_response(call=call, signature=signature)
        config, model, requests = self.make_model(response, provider="google-genai")
        with patch.object(model.client.models, "generate_content", wraps=model.client.models.generate_content) as generate:
            result = ReActFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
        self.assertEqual(result.status, "SUCCESS", result.error_message)
        self.assertEqual((result.llm_calls, result.tool_calls, len(requests)), (4, 3, 4))
        self.assertEqual(result.total_tokens, 180)
        self.assertEqual([name for name, _, _ in guard.log], ["search_flights", "book_seat", "pay"])
        self.assertTrue(all(call.kwargs["config"].automatic_function_calling.disable
                            for call in generate.call_args_list))
        self.assertNotIn(signature, json.dumps(asdict(result)))

    def test_api_plan_is_reviewed_and_pending_cannot_write(self):
        for agent_type in [PlanThenExecuteFlightAgent, HybridFlightAgent]:
            with self.subTest(agent=agent_type.__name__):
                config, model, _ = self.make_model(lambda body, index: openai_response(plan=api_plan()))
                c, db, guard = self.make()
                result = agent_type(c, db, guard).run(fake_mode=False, llm=model, model_config=config, auto_approve=False)
                self.assertEqual(result.status, "PENDING_REVIEW")
                self.assertFalse(db.bookings)
                self.assertEqual([name for name, _, _ in guard.log], ["search_flights"])

    def test_hybrid_api_replan_needs_new_approval(self):
        config, model, requests = self.make_model(
            lambda body, index: openai_response(plan=api_plan("VN122" if index == 1 else "VN124")))
        c, db, guard = self.make("contingency")
        seen = []
        def reviewer(plan):
            seen.append(plan.steps[0].args["flight"])
            return len(seen) == 1
        result = HybridFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config, reviewer=reviewer)
        self.assertEqual(seen, ["VN122", "VN124"])
        self.assertEqual(result.status, "HANDOFF")
        self.assertFalse(db.bookings)
        self.assertEqual(len(requests), 2)

    def test_harness_denies_model_invalid_flight(self):
        config, model, _ = self.make_model(lambda body, index: openai_response(plan=api_plan("QH118")))
        c, db, guard = self.make()
        result = PlanThenExecuteFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
        self.assertEqual(result.status, "HANDOFF")
        self.assertFalse(db.bookings)
        self.assertEqual(guard.log[-1][2]["status"], "denied")

    def test_invalid_schema_handoffs_before_write_and_keeps_usage(self):
        payload = api_plan()
        payload["steps"][0]["args"]["unexpected"] = "field"
        config, model, _ = self.make_model(lambda body, index: openai_response(plan=payload))
        c, db, guard = self.make()
        result = PlanThenExecuteFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
        self.assertEqual(result.status, "HANDOFF")
        self.assertEqual(result.error_category, "MODEL_SCHEMA_ERROR")
        self.assertEqual(result.total_tokens, 34)
        self.assertFalse(db.bookings)

    def test_auth_error_not_scored_as_expected_handoff_and_redacts_key(self):
        import httpx
        def response(body, index):
            return httpx.Response(401, json={"error": {"message": "Invalid contract-test-secret", "type": "invalid_api_key"}})
        config, model, requests = self.make_model(response)
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            results = run_benchmark(config=config, llm=model, scenarios=["fail"], patterns=["plan"])
        result = results[0]
        self.assertEqual(result.error_category, "API_AUTH_ERROR")
        self.assertEqual(aggregate_metrics(results)[0]["correct_behavior"], 0)
        self.assertIsNone(result.total_tokens)
        self.assertEqual(len(requests), 1, "Mặc định không retry SDK")
        with tempfile.TemporaryDirectory() as directory:
            export_results(results, Path(directory), capture.getvalue())
            evidence = "".join(p.read_text(encoding="utf-8-sig") for p in Path(directory).iterdir())
            self.assertNotIn(config.api_key, evidence)
            self.assertIn("[REDACTED]", evidence)

    def test_credit_exhausted_differs_from_rate_limit_and_never_books(self):
        import httpx
        from model_config import classify_model_error
        cases = [("credit_balance_exhausted", "insufficient_quota", "API_QUOTA_EXHAUSTED"),
                 (None, "insufficient_quota", "API_QUOTA_EXHAUSTED"),
                 ("rate_limit_exceeded", "rate_limit_error", "API_RATE_LIMIT")]
        for code, error_type, expected in cases:
            for agent_type in [PlanThenExecuteFlightAgent, ReActFlightAgent, HybridFlightAgent]:
                with self.subTest(code=code, agent=agent_type.__name__):
                    def response(body, index):
                        return httpx.Response(429, json={"error": {"message": "SDK contract test",
                                                                  "type": error_type, "code": code}})
                    config, model, requests = self.make_model(response)
                    c, db, guard = self.make()
                    result = agent_type(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
                    self.assertEqual(result.error_category, expected)
                    self.assertEqual(result.status, "HANDOFF")
                    self.assertFalse(result.is_done)
                    self.assertFalse(db.bookings)
                    self.assertEqual(len(requests), 1)
                    self.assertIsNone(result.total_tokens)
                    captured = next(t for t in result.trace if t.get("step") == "model_error")
                    self.assertEqual(captured["category"], expected)
        # Adapter có thể bọc exception; nhãn vẫn lấy từ cause SDK.
        from openai import RateLimitError
        original = RateLimitError("exhausted", response=httpx.Response(429, request=httpx.Request("POST", "https://test.invalid")),
                                  body={"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota"}})
        wrapped = RuntimeError("adapter")
        wrapped.__cause__ = original
        self.assertEqual(classify_model_error(wrapped), "API_QUOTA_EXHAUSTED")

    def test_missing_usage_not_zero_and_partial_usage_not_full_total(self):
        config, model, _ = self.make_model(lambda body, index: openai_response(plan=api_plan(), usage=False))
        c, db, guard = self.make()
        result = PlanThenExecuteFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
        self.assertEqual(result.status, "SUCCESS")
        self.assertIsNone(result.total_tokens)
        self.assertFalse(result.usage_complete)
        config, model, _ = self.make_model(lambda body, index: openai_response(plan=api_plan("VN122" if index == 1 else "VN124"), usage=index == 1))
        c, db, guard = self.make("contingency")
        result = HybridFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
        self.assertEqual(result.status, "SUCCESS")
        self.assertIsNone(result.total_tokens)
        self.assertEqual(result.reported_tokens["total_tokens"], 34)
        self.assertEqual(result.usage_calls, 1)

    def test_repeat_fresh_db_mixed_modes_and_exported_metrics(self):
        config, model, _ = self.make_model(lambda body, index: openai_response(plan=api_plan()))
        with contextlib.redirect_stdout(io.StringIO()):
            api_results = run_benchmark(config=config, llm=model, repeats=2, scenarios=["success"], patterns=["plan"])
            offline_results = run_benchmark(scenarios=["success"], patterns=["plan"])
        self.assertEqual([r.status for r in api_results], ["SUCCESS", "SUCCESS"])
        self.assertEqual([len(r.output_details["bookings"]) for r in api_results], [1, 1])
        self.assertEqual(len({r.run_id for r in api_results}), 2)
        metrics = aggregate_metrics(offline_results + api_results)
        self.assertEqual(len(metrics), 2)
        self.assertEqual({m["mode"] for m in metrics}, {"offline", "api"})
        self.assertIsNone(metrics[0]["avg_total_tokens"])
        self.assertEqual(metrics[1]["avg_total_tokens"], 34)

    def test_zero_model_invocation_has_zero_api_tokens(self):
        config, model, requests = self.make_model(lambda body, index: self.fail("Search lỗi: không gọi model"))
        c, db, guard = self.make("transient_error")
        result = PlanThenExecuteFlightAgent(c, db, guard).run(fake_mode=False, llm=model, model_config=config)
        self.assertEqual(result.status, "HANDOFF")
        self.assertEqual(result.llm_calls, 0)
        self.assertEqual(result.total_tokens, 0)
        self.assertTrue(result.usage_complete)
        self.assertFalse(requests)


if __name__ == "__main__":
    unittest.main()
