"""Benchmark offline/API, giữ cùng dữ liệu mock và xuất số liệu để viết báo cáo."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from datetime import datetime, timezone
import importlib.metadata
import platform

from agents import AgentExecutionResult, HybridFlightAgent, PlanThenExecuteFlightAgent, ReActFlightAgent
from flight_harness import FlightConstraints, FlightDatabase, HarnessGuard
from model_config import (ModelConfig, ModelConfigurationError, add_model_arguments,
                          create_model, execution_options, load_model_config)

SCENARIOS = ("success", "fail", "contingency", "transient_error")
AGENT_TYPES = {"react": ReActFlightAgent, "plan": PlanThenExecuteFlightAgent, "hybrid": HybridFlightAgent}


class TeeOutput:
    """Hiện tiến độ ngay trong terminal đồng thời giữ log để xuất báo cáo."""
    def __init__(self, terminal, capture):
        self.terminal, self.capture = terminal, capture

    def write(self, text):
        self.capture.write(text)
        written = self.terminal.write(text)
        self.terminal.flush()
        return written

    def flush(self):
        self.terminal.flush()

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def run_benchmark(*, config: ModelConfig | None = None, llm=None, repeats: int = 1,
                  scenarios=None, patterns=None) -> list[AgentExecutionResult]:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats phải là số nguyên >= 1.")
    selected_scenarios = list(SCENARIOS if scenarios is None else scenarios)
    selected_patterns = list(AGENT_TYPES if patterns is None else patterns)
    if not selected_scenarios or any(s not in SCENARIOS for s in selected_scenarios):
        raise ValueError("Kịch bản không hợp lệ.")
    if not selected_patterns or any(p not in AGENT_TYPES for p in selected_patterns):
        raise ValueError("Mẫu Agent không hợp lệ.")
    config = config or ModelConfig()
    if config.mode == "api" and llm is None:
        llm = create_model(config)
    results = []
    print(f"BENCHMARK: {config.mode} | provider={config.provider} | model={config.model}")
    print(f"{len(selected_scenarios) * len(selected_patterns) * repeats} lượt Agent; tự duyệt kế hoạch; tool đặt vé/thanh toán mock.")
    print("Model calls là invocation framework; token lấy từ usage provider; thiếu usage = N/A.")
    for repetition in range(1, repeats + 1):
        for scenario in selected_scenarios:
            print(f"\n>>> LẦN {repetition}/{repeats} | KỊCH BẢN: {scenario} <<<")
            for pattern in selected_patterns:
                # NOTE API 7: DB/guard mới cho TỪNG lượt, kể cả giữa các repeat.
                # API lỗi vẫn lưu lượt HANDOFF cùng category; không bỏ khỏi mẫu thử.
                constraints = FlightConstraints()
                db = FlightDatabase(scenario)
                agent = AGENT_TYPES[pattern](constraints, db, HarnessGuard(constraints, db))
                result = agent.run(**execution_options(config, llm))
                result.repetition = repetition
                results.append(result)
                print(f"{result.pattern_name:<22} | {result.status:<8} | Model: {result.llm_calls} | Tool: {result.tool_calls} "
                      f"| {result.duration_seconds:.3f}s | Token: {result.total_tokens if result.total_tokens is not None else 'N/A'}")
                if result.error_category:
                    print(f"  Lỗi: {result.error_category}: {result.error_message or ''}")
    return results


def aggregate_metrics(results: list[AgentExecutionResult], *, by_scenario=False) -> list[dict]:
    metrics = []
    constraints = FlightConstraints()
    # NOTE API 8: không trộn model offline với model API trong cùng trung bình.
    # Khi so sánh file JSON, mỗi mode/provider/model/pattern có một dòng riêng.
    def group_key(r):
        return (r.model_mode, r.provider, r.model, r.pattern_name, r.scenario if by_scenario else "all")
    for key in dict.fromkeys(group_key(r) for r in results):
        mode, provider, model, pattern, scenario = key
        rows = [r for r in results if group_key(r) == key]
        feasible = [r for r in rows if r.scenario != "fail"]
        recovery = [r for r in rows if r.scenario in {"contingency", "transient_error"}]
        correct = sum(not r.error_category and
                      ((r.scenario == "fail" and r.status == "HANDOFF" and not r.is_done)
                       or (r.scenario != "fail" and r.status == "SUCCESS" and r.is_done)) for r in rows)
        # NOTE SỬA ĐÁNH GIÁ 2: chỉ số tính từ kết quả, không gán sẵn 'tốt/kém'.
        # Vi phạm dưới đây có phạm vi hữu hạn: booking hiệu lực sai ràng buộc.
        violations = sum(1 for r in rows for b in r.output_details.get("bookings", [])
                         if b.get("booking_status") in {"held", "confirmed"}
                         and (not constraints.is_ok(b) or b.get("passenger") != constraints.passenger_name))
        known_usage = [r for r in rows if r.usage_complete]
        error_counts = {category: sum(r.error_category == category for r in rows)
                        for category in dict.fromkeys(r.error_category for r in rows if r.error_category)}
        metrics.append({"mode": mode, "provider": provider, "model": model, "pattern": pattern, "scenario": scenario,
                        "correct_behavior": correct, "scenario_count": len(rows),
                        "feasible_success": sum(r.status == "SUCCESS" and r.is_done for r in feasible),
                        "feasible_count": len(feasible),
                        "recovery_success": sum(r.status == "SUCCESS" and r.is_done for r in recovery),
                        "recovery_count": len(recovery), "avg_model_calls": sum(r.llm_calls for r in rows) / len(rows),
                        "avg_tool_calls": sum(r.tool_calls for r in rows) / len(rows),
                        "invalid_active_bookings": violations,
                        "avg_duration_seconds": mean(r.duration_seconds for r in rows),
                        "std_duration_seconds": pstdev(r.duration_seconds for r in rows),
                        "usage_complete_runs": len(known_usage),
                        "avg_input_tokens": mean(r.input_tokens for r in known_usage) if known_usage else None,
                        "avg_output_tokens": mean(r.output_tokens for r in known_usage) if known_usage else None,
                        "avg_total_tokens": mean(r.total_tokens for r in known_usage) if known_usage else None,
                        "runs_with_errors": sum(bool(r.error_category) for r in rows),
                        "api_error_runs": sum(bool(r.error_category and r.error_category.startswith("API_")) for r in rows),
                        "schema_error_runs": sum(r.error_category == "MODEL_SCHEMA_ERROR" for r in rows),
                        "error_counts": error_counts})
    return metrics


def print_summary_table(results: list[AgentExecutionResult]):
    for identity in dict.fromkeys((r.model_mode, r.provider, r.model) for r in results):
        selected = [r for r in results if (r.model_mode, r.provider, r.model) == identity]
        print(f"\n--- {identity[0]} | {identity[1]} | {identity[2]} ---")
        print("Mẫu                   | Kịch bản        | Lần | Trạng thái | is_done | Model | Tool")
        print("-" * 92)
        for r in selected:
            print(f"{r.pattern_name:<22} | {r.scenario:<15} | {r.repetition:<3} | {r.status:<10} | {str(r.is_done):<7} | {r.llm_calls:<5} | {r.tool_calls}")
        print("\nMẫu                   | Đúng kỳ vọng | Đặt được* | Phục hồi | Model TB | Tool TB | Booking sai")
        for m in aggregate_metrics(selected):
            print(f"{m['pattern']:<22} | {m['correct_behavior']}/{m['scenario_count']:<10} | "
                  f"{m['feasible_success']}/{m['feasible_count']:<7} | {m['recovery_success']}/{m['recovery_count']:<6} | "
                  f"{m['avg_model_calls']:<8.2f} | {m['avg_tool_calls']:<7.2f} | {m['invalid_active_bookings']}")
        print("\nMẫu                   | Giây TB ± SD     | Token TB** | Usage đủ | Lượt lỗi")
        for m in aggregate_metrics(selected):
            token = f"{m['avg_total_tokens']:.1f}" if m['avg_total_tokens'] is not None else "N/A"
            print(f"{m['pattern']:<22} | {m['avg_duration_seconds']:.3f} ± {m['std_duration_seconds']:.3f} "
                  f"| {token:<10} | {m['usage_complete_runs']}/{m['scenario_count']:<7} | {m['runs_with_errors']}")
    print("(*) Chỉ tính các kịch bản có lời giải. Case fail kỳ vọng HANDOFF.")
    print("(**) Chỉ tính lượt có usage đủ; xem số lượt ở cột Usage đủ. Không suy token/tiền từ calls.")
    print("SD là độ lệch chuẩn tổng thể; JSON by_scenario tách thống kê từng kịch bản khi lặp.")
    print("Không phát hiện booking sai trong bộ thử này không chứng minh an toàn tuyệt đối.")


def export_results(results: list[AgentExecutionResult], output_dir: Path, log: str):
    # NOTE SỬA ĐÁNH GIÁ 3: xuất trace và số liệu mới để báo cáo có thể tái lập.
    # Đặt trong after_fixes để giữ lại bằng chứng của lần rà soát trước sửa.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "benchmark_metrics.json").write_text(json.dumps(aggregate_metrics(results), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "benchmark_output.txt").write_text(log, encoding="utf-8")
    (output_dir / "benchmark_metrics_by_scenario.json").write_text(
        json.dumps(aggregate_metrics(results, by_scenario=True), ensure_ascii=False, indent=2), encoding="utf-8")
    # NOTE API 9: CSV dễ đưa vào bảng báo cáo; manifest chỉ chứa cấu hình công
    # khai và phiên bản, không xuất .env hoặc api_key.
    columns = ["run_id", "model_mode", "provider", "model", "pattern_name", "scenario", "repetition",
               "status", "is_done", "llm_calls", "tool_calls", "duration_seconds", "review_seconds",
               "input_tokens", "output_tokens", "total_tokens", "usage_complete", "error_category"]
    with (output_dir / "benchmark_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: getattr(r, k) for k in columns} for r in results)
    versions = {}
    for package in ["langchain", "langchain-core", "langgraph", "pydantic", "langchain-openai", "langchain-google-genai"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {"exported_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "packages": versions, "run_count": len(results), "auto_approve": True,
                "settings": list({json.dumps(r.settings, sort_keys=True): r.settings for r in results}.values())}
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS)
    parser.add_argument("--patterns", nargs="+", choices=list(AGENT_TYPES))
    parser.add_argument("--compare-json", nargs="+", type=Path, help="So sánh các benchmark_results.json đã có, không gọi API")
    parser.add_argument("--output-dir", type=Path, help="Thư mục xuất JSON/trace/log benchmark")
    args = parser.parse_args(argv)
    config, llm = None, None
    try:
        if args.compare_json:
            results = [AgentExecutionResult(**row) for path in args.compare_json
                       for row in json.loads(path.read_text(encoding="utf-8"))]
        else:
            config = load_model_config(args)
            llm = create_model(config)
            if args.check_config:
                print(json.dumps(config.redact(config.public_settings()), ensure_ascii=False, indent=2))
                print("Cấu hình/SDK hợp lệ; chưa gọi API, chưa kiểm tra key hoặc quyền truy cập model tại provider.")
                return 0
            if args.repeats < 1:
                parser.error("--repeats phải >= 1")
    except (ModelConfigurationError, OSError, ValueError, TypeError) as exc:
        print(f"Lỗi cấu hình/dữ liệu: {exc}", file=sys.stderr)
        return 2
    capture = io.StringIO()
    with contextlib.redirect_stdout(TeeOutput(sys.stdout, capture)):
        if not args.compare_json:
            results = run_benchmark(config=config, llm=llm, repeats=args.repeats,
                                    scenarios=args.scenarios, patterns=args.patterns)
        print_summary_table(results)
    if args.output_dir:
        export_results(results, args.output_dir, capture.getvalue())
        print(f"\nĐã xuất: {args.output_dir.resolve()}")
    return 1 if not args.compare_json and any(r.error_category for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
