"""BTVN#3: CLI cho ba mẫu Agent, duyệt kế hoạch và bốn failure modes."""

from __future__ import annotations

import json
import sys
import argparse

from agents import HybridFlightAgent, Plan, PlanThenExecuteFlightAgent, ReActFlightAgent
from evaluate_agents import AGENT_TYPES, SCENARIOS, print_summary_table, run_benchmark
from flight_harness import FlightConstraints, FlightDatabase, HarnessGuard
from model_config import (ModelConfig, ModelConfigurationError, add_model_arguments,
                          create_model, execution_options, load_model_config)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def print_banner(config: ModelConfig | None = None):
    config = config or ModelConfig()
    print("=" * 80)
    print("BTVN#3: AGENT ĐẶT VÉ BẰNG LANGCHAIN / LANGGRAPH + HARNESS")
    print(config.redact(f"Mode: {config.mode} | Provider: {config.provider} | Model: {config.model}"))
    print("Đặt chỗ và thanh toán đều là mock; nhánh API dùng LLM thật qua provider đã cấu hình.")
    print("=" * 80)


def choose_scenario() -> str:
    print("\n1. success         - Có chuyến VN122 hợp lệ")
    print("2. fail            - Không có chuyến đạt cả giờ và giá")
    print("3. contingency     - VN122 mất ghế sau search, VN124 là dự phòng")
    print("4. transient_error - Search lỗi tạm thời ở lần đầu")
    mapping = {"1": "success", "2": "fail", "3": "contingency", "4": "transient_error"}
    while True:
        choice = input("Chọn kịch bản (1-4) [1]: ").strip() or "1"
        if choice in mapping:
            return mapping[choice]


def review_plan(plan: Plan) -> bool:
    # NOTE SỬA CLI 1: callback được Agent gọi sau khi lập kế hoạch. CLI không
    # search / sinh plan lần thứ hai; đối tượng được xem chính là plan được chạy.
    print("\n--- KẾ HOẠCH CHỜ DUYỆT ---")
    for index, step in enumerate(plan.steps, 1):
        print(f"  {index}. {step.tool}({step.args})")
    while True:
        answer = input("Phê duyệt kế hoạch này? (y/n): ").strip().lower()
        if answer in {"y", "n"}:
            return answer == "y"


def print_result(result):
    print("\n--- TRACE THỰC THI ---")
    for entry in result.trace:
        step, phase = entry.get("step"), entry.get("phase")
        if step == "model_call":
            print(f"[Model invocation #{entry['call']}]")
        elif step == "model_response":
            if entry["content"]:
                print(f"  [Model response] {entry['content']}")
            # NOTE API 15: ReAct có thể chỉ trả tool calls, không có text.
            # In đề xuất của model riêng để phân biệt với tool đã được Harness chạy.
            for call in entry.get("tool_calls", []):
                print(f"  [Model tool call] {call['name']}({call['args']})")
        elif step == "model_usage":
            print(f"  [Usage] {entry['usage']}")
        elif step == "model_error":
            print(f"  [Model error] {entry['category']}")
        elif step == "plan_review":
            print(f"[Human Review] {entry['status']}")
        elif step == "plan_generated":
            print(f"[Plan] {entry['plan']}")
        elif "tool" in entry:
            print(f"  [Tool] {entry['tool']}({entry.get('args', {})}) -> {entry['result'].get('status')}")
        elif phase in {"ADAPTIVE_REPLANNING", "REACTIVE_RECOVERY"}:
            print(f"[{phase}] {entry.get('action')}")
    print("\n--- KẾT QUẢ HARNESS ---")
    print(f"Trạng thái: {result.status} | is_done: {result.is_done}")
    print(f"Model calls ({result.model_mode}): {result.llm_calls} | Tool calls: {result.tool_calls}")
    print(f"Provider: {result.provider} | Model: {result.model} | Run: {result.run_id}")
    print(f"Thời gian xử lý: {result.duration_seconds:.3f}s | Chờ duyệt: {result.review_seconds:.3f}s")
    print(f"Token vào/ra/tổng: {result.input_tokens if result.input_tokens is not None else 'N/A'} / "
          f"{result.output_tokens if result.output_tokens is not None else 'N/A'} / "
          f"{result.total_tokens if result.total_tokens is not None else 'N/A'}")
    if result.error_category:
        print(f"Lỗi: {result.error_category} | {result.error_message or ''}")
    print(json.dumps(result.output_details, ensure_ascii=False, indent=2))


def _create_agent(agent_type, scenario=None):
    scenario = scenario or choose_scenario()
    constraints = FlightConstraints()
    db = FlightDatabase(scenario)
    print(f"\nRàng buộc: {constraints.to_prompt()}")
    return agent_type(constraints, db, HarnessGuard(constraints, db))


def run_react_interactive(config=None, llm=None):
    print_result(_create_agent(ReActFlightAgent).run(**execution_options(config, llm)))


def run_plan_then_execute_interactive(config=None, llm=None):
    # NOTE SỬA CLI 2: lỗi search bàn giao ngay. Nếu người từ chối hoặc chưa
    # duyệt, Agent không thể đi vào bước book/pay dù CLI có gọi run().
    print_result(_create_agent(PlanThenExecuteFlightAgent).run(
        auto_approve=False, reviewer=review_plan, **execution_options(config, llm)))


def run_hybrid_interactive(config=None, llm=None):
    # NOTE SỬA CLI 3: Hybrid cũng có duyệt; khi replan đổi chuyến, callback
    # hiện kế hoạch mới và yêu cầu một quyết định phê duyệt mới.
    print_result(_create_agent(HybridFlightAgent).run(
        auto_approve=False, reviewer=review_plan, **execution_options(config, llm)))


def run_benchmark_cli(config=None, llm=None):
    print_summary_table(run_benchmark(config=config, llm=llm))


def test_failure_modes():
    print("Demo Harness dùng mock offline để tái lập các tình huống lỗi.")
    constraints = FlightConstraints()
    args = {"origin": constraints.origin, "destination": constraints.destination, "date": constraints.date}
    guard = HarnessGuard(constraints, FlightDatabase("success"))
    print("\n1. Goal Drift: cố đặt chuyến buổi chiều QH118")
    print(guard.execute_tool("book_seat", {"flight": "QH118"}))
    print("\n2. Loop: lần gọi thứ tư bị từ chối và đặt cờ dừng Agent")
    guard.reset()
    for index in range(4):
        print(f"Lần {index + 1}: {guard.execute_tool('search_flights', args)}")
    print("\n3. Hallucination: kiểm cả mã chuyến và giá tiền")
    guard.reset()
    guard.execute_tool("search_flights", args)
    for answer in ["VN999 giá 1,200,000 VND.", "VN122 giá 1,200,000 VND."]:
        print(f"{answer} -> {guard.check_hallucination(answer)}")

    # NOTE SỬA CLI 4: bổ sung demo thứ tư. {} được chuẩn hóa thành lỗi có
    # retryable=True; Hybrid thử lại và vẫn phải nghiệm thu booking bằng code.
    class EmptyOnceDatabase(FlightDatabase):
        def search_flights(self, origin, destination, date):
            if not getattr(self, "empty_returned", False):
                self.empty_returned = True
                return {}
            return super().search_flights(origin, destination, date)

    print("\n4. State Corruption: search trả {} một lần, Hybrid phục hồi")
    db = EmptyOnceDatabase()
    print_result(HybridFlightAgent(constraints, db, HarnessGuard(constraints, db)).run())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--pattern", choices=list(AGENT_TYPES), help="Chạy một mẫu trực tiếp để demo/chụp terminal")
    parser.add_argument("--scenario", choices=SCENARIOS, default="success")
    parser.add_argument("--auto-approve", action="store_true", help="Tự duyệt kế hoạch cho demo tool mock")
    args = parser.parse_args(argv)
    try:
        config = load_model_config(args)
        llm = create_model(config)
    except ModelConfigurationError as exc:
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        return 2
    if args.check_config:
        print(json.dumps(config.redact(config.public_settings()), ensure_ascii=False, indent=2))
        print("Cấu hình/SDK hợp lệ; chưa gọi API, chưa kiểm tra key hoặc quyền truy cập model tại provider.")
        return 0
    print_banner(config)
    # NOTE API 10: cùng menu/cùng điểm duyệt ở hai mode. --auto-approve chỉ là
    # tùy chọn demo; mặc định Plan/Hybrid vẫn yêu cầu duyệt đúng từng kế hoạch.
    if args.pattern:
        agent = _create_agent(AGENT_TYPES[args.pattern], args.scenario)
        options = execution_options(config, llm)
        if args.pattern != "react":
            options.update(auto_approve=args.auto_approve, reviewer=None if args.auto_approve else review_plan)
        result = agent.run(**options)
        print_result(result)
        return 1 if result.error_category else 0
    actions = {"1": lambda: run_react_interactive(config, llm),
               "2": lambda: run_plan_then_execute_interactive(config, llm),
               "3": lambda: run_hybrid_interactive(config, llm),
               "4": lambda: run_benchmark_cli(config, llm), "5": test_failure_modes}
    while True:
        print("\n1. ReAct\n2. Plan-then-Execute (duyệt plan)\n3. Hybrid (duyệt từng plan)")
        print(f"4. Benchmark {config.mode} (tự duyệt)\n5. Bốn failure modes (mock)\n0. Thoát")
        choice = input("Chọn (0-5): ").strip()
        if choice == "0":
            break
        action = actions.get(choice)
        if action:
            action()
        else:
            print("Lựa chọn không hợp lệ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
