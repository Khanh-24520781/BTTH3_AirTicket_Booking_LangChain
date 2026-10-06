"""BTVN#3: ba mẫu Agent chạy qua LangChain/LangGraph, dùng chung Harness.

NOTE: Model offline mô phỏng quyết định, nhưng .invoke(), messages, tools và
callback đều chạy qua framework. Nhánh API nhận model từ model_config.py
qua main/evaluate với --mode api; offline vẫn mặc định và không gọi mạng.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Literal, TypedDict
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_model_call
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flight_harness import FlightConstraints, FlightDatabase, HarnessGuard, StopAgent, create_langchain_tools
from model_config import ModelConfig, classify_model_error, redact_secrets


class Step(BaseModel):
    # tên tool phải thuộc schema, không nhận method tùy ý của DB.
    tool: Literal["search_flights", "book_seat", "pay", "get_booking", "cancel_booking"]
    args: dict


class Plan(BaseModel):
    steps: list[Step] = Field(default_factory=list, max_length=10)


class PlanArguments(BaseModel):
    # NOTE API 4: schema đóng thay args: dict tự do. Mọi khóa đều khai báo;
    # tham số không dùng là chuỗi rỗng. Cách này còn tránh việc bộ chuyển
    # schema function của Gemini bỏ nhánh null trong anyOf của kiểu nullable.
    model_config = ConfigDict(extra="forbid")
    origin: str = Field(description="search_flights: sân bay đi; tool khác dùng chuỗi rỗng")
    destination: str = Field(description="search_flights: sân bay đến; tool khác dùng chuỗi rỗng")
    date: str = Field(description="search_flights: ngày YYYY-MM-DD; tool khác dùng chuỗi rỗng")
    flight: str = Field(description="book_seat: mã chuyến; tool khác dùng chuỗi rỗng")
    passenger_name: str = Field(description="book_seat: tên khách đúng yêu cầu; tool khác dùng chuỗi rỗng")
    code: str = Field(description="pay/get_booking/cancel_booking: $booking_code; tool khác dùng chuỗi rỗng")
    payment_method: str = Field(description="pay: wallet; tool khác dùng chuỗi rỗng")


class ApiStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["search_flights", "book_seat", "pay", "get_booking", "cancel_booking"]
    args: PlanArguments

    @model_validator(mode="after")
    def check_arguments(self):
        required = {"search_flights": {"origin", "destination", "date"},
                    "book_seat": {"flight", "passenger_name"}, "pay": {"code"},
                    "get_booking": {"code"}, "cancel_booking": {"code"}}[self.tool]
        allowed = required | ({"payment_method"} if self.tool == "pay" else set())
        supplied = {k for k, v in self.args.model_dump().items() if v != ""}
        if not required <= supplied or not supplied <= allowed:
            raise ValueError(f"Tham số không đúng cho tool {self.tool}.")
        return self


class ApiPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[ApiStep] = Field(max_length=10, description="Kế hoạch tuần tự; [] nếu không có chuyến hợp lệ")

    def to_plan(self) -> Plan:
        # Chuyển schema API về Plan chung TRƯỚC điểm duyệt. Executor và Harness
        # vẫn kiểm quyền/DB như offline, không thực thi trực tiếp output của LLM.
        return Plan(steps=[Step(tool=s.tool, args={k: v for k, v in s.args.model_dump().items() if v != ""}) for s in self.steps])


class StructuredOutputError(ValueError):
    """Có response model nhưng không có kế hoạch parse được."""


Reviewer = Callable[[Plan], bool]


@dataclass
class AgentExecutionResult:
    pattern_name: str
    scenario: str
    is_done: bool
    status: str
    llm_calls: int
    tool_calls: int
    output_details: dict
    trace: list[dict] = field(default_factory=list)
    model_mode: str = "offline"
    provider: str = "mock"
    model: str = "offline-policy-v1"
    run_id: str = ""
    repetition: int = 1
    duration_seconds: float = 0.0
    wall_time_seconds: float = 0.0
    review_seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    usage_calls: int = 0
    usage_complete: bool = False
    reported_tokens: dict = field(default_factory=dict)
    error_category: str | None = None
    error_message: str | None = None
    settings: dict = field(default_factory=dict)


class ModelCallCounter(BaseCallbackHandler):
    """Đếm invocation model thực tế, kể cả model giả lập."""

    def __init__(self, trace: list[dict]):
        self.calls = 0
        self.trace = trace
        self.tool_runs = {}
        self.usage_calls = 0
        self.tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.error_categories = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # không tăng số đếm ở nhánh nghiệp vụ nữa. Callback chỉ
        # chạy khi framework thực sự gọi model; không suy ra token/tiền từ nó.
        self.calls += 1
        self.trace.append({"step": "model_call", "call": self.calls})

    def on_llm_end(self, response, **kwargs):
        message = response.generations[0][0].message
        # NOTE API 14: .text chỉ lấy nội dung đọc được để in/chụp terminal.
        # Không sửa AIMessage gốc: Gemini cần giữ thought signature bên trong
        # lịch sử hội thoại khi ReAct gửi kết quả tool cho lần suy luận tiếp.
        self.trace.append({"step": "model_response", "content": message.text,
                           "tool_calls": getattr(message, "tool_calls", [])})
        # NOTE API 5: chỉ cộng usage SDK trả về, không ước lượng từ độ dài text
        # hay số calls. Thiếu usage ở một call => tổng cả lượt là N/A; vẫn lưu
        # phần usage đã nhận để người đọc phân biệt tổng đủ và tổng chưa đủ.
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            raw = (getattr(message, "response_metadata", {}).get("token_usage")
                   or (response.llm_output or {}).get("token_usage") or {})
            usage = {"input_tokens": raw.get("prompt_tokens"), "output_tokens": raw.get("completion_tokens"),
                     "total_tokens": raw.get("total_tokens")}
        if all(type(usage.get(k)) is int and usage[k] >= 0 for k in self.tokens):
            self.usage_calls += 1
            for key in self.tokens:
                self.tokens[key] += usage[key]
            self.trace.append({"step": "model_usage", "usage": {k: usage[k] for k in self.tokens}})

    def on_llm_error(self, error, **kwargs):
        category = classify_model_error(error)
        self.error_categories.append(category)
        # Không chép request/header API hoặc repr SDK vào trace lỗi.
        self.trace.append({"step": "model_error", "category": category, "exception_type": type(error).__name__})

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.tool_runs[run_id] = (serialized.get("name"), kwargs.get("inputs") or {})

    def on_tool_end(self, output, *, run_id, **kwargs):
        name, args = self.tool_runs.pop(run_id, ("unknown", {}))
        result = json.loads(output.content) if isinstance(output, ToolMessage) else output
        self.trace.append({"step": "action", "tool": name, "args": args, "result": result})


def _mock_plan(flights: list[dict], constraints: FlightConstraints) -> Plan:
    """Chính sách của MODEL GIẢ LẬP; Harness vẫn kiểm lại từng đề xuất."""
    candidates = [f for f in flights if constraints.is_ok(f) and f.get("seats_available", 0) > 0]
    if not candidates:
        return Plan()
    return Plan(steps=[
        Step(tool="book_seat", args={"flight": candidates[0]["flight"], "passenger_name": constraints.passenger_name}),
        Step(tool="pay", args={"code": "$booking_code", "payment_method": "wallet"}),
        Step(tool="get_booking", args={"code": "$booking_code"}),
    ])


class OfflineReActModel(BaseChatModel):
    """Model offline đọc observations và trả AIMessage/tool_calls qua LangChain."""

    constraints: FlightConstraints

    @property
    def _llm_type(self) -> str:
        return "offline-flight-react"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        observations = [(m.name, json.loads(m.content)) for m in messages if isinstance(m, ToolMessage)]
        c = self.constraints
        search_args = {"origin": c.origin, "destination": c.destination, "date": c.date}
        call_id = f"call_{sum(isinstance(m, AIMessage) for m in messages) + 1}"

        def call(name, args, content):
            return AIMessage(content=content, tool_calls=[{"name": name, "args": args, "id": call_id}])

        def choose_next():
            searches = [r for name, r in observations if name == "search_flights" and r.get("status") == "ok"]
            attempted = {r.get("flight") for name, r in observations if name == "book_seat"}
            flights = searches[-1]["flights"] if searches else []
            plan = _mock_plan([f for f in flights if f.get("flight") not in attempted], c)
            return call("book_seat", plan.steps[0].args, "Giữ chỗ chuyến phù hợp.") if plan.steps else AIMessage(content="Không còn chuyến phù hợp để đặt.")

        # fake model phản ứng theo ToolMessage vừa nhận; không đọc
        # DB riêng hay viết thẳng vào DB. Runtime create_agent chạy vòng lặp.
        if not observations:
            answer = call("search_flights", search_args, "Tìm chuyến theo yêu cầu.")
        else:
            name, result = observations[-1]
            if name == "search_flights":
                if result.get("status") == "ok":
                    answer = choose_next()
                elif result.get("retryable") and sum(n == name for n, _ in observations) < 2:
                    answer = call(name, search_args, "Thử lại lỗi tạm thời một lần.")
                else:
                    answer = AIMessage(content="Tìm kiếm gặp lỗi dịch vụ; cần người xử lý tiếp.")
            elif name == "book_seat":
                if result.get("status") == "ok":
                    answer = call("pay", {"code": result["code"], "payment_method": "wallet"}, "Thanh toán booking vừa giữ.")
                elif result.get("error_code") == "SOLD_OUT":
                    answer = choose_next()
                else:
                    answer = AIMessage(content="Giữ chỗ không thành công; cần người xử lý tiếp.")
            elif name in {"pay", "get_booking"} and result.get("status") == "ok" and result.get("paid"):
                answer = AIMessage(content=f"Đã thanh toán chuyến {result['flight']}, giá {result['price']:,} VND.")
            else:
                answer = AIMessage(content="Chưa xác nhận hoàn thành; cần kiểm tra trạng thái booking.")
        return ChatResult(generations=[ChatGeneration(message=answer)])


class BaseFlightAgent:
    pattern_name = "Base"

    def __init__(self, constraints: FlightConstraints, db: FlightDatabase, guard: HarnessGuard):
        self.constraints, self.db, self.guard = constraints, db, guard
        self.tools = {t.name: t for t in create_langchain_tools(guard)}

    def _start(self, fake_mode: bool, llm, model_config: ModelConfig | None = None):
        if not fake_mode and llm is None:
            raise ValueError("fake_mode=False yêu cầu truyền llm=model đã cấu hình.")
        if model_config and (model_config.mode == "offline") != fake_mode:
            raise ValueError("Cấu hình mode và fake_mode không khớp; không được tự fallback sang offline.")
        self.guard.reset()
        self.trace: list[dict] = []
        self.counter = ModelCallCounter(self.trace)
        self.config = model_config
        self.model_mode = model_config.mode if model_config else ("offline" if fake_mode else "provided_model")
        self.provider = model_config.provider if model_config else ("mock" if fake_mode else "injected")
        self.model_name = model_config.model if model_config else ("offline-policy-v1" if fake_mode else
                          str(getattr(llm, "model_name", None) or getattr(llm, "model", "injected-model")))
        self.started_at = perf_counter()
        self.review_seconds = 0.0
        self.run_id = uuid4().hex
        self.error_category = self.error_message = None

    def _record_error(self, exc):
        self.error_category = classify_model_error(exc)
        self.error_message = self._redact(f"{type(exc).__name__}: {exc}")[:1500]
        return self.error_message

    def _redact(self, value):
        return self.config.redact(value) if self.config else redact_secrets(value)

    def _tool(self, name: str, args: dict) -> dict:
        # Các executor cũng dùng wrapper .invoke(), không bỏ qua LangChain tools.
        if name not in self.tools:
            return self.guard.execute_tool(name, args)
        try:
            return self.tools[name].invoke(args)
        except Exception as exc:
            result = {"status": "error", "error_code": "TOOL_SCHEMA_ERROR", "retryable": False, "error": str(exc)}
            self.guard.log.append((name, deepcopy(args), result))
            return result

    def _search(self) -> dict:
        c = self.constraints
        args = {"origin": c.origin, "destination": c.destination, "date": c.date}
        result = self._tool("search_flights", args)
        self.trace.append({"step": "search", "phase": "PHASE_1_DISCOVERY", "tool": "search_flights", "args": args, "result": result})
        return result

    def _plan(self, flights: list, fake_mode: bool, llm) -> Plan:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Lập toàn bộ kế hoạch đặt đúng một vé. Dùng book_seat, pay, get_booking. "
             "Chỉ chọn chuyến đạt TẤT CẢ ràng buộc và còn ghế. Điền hành khách đúng yêu cầu. "
             "Mã chưa biết dùng $booking_code. Không có chuyến phù hợp thì steps rỗng. {format_instructions}"),
            ("human", "Yêu cầu: {goal}\nDữ liệu tool: {flights}"),
        ])
        # offline cũng gọi một ChatModel thật qua chain, sau đó
        # parse JSON bằng schema. LLM do người dùng truyền dùng structured output.
        api_schema = self.config is not None and self.config.mode == "api"
        schema = ApiPlan if api_schema else Plan
        parser = PydanticOutputParser(pydantic_object=schema)
        if fake_mode:
            model = GenericFakeChatModel(messages=iter([AIMessage(content=_mock_plan(flights, self.constraints).model_dump_json())]))
            planner = prompt | model | parser
        else:
            options = {"method": self.config.plan_method, "include_raw": True} if api_schema else {}
            planner = prompt | llm.with_structured_output(schema, **options)
        result = planner.invoke({"goal": self.constraints.to_prompt(), "flights": json.dumps(flights, ensure_ascii=False),
                                 "format_instructions": parser.get_format_instructions()}, config={"callbacks": [self.counter]})
        if api_schema:
            # include_raw giữ usage ngay cả khi parse lỗi. Không retry bằng mock
            # hoặc chạy plan chưa parse được. Truncated/refusal cũng phải bàn giao.
            if result.get("parsing_error"):
                raise result["parsing_error"]
            if result.get("parsed") is None:
                raise StructuredOutputError("Model không trả được kế hoạch có cấu trúc (có thể từ chối hoặc hết giới hạn token).")
            parsed = result["parsed"]
            plan = (parsed if isinstance(parsed, ApiPlan) else ApiPlan.model_validate(parsed)).to_plan()
        else:
            plan = result if isinstance(result, Plan) else Plan.model_validate(result)
        self.trace.append({"step": "plan_generated", "phase": "PHASE_2_MACRO_PLAN", "plan": plan.model_dump()})
        return plan

    def _review(self, plan: Plan, auto_approve: bool, reviewer: Reviewer | None) -> str:
        # reviewer thấy bản sao đúng kế hoạch sắp thực thi. Nếu
        # sửa nội dung trong callback, phải lập/duyệt lại thay vì chạy âm thầm.
        if reviewer is None:
            decision = "approved" if auto_approve is True else "pending"
        else:
            displayed = plan.model_copy(deep=True)
            before_review = perf_counter()
            try:
                approved = reviewer(displayed) is True
            finally:
                self.review_seconds += perf_counter() - before_review
            decision = "approved" if approved and displayed == plan else "rejected"
        self.trace.append({"step": "plan_review", "phase": "HUMAN_REVIEW", "status": decision, "plan": plan.model_dump()})
        return decision

    def _execute_steps(self, plan: Plan, retry_transient: bool = False):
        # placeholder chỉ lấy mã từ lần book trong chính plan này,
        # không lấy một booking ngẫu nhiên từ log hoặc phiên trước.
        booking_code = ""
        for index, step in enumerate(plan.steps, 1):
            args = deepcopy(step.args)
            if args.get("code") == "$booking_code":
                args["code"] = booking_code
            result = self._tool(step.tool, args)
            self.trace.append({"step": f"execute_{index}", "phase": "PHASE_3_EXECUTION", "tool": step.tool, "args": args, "result": result})
            # Hybrid chỉ retry lỗi có cờ retryable, một lần, với đúng tham số
            # đã duyệt. Không retry thanh toán mơ hồ vì có nguy cơ trả hai lần.
            if retry_transient and step.tool != "pay" and result.get("retryable") is True:
                self.trace.append({"phase": "REACTIVE_RECOVERY", "action": f"retry_{step.tool}"})
                result = self._tool(step.tool, args)
                self.trace.append({"step": f"execute_{index}_retry", "phase": "PHASE_3_EXECUTION", "tool": step.tool, "args": args, "result": result})
            if result.get("status") != "ok":
                return step.tool, args, result
            if step.tool == "book_seat":
                booking_code = result["code"]
        return None

    def _finish(self, answer: str = "", reason: str = "", pending: bool = False) -> AgentExecutionResult:
        done = self.guard.is_done()
        hallucinations = self.guard.check_hallucination(answer) if answer else []
        # kiểm ảo giác được nối vào đầu ra Agent. DB có thể đã
        # hoàn thành nhưng lời model sai; giữ trạng thái DB và bàn giao để xác minh.
        if hallucinations:
            reason = f"Câu trả lời có dữ kiện chưa được tool xác nhận: {hallucinations}. Nhờ người xác minh booking hiện tại."
        reason = reason or self.guard.stop_reason or ""
        if not done and not reason:
            failures = [r for _, _, r in self.guard.log if r.get("status") != "ok"]
            if failures:
                reason = f"Tool chưa hoàn thành: {failures[-1]}. Nhờ người kiểm tra và quyết định bước tiếp theo."
        status = "PENDING_REVIEW" if pending else ("SUCCESS" if done and not reason else "HANDOFF")
        output = {"result": "DONE" if status == "SUCCESS" else status,
                  "bookings": deepcopy(list(self.db.bookings.values())), "model_mode": self.model_mode,
                  "model_answer": answer, "hallucinations": hallucinations}
        if status != "SUCCESS":
            output["handoff"] = self.guard.handoff(reason or None)
        wall_time = perf_counter() - self.started_at
        complete_usage = self.model_mode == "api" and self.counter.calls == self.counter.usage_calls
        totals = self.counter.tokens if complete_usage else {k: None for k in self.counter.tokens}
        # NOTE API 6: duration bỏ thời gian nhập quyết định tại callback duyệt;
        # benchmark tự duyệt nên có thể so sánh trực tiếp. Wall time vẫn lưu riêng.
        return AgentExecutionResult(self.pattern_name, self.db.scenario, done, status,
                                    self.counter.calls, len(self.guard.log), self._redact(output),
                                    self._redact(deepcopy(self.trace)), self.model_mode,
                                    provider=self.provider, model=self.model_name, run_id=self.run_id,
                                    duration_seconds=round(max(0.0, wall_time - self.review_seconds), 6),
                                    wall_time_seconds=round(wall_time, 6), review_seconds=round(self.review_seconds, 6),
                                    **totals, usage_calls=self.counter.usage_calls, usage_complete=complete_usage,
                                    reported_tokens=deepcopy(self.counter.tokens) if self.counter.usage_calls else {},
                                    error_category=self.error_category or next(iter(self.counter.error_categories), None),
                                    error_message=self.error_message,
                                    settings=self._redact(self.config.public_settings()) if self.config else {})


class ReActFlightAgent(BaseFlightAgent):
    pattern_name = "ReAct"

    def __init__(self, constraints, db, guard):
        super().__init__(constraints, db, guard)
        self.max_steps = 10

    def run(self, fake_mode: bool = True, llm=None, model_config: ModelConfig | None = None) -> AgentExecutionResult:
        self._start(fake_mode, llm, model_config)
        model = OfflineReActModel(constraints=self.constraints) if fake_mode else llm

        @wrap_model_call
        def stop_on_loop(request, handler):
            if self.guard.stop_reason:
                raise StopAgent(self.guard.stop_reason)
            return handler(request)

        # create_agent điều phối vòng model -> tools -> model.
        # Middleware cưỡng chế max_steps, thay cho biến ngân sách chưa được dùng.
        answer, reason = "", ""
        try:
            self.runtime = create_agent(model=model, tools=list(self.tools.values()),
                                        system_prompt="Hỗ trợ đặt đúng một vé và thanh toán mock. Tuân thủ tất cả ràng buộc. "
                                        "Dùng mã booking thực tế do book_seat trả về. SOLD_OUT: chọn chuyến hợp lệ khác; "
                                        "retry lỗi tạm thời tối đa một lần, không retry pay. Lỗi không phục hồi: dừng và bàn giao. "
                                        "Chỉ báo hoàn thành khi tool xác nhận paid. Không tự tạo mã chuyến hay giá tiền.",
                                        middleware=[stop_on_loop, ModelCallLimitMiddleware(run_limit=self.max_steps, exit_behavior="error")])
            result = self.runtime.invoke({"messages": [{"role": "user", "content": self.constraints.to_prompt()}]},
                                         config={"callbacks": [self.counter], "recursion_limit": self.max_steps * 4 + 10})
            # text bỏ các block reasoning/provider metadata; chỉ kiểm lời trả
            # cuối cùng người dùng đọc được, không str() toàn bộ list content.
            answer = result["messages"][-1].text
        except Exception as exc:
            reason = f"Agent dừng an toàn: {self._record_error(exc)}. Nhờ người kiểm tra trạng thái hiện tại."
        return self._finish(answer, reason)


class PlanThenExecuteFlightAgent(BaseFlightAgent):
    pattern_name = "Plan-then-Execute"

    def prepare_plan(self, fake_mode: bool = True, llm=None, model_config: ModelConfig | None = None) -> Plan:
        self._start(fake_mode, llm, model_config)
        self._plan_error = ""
        self._consumed = False
        self._prepared_json = None
        try:
            search = self._search()
            if search.get("status") != "ok":
                # lỗi search không phải danh sách chuyến rỗng.
                # Mẫu kế hoạch cố định bàn giao ngay, không gọi model với data lỗi.
                self._plan_error = f"Tìm kiếm lỗi: {search.get('error', search.get('reason'))}. Bạn muốn thử lại dịch vụ?"
                plan = Plan()
            else:
                plan = self._plan(search["flights"], fake_mode, llm)
        except Exception as exc:
            self._plan_error = f"Không lập được kế hoạch: {self._record_error(exc)}. Nhờ người kiểm tra."
            plan = Plan()
        self._prepared_json = plan.model_dump_json()
        return plan

    def execute_plan(self, plan: Plan, approved: bool = False) -> AgentExecutionResult:
        # lưu fingerprint kế hoạch đã chuẩn bị. Không tìm kiếm /
        # sinh lại plan khi duyệt; không cho chạy plan đã sửa hoặc chạy hai lần.
        if not hasattr(self, "counter"):
            raise ValueError("Gọi prepare_plan() trước execute_plan().")
        if self._consumed or plan.model_dump_json() != self._prepared_json:
            return self._finish(reason="Kế hoạch đã thay đổi hoặc đã thực thi; cần chuẩn bị và duyệt lại.")
        if not plan.steps:
            return self._finish(reason=self._plan_error)
        if approved is not True:
            return self._finish(reason="Kế hoạch đang chờ người phê duyệt; chưa thực hiện book/pay.", pending=True)
        self._consumed = True
        failure = self._execute_steps(plan)
        reason = f"Bước {failure[0]} thất bại: {failure[2]}. Nhờ người xử lý tiếp." if failure else ""
        return self._finish(reason=reason)

    def run(self, fake_mode: bool = True, auto_approve: bool = True, llm=None, reviewer: Reviewer | None = None,
            model_config: ModelConfig | None = None) -> AgentExecutionResult:
        plan = self.prepare_plan(fake_mode, llm, model_config)
        if not plan.steps:
            return self.execute_plan(plan)
        try:
            decision = self._review(plan, auto_approve, reviewer)
        except Exception as exc:
            return self._finish(reason=f"Duyệt kế hoạch lỗi: {exc}. Chưa thực thi.")
        if decision == "rejected":
            self._consumed = True
            return self._finish(reason="Người duyệt từ chối kế hoạch. Bạn muốn thay đổi yêu cầu nào?")
        return self.execute_plan(plan, approved=decision == "approved")


class HybridState(TypedDict, total=False):
    flights: list[dict]
    plan: Plan
    attempted: list[str]
    replans: int
    search_attempts: int
    next_node: str
    reason: str
    pending: bool


class HybridFlightAgent(BaseFlightAgent):
    pattern_name = "Lai (Hybrid)"

    def __init__(self, constraints, db, guard):
        super().__init__(constraints, db, guard)
        self.max_replans = 2

    def run(self, fake_mode: bool = True, auto_approve: bool = True, llm=None, reviewer: Reviewer | None = None,
            model_config: ModelConfig | None = None) -> AgentExecutionResult:
        self._start(fake_mode, llm, model_config)

        def discovery(state):
            result = self._search()
            attempts = state.get("search_attempts", 0) + 1
            if result.get("status") == "ok":
                return {"flights": result["flights"], "search_attempts": attempts, "next_node": "plan"}
            if result.get("retryable") is True and attempts < 2:
                self.trace.append({"phase": "REACTIVE_RECOVERY", "action": "retry_search", "reason": result.get("error_code")})
                return {"search_attempts": attempts, "next_node": "discovery"}
            return {"reason": f"Dịch vụ tìm kiếm lỗi: {result}. Bạn muốn thử lại sau?", "next_node": "finish"}

        def planning(state):
            available = [f for f in state["flights"] if f.get("flight") not in state.get("attempted", [])]
            plan = self._plan(available, fake_mode, llm)
            return {"plan": plan, "next_node": "review" if plan.steps else "finish"}

        def review(state):
            # mỗi plan mới, kể cả plan đổi chuyến, phải đi qua
            # điểm duyệt. Pending/từ chối kết thúc graph trước mọi thao tác ghi.
            decision = self._review(state["plan"], auto_approve, reviewer)
            if decision == "approved":
                return {"next_node": "execute"}
            return {"next_node": "finish", "pending": decision == "pending",
                    "reason": "Kế hoạch đang chờ duyệt." if decision == "pending" else "Người duyệt từ chối kế hoạch; chưa thực thi plan này."}

        def execute(state):
            failure = self._execute_steps(state["plan"], retry_transient=True)
            if failure is None:
                return {"next_node": "finish"}
            name, args, result = failure
            # Chỉ đổi chuyến khi giữ chỗ bị SOLD_OUT. Lỗi thanh toán phải bàn
            # giao cùng booking held, không tạo thêm vé hay thanh toán lại.
            if name == "book_seat" and result.get("error_code") == "SOLD_OUT" and state.get("replans", 0) < self.max_replans:
                return {"attempted": state.get("attempted", []) + [args["flight"]], "next_node": "replan"}
            return {"next_node": "finish", "reason": f"Bước {name} thất bại hoặc hết ngân sách replan: {result}. Nhờ người xử lý tiếp."}

        def replan(state):
            count = state.get("replans", 0) + 1
            self.trace.append({"phase": "ADAPTIVE_REPLANNING", "replan": count,
                               "action": "Lập kế hoạch mới, bỏ chuyến đã hết chỗ.", "attempted": state["attempted"]})
            return {"replans": count, "next_node": "plan"}

        # LangGraph biểu diễn các pha và điều kiện chuyển pha.
        # Giới hạn search retry + max_replans thực sự cưỡng chế vòng phục hồi.
        builder = StateGraph(HybridState)
        for name, node in [("discovery", discovery), ("plan", planning), ("review", review),
                           ("execute", execute), ("replan", replan), ("finish", lambda state: {})]:
            builder.add_node(name, node)
        builder.add_edge(START, "discovery")
        routes = {name: name for name in ["discovery", "plan", "review", "execute", "replan", "finish"]}
        for name in ["discovery", "plan", "review", "execute", "replan"]:
            builder.add_conditional_edges(name, lambda state: state["next_node"], routes)
        builder.add_edge("finish", END)
        self.runtime = builder.compile()
        try:
            state = self.runtime.invoke({"attempted": [], "replans": 0, "search_attempts": 0},
                                        config={"recursion_limit": 12 + 5 * max(0, self.max_replans)})
            return self._finish(reason=state.get("reason", ""), pending=state.get("pending", False))
        except Exception as exc:
            return self._finish(reason=f"Hybrid dừng an toàn: {self._record_error(exc)}. Nhờ người kiểm tra booking hiện tại.")
